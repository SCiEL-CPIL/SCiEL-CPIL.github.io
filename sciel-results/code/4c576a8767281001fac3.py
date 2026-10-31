import math
import time
import random
import numpy as np
from scipy import signal

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _fft_score(U):
    """
    Fast score calculation using FFT.
    Returns the score value.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: A * A
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    # Count non-zero elements in S[0 : 2*M+1]
    # Using a small epsilon to handle floating point errors
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Use SciPy correlate for robustness
    # autocorr[k] corresponds to sum A[i] A[i+k] which is related to differences
    # Specifically, we want the number of distinct values in {u-v | u,v in U}
    # This is the support of the autocorrelation function (shifted).
    autocorr = signal.correlate(A, A, mode='full')
    # The differences range from -M to M.
    # The center of the 'full' correlation is at index L-1.
    # So we look at indices L-1-M to L-1+M.
    diff_count = np.sum(autocorr[L-1-M : L-1+M+1] > 0.1)
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _gen_golomb_optimal(n):
    """Return known optimal Golomb rulers for small n."""
    optimal = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 55],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 67],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 80],
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _local_search(U, time_budget=10.0):
    """
    Fast local search: try moving each element by small integers.
    Uses FFT scoring for speed.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    
    def get_score(U_list):
        if max(U_list) < 1000:
            # Exact score for small sets to avoid FFT overhead/errors
            sumset = set()
            diffset = set()
            for a in U_list:
                for b in U_list:
                    sumset.add(a + b)
                    diffset.add(a - b)
            M = max(U_list)
            if M == 0: return 0.0
            denom = math.log(2 * M + 1, 2)
            if denom == 0: return 0.0
            return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom
        else:
            return _fft_score(U_list)

    best_s = get_score(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            max_delta = max(5, int(math.log2(max(best_U) + 1)))
            deltas = list(range(-max_delta, max_delta + 1))
            if 0 in deltas:
                deltas.remove(0)
            rng.shuffle(deltas)
            
            for delta in deltas:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                s = get_score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            # Occasional random jump
            if rng.random() < 0.2:
                idx = rng.randint(1, n-1)
                new_val = max(1, best_U[idx] + rng.randint(-20, 20))
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = get_score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U

    return best_U, best_s

def _simulated_annealing(U, time_budget=30.0):
    """
    Simulated Annealing with a 'repair' mechanism.
    Strategy:
    1. Randomly select an element (not 0).
    2. Propose a new value, biased towards reducing the max U (shrinking diameter).
    3. If the new value is invalid (duplicate or negative), repair it by finding the nearest valid value or rejecting.
    4. Accept based on Metropolis criterion.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    
    current_U = list(U)
    current_s = _fft_score(current_U)
    best_U = list(current_U)
    best_s = current_s
    
    T = 0.1 # Initial temperature
    T_min = 1e-4
    cooling = 0.999
    
    rng = random.Random(123)
    start_time = time.time()
    
    n = len(current_U)
    M = max(current_U)
    
    while time.time() - start_time < time_budget and T > T_min and _remaining() > 5.0:
        # Select an element to move (index 0 is fixed at 0)
        i = rng.randint(1, n - 1)
        old_val = current_U[i]
        
        # Proposal: Move towards center (reduce max) or random small jump
        # Bias towards smaller values to reduce denominator
        # Let's try to move it to a random position in [1, M] but with a bias
        # Or simply a random walk with a drift towards 0.
        
        # Random walk with drift
        step = rng.randint(-10, 10)
        # Add drift towards 0 to shrink the set
        drift = -int(old_val * 0.05)
        new_val = old_val + step + drift
        
        # Ensure new_val is not 0 (unless i==0 which we skipped) and >= 0
        if new_val < 1:
            new_val = 1
            
        # Check validity
        if new_val in current_U:
            # Repair: try nearby values
            found = False
            for offset in range(1, 10):
                if new_val + offset not in current_U:
                    new_val = new_val + offset
                    found = True
                    break
                if new_val - offset >= 1 and new_val - offset not in current_U:
                    new_val = new_val - offset
                    found = True
                    break
            if not found:
                # Skip this move if we can't repair easily
                T *= cooling
                continue
        
        # Apply move
        current_U[i] = new_val
        current_U = sorted(current_U)
        
        # Score
        new_s = _fft_score(current_U)
        
        # Acceptance
        delta_s = new_s - current_s
        if delta_s > 0 or rng.random() < math.exp(delta_s / T):
            current_s = new_s
            if new_s > best_s:
                best_s = new_s
                best_U = list(current_U)
        else:
            # Revert
            current_U[i] = old_val
            current_U = sorted(current_U)
            
        T *= cooling
        
    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Random initial sets
    # Try a few different sizes
    for n in [5, 6, 7, 8, 10]:
        U = [0]
        for _ in range(n - 1):
            U.append(random.randint(1, 100))
        U = sorted(set(U))
        if len(U) >= 2:
            candidates.append(U)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    def fast_score(U):
        if max(U) < 1000:
             # Use exact for small
             sumset = set()
             diffset = set()
             for a in U:
                 for b in U:
                     sumset.add(a + b)
                     diffset.add(a - b)
             M = max(U)
             if M == 0: return 0.0
             denom = math.log(2 * M + 1, 2)
             if denom == 0: return 0.0
             return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom
        return _fft_score(U)

    scored = [(fast_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Run Simulated Annealing on top candidates
    # SA is better at escaping local minima and can aggressively shrink the set
    top_candidates = [s for _, s in scored[:5]]
    
    for s in top_candidates:
        if _remaining() < 10.0:
            break
        # Allocate more time to SA as it's the main driver
        sa_budget = min(20.0, _remaining() - 5.0)
        if sa_budget < 5.0:
            continue
        result, rs = _simulated_annealing(s, time_budget=sa_budget)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish with local search
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(15.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
