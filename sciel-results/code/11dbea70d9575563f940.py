import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_fft(U):
    """
    Fast score calculation using FFT.
    U: list of non-negative integers.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    # We need L >= 2*M + 1 for linear convolution/correlation
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
    # Threshold 0.1 to avoid floating point noise
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Use autocorrelation
    F_R = F_A * np.conj(F_A)
    R = np.fft.ifft(F_R).real
    
    # The differences a-b range from -M to M.
    # In the circular convolution/autocorrelation of size L, the difference d corresponds to index d % L.
    # Since L > 2M, indices 0..M correspond to differences 0..M, and L-M..L-1 correspond to -M..-1.
    # We need the count of non-zero entries in R for these indices.
    diff_indices = list(range(0, M+1)) + list(range(L-M, L))
    diff_count = np.sum(R[diff_indices] > 0.1)
    
    if sum_count == 0 or diff_count == 0:
        return 0.0

    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _score_small(U):
    """Exact O(n^2) score for small sets."""
    U = sorted(set(int(x) for x in U if x >= 0))
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(len(diffset) / len(sumset), 2) / denom

def _get_score(U):
    if max(U) < 1000:
        return _score_small(U)
    return _score_fft(U)

def _generate_initial_candidates():
    """
    Generate diverse initial candidates.
    1. Random sparse sets.
    2. Random dense sets (higher density).
    3. Structured sets (arithmetic progressions, etc.)
    """
    candidates = []
    
    # 1. Random sparse sets
    for n in [10, 20, 50, 100]:
        for trial in range(3):
            U = [0]
            max_val = n * n * 2
            while len(U) < n:
                x = random.randint(1, max_val)
                if x not in U:
                    U.append(x)
            U.sort()
            candidates.append(U)
            
    # 2. Random dense sets (smaller range)
    for n in [10, 20, 50]:
        for trial in range(3):
            U = [0]
            max_val = n * 5
            while len(U) < n:
                x = random.randint(1, max_val)
                if x not in U:
                    U.append(x)
            U.sort()
            candidates.append(U)
            
    # 3. Arithmetic Progression
    for n in [10, 20, 50]:
        d = random.randint(1, 5)
        U = list(range(0, n * d, d))
        candidates.append(U)
        
    # 4. Geometric Progression
    for n in [10, 20]:
        r = random.randint(2, 4)
        U = [0]
        cur = 1
        while len(U) < n:
            U.append(cur)
            cur *= r
        candidates.append(U)
        
    return candidates

def _simulated_annealing(U_initial, time_budget):
    """
    Simulated Annealing to optimize the set.
    Moves:
    1. Insert a random element.
    2. Delete a random element (not 0).
    3. Shift a random element (not 0) by a small delta.
    """
    U = sorted(list(U_initial))
    n = len(U)
    current_score = _get_score(U)
    best_U = list(U)
    best_score = current_score
    
    rng = random.Random(123)
    start_time = time.time()
    
    # Initial temperature
    T = 0.1
    cooling_rate = 0.99
    
    while time.time() - start_time < time_budget and _remaining() > 2.0:
        # Generate a move
        move_type = rng.choice(['insert', 'delete', 'shift'])
        
        if move_type == 'insert' and len(U) < 200:
            # Insert a random element
            max_val = max(U) * 2 + 10
            new_val = rng.randint(1, max_val)
            if new_val not in U:
                U.append(new_val)
                U.sort()
                new_score = _get_score(U)
                
                if new_score > current_score:
                    current_score = new_score
                    best_U = list(U)
                    best_score = current_score
                else:
                    # Metropolis criterion
                    delta = new_score - current_score
                    if rng.random() < math.exp(delta / T):
                        current_score = new_score
                    else:
                        # Revert
                        U.pop(U.index(new_val))
                        U.sort()
        
        elif move_type == 'delete' and len(U) > 2:
            # Delete a random element (not 0)
            i = rng.randint(1, len(U) - 1)
            val_to_remove = U[i]
            U.pop(i)
            U.sort()
            new_score = _get_score(U)
            
            if new_score > current_score:
                current_score = new_score
                best_U = list(U)
                best_score = current_score
            else:
                delta = new_score - current_score
                if rng.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Revert
                    U.append(val_to_remove)
                    U.sort()
                    
        elif move_type == 'shift' and len(U) > 1:
            # Shift a random element (not 0)
            i = rng.randint(1, len(U) - 1)
            orig_val = U[i]
            delta_shift = rng.randint(-10, 10)
            new_val = orig_val + delta_shift
            
            if new_val >= 0 and new_val not in U:
                U[i] = new_val
                U.sort()
                new_score = _get_score(U)
                
                if new_score > current_score:
                    current_score = new_score
                    best_U = list(U)
                    best_score = current_score
                else:
                    delta = new_score - current_score
                    if rng.random() < math.exp(delta / T):
                        current_score = new_score
                    else:
                        # Revert
                        U[i] = orig_val
                        U.sort()
        
        # Cool down
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
            
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1]
    best_score = 0.0
    
    # 1. Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # 2. Score all candidates
    scored = []
    for U in candidates:
        if _remaining() < 5.0:
            break
        s = _get_score(U)
        scored.append((s, U))
        
    if not scored:
        return [0, 1]
        
    scored.sort(key=lambda x: -x[0])
    best_s, best_U = scored[0]
    
    # 3. Refine top candidates using Simulated Annealing
    # Allocate time based on remaining budget
    # We have ~135s available. Let's use 40s per top candidate, up to 3 candidates.
    time_per_refine = min(40.0, _remaining() / 3.0)
    
    for s, U in scored[:3]:
        if _remaining() < time_per_refine + 2.0:
            break
            
        res_U, res_s = _simulated_annealing(U, time_per_refine)
        
        if res_s > best_s:
            best_s = res_s
            best_U = res_U
            
    # 4. Final Polish: Try to reduce max(U) by shifting the entire set (except 0) down?
    # Or just a final local search on the best one.
    if _remaining() > 10.0:
        # Quick local search on the best one found
        U = list(best_U)
        U.sort()
        current_s = best_s
        rng = random.Random(999)
        
        # Simple hill climbing for a few seconds
        polish_start = time.time()
        while time.time() - polish_start < 5.0 and _remaining() > 2.0:
            i = rng.randint(1, len(U) - 1)
            orig_val = U[i]
            for new_val in range(max(0, orig_val - 5), orig_val + 6):
                if new_val == orig_val:
                    continue
                if new_val == 0 and i != 0:
                    continue
                if new_val in U and U.index(new_val) != i:
                    continue
                    
                U[i] = new_val
                U.sort()
                new_s = _get_score(U)
                
                if new_s > current_s + 1e-10:
                    current_s = new_s
                    best_U = list(U)
                    best_s = current_s
                    break
                else:
                    U[i] = orig_val
                    U.sort()
            
    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U

