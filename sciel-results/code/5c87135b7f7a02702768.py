import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_fast(U):
    """
    Fast score calculation using FFT.
    """
    U = sorted(set(int(x) for x in U if x >= 0))
    n = len(U)
    if n < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    
    # Determine FFT size
    # Sumset range: [0, 2M]. Length 2M+1.
    # Diffset range: [-M, M]. Length 2M+1.
    # To avoid circular convolution, we need L >= 2M+1.
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
    sum_count = np.sum(S[:2*M+1] > 0.1)
    
    # Diffset: Autocorrelation
    # Using scipy if available is faster, but let's try to stick to numpy for portability
    # or use the property that diffset support is related to correlation.
    # Correlation of A with A.
    # R[k] = sum A[i] A[i+k]
    # This is the same as convolution of A and reverse(A) shifted.
    # Let's just use scipy.signal.correlate if we can import it safely.
    try:
        from scipy import signal
        autocorr = signal.correlate(A, A, mode='full')
        # Peak is at L-1
        # We want indices corresponding to diffs -M to M
        # In full correlation of length L, peak is at L-1.
        # Index L-1 + d corresponds to diff d.
        # So we look at [L-1-M, L-1+M]
        diff_count = np.sum(autocorr[L-1-M : L-1+M+1] > 0.1)
    except ImportError:
        # Fallback to numpy
        B = A[::-1]
        F_B = np.fft.fft(B)
        F_D = F_A * F_B
        D = np.fft.ifft(F_D).real
        # The convolution of A and B (reversed) gives counts.
        # The peak (diff 0) is at index L-1?
        # Let's check: A=[1,1,0,0], B=[0,0,1,1]. Conv=[0,1,2,1,0,0,0].
        # Peak at 2. L=4. L-1=3.
        # It seems the peak is at L-1.
        # So we check D[L-1-M : L-1+M+1]
        diff_count = np.sum(D[L-1-M : L-1+M+1] > 0.1)
        
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
        13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 72, 80],
        14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 60, 72, 80, 94]
    }
    if n in optimal:
        return list(optimal[n])
    return None

def _local_search(U, time_budget=10.0):
    """
    Fast local search: try moving each element by small integers.
    """
    U = sorted(list(U))
    if 0 not in U:
        U.insert(0, 0)
    best_U = list(U)
    best_s = _score_fast(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    # Deltas to try
    deltas = [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -15, 15]
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            for delta in deltas:
                new_val = orig + delta
                if new_val < 0:
                    continue
                if new_val in best_U:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                
                s = _score_fast(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    improved = True
                    break
        if not improved:
            # Random perturbation
            if rng.random() < 0.3:
                idx = rng.randint(1, n-1)
                new_val = max(1, best_U[idx] + rng.randint(-20, 20))
                if new_val not in best_U:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    s = _score_fast(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers
    for n in range(3, 15):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add a few random variations
    for n in range(3, 12):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(3):
                perturbed = list(base)
                if len(perturbed) > 2:
                    i, j = random.sample(range(1, len(perturbed)), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    perturbed = sorted(perturbed)
                    # Check if still Golomb? Not strictly necessary, but good for diversity
                    candidates.append(perturbed)

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates
    scored = [(_score_fast(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    # We have ~150s.
    # Score calculation is fast.
    # Let's allocate time to top 5 candidates.
    top_candidates = [s for _, s in scored[:5]]
    
    # Give more time to the best candidate
    time_per_candidate = 15.0
    
    for s in top_candidates:
        if _remaining() < time_per_candidate + 5.0:
            break
        result, rs = _local_search(s, time_budget=time_per_candidate)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found
    if _remaining() > 10.0:
        polished, ps = _local_search(best_U, time_budget=min(20.0, _remaining() - 2.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
