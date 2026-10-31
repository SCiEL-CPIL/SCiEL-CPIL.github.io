import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

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

def _fft_score(U):
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
    
    # Determine FFT size - need L >= 2M+1 for no circular aliasing
    L = 1
    while L < 2 * M + 1:
        L *= 2
        
    # Create indicator vector
    A = np.zeros(L, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # Sumset: compute via FFT
    F_A = np.fft.fft(A)
    F_S = F_A * F_A
    S = np.fft.ifft(F_S).real
    
    # Count non-zero elements in S[0 : 2*M+1]
    sum_count = np.sum(S[:2*M+1] > 0.5)
    
    # Diffset: use autocorrelation via FFT
    # Autocorrelation R[k] = sum_j A[j] A[j+k] for k in [-M, M]
    # This equals convolution of A with reversed A
    A_rev = A[::-1]
    F_AR = np.fft.fft(A_rev)
    F_D = F_A * F_AR
    D = np.fft.ifft(F_D).real
    
    # The convolution D has the peak at index 0 (circular)
    # Differences d = a-b correspond to indices: d=0 at index 0, d>0 at index L-d, d<0 at index |d|
    # So for d in [-M, M]: index = d if d>=0 else L+d
    # Let's build the count
    diff_count = 0
    # d = 0
    if D[0] > 0.5:
        diff_count += 1
    # d > 0: index = L - d
    for d in range(1, M + 1):
        if D[L - d] > 0.5:
            diff_count += 1
        # d < 0: index = |d| = -d
        if D[-d] > 0.5:
            diff_count += 1
    
    denom = math.log(2 * M + 1, 2)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count, 2) / denom

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences are unique)."""
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

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
    best_s = _fft_score(best_U)
    start = time.time()
    rng = random.Random(42)
    n = len(best_U)
    
    # Precompute a set for O(1) membership checks
    best_set = set(best_U)
    
    while time.time() - start < time_budget and _remaining() > 2.0:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(1, n))
        rng.shuffle(indices)
        
        for i in indices:
            if _remaining() < 2.0:
                break
            orig = best_U[i]
            # Try a range of shifts - both small and moderate
            # Scale shifts with the span to explore more effectively
            span = best_U[-1] if best_U else 1
            max_shift = max(5, span // 10)
            
            for delta in range(-max_shift, max_shift + 1):
                if delta == 0:
                    continue
                new_val = orig + delta
                if new_val < 0 or new_val in best_set:
                    continue
                new_U = list(best_U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                new_set = set(new_U)
                
                s = _fft_score(new_U)
                if s > best_s + 1e-10:
                    best_s = s
                    best_U = new_U
                    best_set = new_set
                    improved = True
                    break
        
        if not improved:
            # Try adding or removing elements
            if _remaining() > 3.0:
                # Try adding a new element
                if rng.random() < 0.5 and n < 20:
                    # Try values in a reasonable range
                    max_val = best_U[-1] if best_U else 10
                    for _ in range(5):
                        new_val = rng.randint(1, max_val + max_val // 5)
                        if new_val not in best_set:
                            new_U = list(best_U) + [new_val]
                            new_U = sorted(new_U)
                            new_set = set(new_U)
                            s = _fft_score(new_U)
                            if s > best_s:
                                best_s = s
                                best_U = new_U
                                best_set = new_set
                                n = len(best_U)
                                break
                
                # Try removing an element
                elif n > 3:
                    idx = rng.randint(1, n-1)
                    new_U = list(best_U)
                    new_U.pop(idx)
                    new_set = set(new_U)
                    s = _fft_score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U
                        best_set = new_set
                        n = len(best_U)
            
            # Random perturbation if stuck
            if rng.random() < 0.3:
                idx = rng.randint(1, n-1)
                span = best_U[-1] if best_U else 10
                new_val = max(1, best_U[idx] + rng.randint(-span//5, span//5))
                if new_val not in best_set:
                    new_U = list(best_U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    new_set = set(new_U)
                    s = _fft_score(new_U)
                    if s > best_s:
                        best_s = s
                        best_U = new_U
                        best_set = new_set

    return best_U, best_s

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    candidates = []
    
    # 1. Known optimal Golomb rulers for small n
    for n in range(3, 13):
        g = _gen_golomb_optimal(n)
        if g:
            candidates.append(g)
            
    # 2. Add random variations of small Golomb rulers
    for n in range(3, 10):
        base = _gen_golomb_optimal(n)
        if base:
            for _ in range(8):
                perturbed = list(base)
                if len(perturbed) > 2:
                    # Try swapping two elements
                    if random.random() < 0.5:
                        i, j = random.sample(range(1, len(perturbed)), 2)
                        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                    else:
                        # Try perturbing one element
                        i = random.randint(1, len(perturbed) - 1)
                        perturbed[i] = max(1, perturbed[i] + random.randint(-3, 3))
                    perturbed = sorted(perturbed)
                    if len(set(perturbed)) == len(perturbed) and perturbed[0] == 0:
                        candidates.append(perturbed)
    
    # 3. Generate some random sets with varying sizes
    for n in [5, 7, 9, 11, 13]:
        for _ in range(3):
            # Generate a random set with 0
            vals = set([0])
            max_val = random.randint(20, 80)
            while len(vals) < n:
                vals.add(random.randint(1, max_val))
            candidates.append(sorted(list(vals)))

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen and len(key) >= 2 and key[0] == 0:
            seen.add(key)
            unique.append(list(c))

    # Score all initial candidates using FFT
    scored = [(_fft_score(s), s) for s in unique]
    scored.sort(key=lambda x: -x[0])

    if not scored:
        return [0, 1]

    best_U = scored[0][1]
    best_s = scored[0][0]

    # Local search on top candidates
    top_candidates = [s for _, s in scored[:15]]
    
    # Allocate time based on remaining budget
    total_time = _remaining()
    per_set_time = max(3.0, total_time / len(top_candidates) - 2.0)
    
    for s in top_candidates:
        if _remaining() < 5.0:
            break
        result, rs = _local_search(s, time_budget=per_set_time)
        if rs > best_s:
            best_s = rs
            best_U = result

    # Final polish on the best found with more iterations
    if _remaining() > 8.0:
        polished, ps = _local_search(best_U, time_budget=min(20.0, _remaining() - 3.0))
        if ps > best_s:
            best_s = ps
            best_U = polished

    # Ensure validity
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U
