"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _score_of(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Use FFT to compute |U+U| and |U-U|
    # For U+U: convolution of indicator with itself
    # For U-U: convolution of indicator with reversed indicator
    size = 2 * max_u + 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Indicator vector for U
    ind = np.zeros(fft_size)
    for u in U:
        ind[u] = 1.0
    
    # U+U: convolution of ind with ind
    fft_ind = np.fft.fft(ind)
    sum_conv = np.fft.ifft(fft_ind * fft_ind).real
    # U-U: convolution of ind with reversed ind
    # U-U = {a-b : a,b in U} = {-(b-a)} so |U-U| = |U+(-U)|
    # ind_neg[i] = ind[-i mod fft_size] = ind[fft_size - i] for i>0
    ind_neg = np.zeros(fft_size)
    ind_neg[0] = 1.0
    for u in U:
        ind_neg[fft_size - u] = 1.0
    fft_ind_neg = np.fft.fft(ind_neg)
    diff_conv = np.fft.ifft(fft_ind * fft_ind_neg).real
    
    # Count non-zero entries
    # U+U values range from 0 to 2*max_u
    sum_count = np.sum(sum_conv[:size] > 0.5)
    # U-U values range from -max_u to max_u, mapped to 0..2*max_u in convolution
    # Actually diff_conv[i] counts pairs (a,b) with a-b = i - fft_size/2... 
    # Let me think more carefully.
    # ind[k]=1 if k in U. ind_neg[j]=1 if -j mod fft_size in U, i.e., j = fft_size - u for u in U.
    # (ind * ind_neg)[m] = sum_k ind[k] * ind_neg[m-k] = sum_k ind[k] * ind_neg[m-k]
    # ind_neg[m-k] = 1 iff (m-k) mod fft_size in {-u mod fft_size for u in U}
    # = 1 iff exists u in U such that (m-k) mod fft_size = (fft_size - u) mod fft_size
    # = 1 iff exists u in U such that k - u ≡ m (mod fft_size)
    # = 1 iff exists u in U such that k = u + m (mod fft_size)
    # So (ind * ind_neg)[m] = |{u in U : u + m mod fft_size in U}|
    # This is the autocorrelation-like quantity. For m=0: |U|.
    # For the difference set: U-U = {a-b : a,b in U}
    # Number of representations of d = a-b is |{(a,b) in U×U : a-b=d}|
    # = sum_a ind[a] * ind[a-d] (for d>=0) or similar
    # Actually, let's just use a simpler approach for diff.
    
    # Simpler: |U-U| = number of distinct values a-b for a,b in U
    # = number of d in [-max_u, max_u] such that exists a,b in U with a-b=d
    # = number of d such that U ∩ (U+d) is non-empty
    # This is the support of the "cross-correlation" 
    # Let me just compute it directly for smaller sets or use the FFT result properly.
    
    # The convolution ind * ind_neg gives: for each shift s, count of pairs (x,y) in U×U with x-y = s (mod fft_size)
    # Since all values are in [0, max_u], differences are in [-max_u, max_u], no wraparound if fft_size > 2*max_u
    # diff_conv[s] for s in [0, fft_size): represents difference = s if s <= max_u, or s - fft_size if s > fft_size/2
    # Actually: diff_conv[m] = |{(a,b) in U² : a - b ≡ m (mod fft_size)}|
    # For m in [0, max_u]: this is |{(a,b): a-b=m}|
    # For m in [fft_size-max_u, fft_size): this is |{(a,b): a-b=m-fft_size}| = |{(a,b): b-a=fft_size-m}|
    
    # Count distinct differences
    diff_count = 0
    # Positive differences: m from 1 to max_u
    for m in range(1, max_u + 1):
        if diff_conv[m] > 0.5:
            diff_count += 1
    # Negative differences: m from fft_size-max_u to fft_size-1
    for m in range(fft_size - max_u, fft_size):
        if diff_conv[m] > 0.5:
            diff_count += 1
    # Zero difference: always present
    diff_count += 1
    
    # sum_count: distinct values in U+U, range [0, 2*max_u]
    # sum_conv[m] = |{(a,b) in U² : a+b=m}| for m in [0, 2*max_u]
    sum_count = np.sum(sum_conv[0:size] > 0.5)
    
    if sum_count == 0:
        return 0.0
    
    import math
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    score = 1.0 + log_ratio / log_denom
    return score


def _score_of_fast(U):
    """Faster score using direct computation for smaller sets."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Direct computation for smaller sets
    if n <= 200:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        import math
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_of(U)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with a seed set
    # Strategy: use sets with large gaps to maximize |U-U|/|U+U|
    # A geometric progression-like set can work well
    best_U = [0, 1, 3]
    best_score = _score_of_fast(best_U)
    
    # Try various structured sets first
    candidate_sets = []
    
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        for n in range(2, 20):
            s = [0]
            val = 1
            for i in range(n):
                val *= base
                s.append(val)
                if val > 100000:
                    break
            if len(s) >= 2:
                candidate_sets.append(s)
    
    # Arithmetic progressions with gaps
    for gap in [2, 3, 5, 7, 10]:
        for n in range(2, 30):
            s = [i * gap for i in range(n)]
            if s[-1] <= 100000:
                candidate_sets.append(s)
    
    # Mixed: start with dense, then add sparse elements
    for base_n in [5, 10, 15]:
        for gap in [3, 5, 8]:
            s = list(range(base_n))
            val = base_n * gap
            while val <= 50000:
                s.append(val)
                val += gap * 2
            candidate_sets.append(s)
    
    # Evaluate all candidates
    for s in candidate_sets:
        if time.time() > deadline:
            break
        s = sorted(set(s))
        if 0 not in s:
            s.append(0)
            s = sorted(s)
        sc = _score_of_fast(s)
        if sc > best_score:
            best_score = sc
            best_U = s[:]
    
    # Local search: try adding/removing/shifting elements
    U = best_U[:]
    current_score = best_score
    
    # Define mutation operations
    def try_add(U, x):
        if x < 0 or x in U or x > 200000:
            return None
        new_U = U + [x]
        return sorted(new_U)
    
    def try_remove(U, idx):
        if idx < 0 or idx >= len(U) or U[idx] == 0:
            return None
        new_U = U[:idx] + U[idx+1:]
        if len(new_U) < 2:
            return None
        return new_U
    
    def try_shift(U, idx, delta):
        if idx == 0:  # Can't move 0
            return None
        new_val = U[idx] + delta
        if new_val < 0 or new_val in U or new_val > 200000:
            return None
        new_U = U[:]
        new_U[idx] = new_val
        return sorted(new_U)
    
    # Iterative local search
    improved = True
    max_iterations = 50000
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        U_sorted = sorted(U)
        n = len(U_sorted)
        
        # Try random additions
        for _ in range(20):
            if time.time() > deadline:
                break
            if np.random.random() < 0.5 and U_sorted:
                max_u = U_sorted[-1]
                # Try values near existing elements or at larger scales
                candidates = []
                for u in U_sorted:
                    if np.random.random() < 0.3:
                        candidates.append(u + np.random.randint(1, 10))
                        candidates.append(u - np.random.randint(1, 10))
                # Try larger values
                if U_sorted:
                    candidates.append(np.random.randint(1, max(2, max_u * 2)))
                    candidates.append(max_u + np.random.randint(1, 50))
                
                for c in candidates[:5]:
                    new_U = try_add(U_sorted, c)
                    if new_U is None:
                        continue
                    sc = _score_of_fast(new_U)
                    if sc > current_score + 1e-12:
                        U = new_U
                        current_score = sc
                        improved = True
                        break
            else:
                # Try removal
                if len(U_sorted) > 2:
                    idx = np.random.randint(1, len(U_sorted))
                    new_U = try_remove(U_sorted, idx)
                    if new_U is not None:
                        sc = _score_of_fast(new_U)
                        if sc > current_score + 1e-12:
                            U = new_U
                            current_score = sc
                            improved = True
                            break
        
        # Try shifts
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(U_sorted) > 2:
                idx = np.random.randint(1, len(U_sorted))
                delta = np.random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
                new_U = try_shift(U_sorted, idx, delta)
                if new_U is not None:
                    sc = _score_of_fast(new_U)
                    if sc > current_score + 1e-12:
                        U = new_U
                        current_score = sc
                        improved = True
                        break
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_U = U[:]
    
    return best_U
# EVOLVE-BLOCK-END

