"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def _score_set_fft(U_set):
    """
    Compute the score using FFT-based convolution.
    
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    
    We use FFT to compute:
    - Sumset: |{a+b : a,b in U}| via autocorrelation of indicator
    - Diffset: |{a-b : a,b in U}| via cross-correlation
    """
    if not U_set or 0 not in U_set:
        return 0.0
    n = len(U_set)
    if n <= 1:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    # For small sets, use direct computation (faster overhead)
    if n <= 30:
        sumset = set()
        diffset = set()
        U_list = list(U_set)
        for a in U_list:
            for b in U_list:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom
    
    # FFT-based approach for larger sets
    # Sumset: use convolution of indicator with itself
    # If U has elements in [0, M], then sumset is in [0, 2M]
    # Create indicator array
    size = 2 * max_u + 1
    fft_size = _next_pow2(2 * size)
    
    # Indicator for U
    f = np.zeros(fft_size, dtype=np.float64)
    for u in U_set:
        f[u] = 1.0
    
    # Autocorrelation for sumset: (f * f) gives counts of pairs summing to each value
    F = np.fft.rfft(f)
    autocorr = np.fft.irfft(F * np.conj(F), n=fft_size)
    
    # Sumset size: number of positions where autocorr > 0.5
    # (autocorr[k] counts pairs (a,b) with a+b=k)
    sumset_size = np.sum(autocorr[:size] > 0.5)
    
    # Diffset: use cross-correlation
    # We need |{a-b : a,b in U}|
    # Shift: let g[x] = 1 if x in U, else 0
    # Cross-correlation h[d] = sum_x g[x] * g[x+d]
    # This counts pairs (x, x+d) both in U, i.e., differences of d
    # For negative differences, we need to handle the full range
    
    # Alternative: embed U in a larger array and compute full correlation
    # Let's use the fact that diffset = {a-b : a,b in U}
    # Create an array where we place U at positions [offset, offset+max_u]
    # and compute correlation with itself
    
    # Simpler: use the relationship that |U-U| can be computed via
    # the fact that U-U contains all differences
    # Let's compute it differently: for each possible difference d in [-max_u, max_u],
    # check if there exist a,b in U with a-b=d
    
    # Actually, let's use FFT properly for the difference set
    # Place U in array of length L >= 2*max_u + 1
    # The difference set can be found via:
    # If we define h[d] = 1 if d in U-U, then
    # h = cross_correlation(f, f) where f is the indicator
    
    # Let me use a different approach: 
    # For the difference set, we can compute:
    # diff_count[d] = |{a in U : a + d in U}| for d >= 0
    # and diff_count[-d] = diff_count[d] by symmetry (if we're careful)
    
    # Actually, U-U is not symmetric unless U is symmetric around 0
    # Let me just compute it directly for the range [-max_u, max_u]
    
    # For efficiency with FFT, let's compute the autocorrelation which gives us
    # the count of pairs with each difference
    # autocorr[k] = sum_i f[i] * f[i+k] for k >= 0 (with proper indexing)
    
    # Let me redefine: use the full FFT correlation
    # The linear cross-correlation of f with itself gives us the difference counts
    
    # For now, let's use a hybrid: FFT for sumset, direct for diffset if manageable
    # Or better: compute both via FFT
    
    # Let me compute diffset via:
    # Create array g of length 2*max_u+1 where g[max_u + x] = 1 if x in U
    # Then autocorrelation of g gives us the difference structure
    
    g = np.zeros(fft_size, dtype=np.float64)
    offset = max_u
    for u in U_set:
        g[offset + u] = 1.0
    
    G = np.fft.rfft(g)
    auto_g = np.fft.irfft(G * np.conj(G), n=fft_size)
    
    # auto_g[k] counts pairs (i,j) with i-j = k-offset... 
    # Actually, let me think more carefully.
    # If g[x] = 1 for x in {offset+u : u in U}, then
    # auto_g[d] = sum_x g[x] * g[x+d]
    # This counts pairs (x, x+d) both in support of g
    # So for d >= 0, auto_g[d] = |{u in U : u+d in U}| (shifted by offset)
    # For the difference set, we need all d such that auto_g[d+offset] > 0 or auto_g[d-offset] > 0
    
    # This is getting complex. Let me just use direct computation for diffset
    # since the sumset FFT already saves significant time
    
    # Direct diffset computation
    U_list = list(U_set)
    diffset = set()
    for a in U_list:
        for b in U_list:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size <= 0 or diffset_size <= 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set(U_set):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U_set or 0 not in U_set:
        return 0.0
    return _score_set_fft(U_set)

def _golomb_ruler(n):
    """Generate a Golomb ruler with n marks (heuristic)."""
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    # Use a greedy approach
    marks = [0]
    for i in range(1, n):
        # Try to place next mark
        placed = False
        for pos in range(marks[-1] + 1, marks[-1] * 3 + 10):
            # Check if all differences are distinct
            diffs = set()
            ok = True
            for j in range(len(marks)):
                d = pos - marks[j]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if ok:
                marks.append(pos)
                placed = True
                break
        if not placed:
            # Fallback: just add a large gap
            marks.append(marks[-1] + max(marks[-1] + 1, 10))
    return marks

def _local_search(U_set, deadline):
    """Perform local search to improve the set."""
    best_set = set(U_set)
    best_score = _score_set(best_set)
    
    max_val = max(best_set) if best_set else 0
    search_range = max_val * 2 + 20
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding new elements
        for new_elem in range(1, search_range + 1):
            if new_elem in best_set:
                continue
            candidate = best_set | {new_elem}
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                max_val = max(best_set)
                search_range = max_val * 2 + 20
                improved = True
                if time.time() > deadline:
                    break
        
        # Try removing elements (keep 0)
        if time.time() > deadline:
            break
        for elem in list(best_set):
            if elem == 0:
                continue
            candidate = best_set - {elem}
            if len(candidate) < 2:
                continue
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                max_val = max(best_set) if best_set else 0
                search_range = max_val * 2 + 20
                improved = True
                if time.time() > deadline:
                    break
        
        # Try swapping: remove one, add another
        if time.time() > deadline:
            break
        for elem in list(best_set):
            if elem == 0:
                continue
            for new_elem in range(1, search_range + 1):
                if new_elem in best_set or new_elem == elem:
                    continue
                candidate = (best_set - {elem}) | {new_elem}
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    max_val = max(best_set)
                    search_range = max_val * 2 + 20
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    return best_set, best_score

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    best_set = {0, 1, 3}
    best_score = _score_set(best_set)
    
    # Try analytical constructions
    # Powers of 3
    for k in range(2, 15):
        U = {0}
        val = 1
        for i in range(k):
            U.add(val)
            val *= 3
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Squares
    for k in range(2, 20):
        U = {i * i for i in range(k)}
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Binary
    for k in range(2, 12):
        U = {1 << i for i in range(k)}
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Golomb rulers
    for n in range(3, 15):
        marks = _golomb_ruler(n)
        U = set(marks)
        if 0 not in U:
            U.add(0)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)
    
    # Local search
    best_set, best_score = _local_search(best_set, deadline)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
