"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _next_pow2(n):
    """Return the next power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def _score_set_fft(U):
    """
    Compute the score for a set U of nonnegative integers containing 0
    using FFT for efficiency on larger sets.
    """
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n <= 1:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0

    # For small sets, use direct computation
    if n <= 30:
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom

    # FFT-based computation for larger sets
    # Sumset: U+U has elements in [0, 2*max_u]
    size_sum = 2 * max_u + 1
    fft_size_sum = _next_pow2(2 * size_sum)

    # Create indicator vector for U
    vec_sum = np.zeros(fft_size_sum, dtype=np.float64)
    for x in U_set:
        vec_sum[x] = 1.0

    # FFT to compute sumset
    fft_sum = np.fft.rfft(vec_sum)
    conv_sum = np.fft.irfft(fft_sum * fft_sum, n=fft_size_sum)
    # Count how many positions have conv_sum > 0.5 (indicating at least one pair sums to that value)
    sumset_size = np.sum(conv_sum[:size_sum] > 0.5)

    # Difference set: U-U has elements in [-max_u, max_u]
    # Shift by max_u so we work in [0, 2*max_u]
    size_diff = 2 * max_u + 1
    fft_size_diff = _next_pow2(2 * size_diff)

    # Create indicator vector for U shifted
    vec_diff = np.zeros(fft_size_diff, dtype=np.float64)
    for x in U_set:
        vec_diff[x] = 1.0

    # For difference set, we need correlation: sum over a,b of f(a)*f(b) where a-b = d
    # This is equivalent to convolution of f with reversed f
    vec_rev = vec_diff.copy()
    vec_rev[max_u + 1 : 2 * max_u + 2] = vec_diff[0 : max_u + 1][::-1]
    # Actually, let's use a cleaner approach
    # Difference d = a - b. We want to count distinct d values.
    # Use the fact that the autocorrelation gives us the count.

    # Cleaner: create vector g where g[x] = 1 if x in U, else 0
    # Then the number of pairs (a,b) with a-b=d is sum_x g[x]*g[x-d]
    # This is cross-correlation, which can be done via FFT

    # For d in range [-max_u, max_u], count = sum_{x} g[x] * g[x-d]
    # Let's create an array h of size 2*max_u+1 where h[i] = g[i] for i in U
    # Then we want for each shift d, the inner product of g and shifted g

    # Use FFT: create vector of length >= 2*max_u+1
    g = np.zeros(2 * max_u + 1, dtype=np.float64)
    for x in U_set:
        g[x] = 1.0

    # Autocorrelation via FFT
    fft_g = np.fft.rfft(g, n=fft_size_diff)
    autocorr = np.fft.irfft(fft_g * np.conj(fft_g), n=fft_size_diff)
    # autocorr[d] = sum_x g[x] * g[x+d] for appropriate indexing
    # The autocorrelation at lag d gives sum_x g[x+d]*g[x]
    # We need to handle the circular wrap-around, but since we padded enough,
    # the linear autocorrelation is in the first 2*max_u+1 elements

    # For difference d (can be negative), the count is:
    # if d >= 0: autocorr[d] = sum_x g[x] * g[x+d]  -> pairs where a-b = -d? Let me think...
    # Actually autocorr[d] = sum_x g[x+d] * g[x]
    # If we let a = x+d, b = x, then a-b = d, so autocorr[d] counts pairs with difference d (for d >= 0)
    # For d < 0, autocorr[|d|] = sum_x g[x+|d|] * g[x], which counts pairs with difference |d|, i.e., a-b = |d|
    # But we also need pairs with a-b = -|d|, which is the same count by symmetry? No.
    # autocorr[d] for d >= 0 gives count of (a,b) with a-b = d (where a = x+d, b = x)
    # For negative differences, we need autocorr at negative lags, which equals autocorr at positive lags
    # due to the property autocorr[-d] = autocorr[d] for real signals? No, that's not right.

    # Let me reconsider. The autocorrelation R[d] = sum_x f[x+d] * f[x]
    # For d >= 0: R[d] = sum_{x} f[x+d]*f[x] = number of pairs (a,b) with a-b=d where a=x+d, b=x
    # For d < 0: R[d] = sum_x f[x+d]*f[x]. Let d' = -d > 0. R[-d'] = sum_x f[x-d']*f[x]
    #   = sum_y f[y]*f[y+d'] (substituting y = x-d') = R[d']
    # So R[-d] = R[d] for the autocorrelation of a real signal? That's not generally true.
    # Actually for autocorrelation, R[d] = R[-d] IS true because:
    # R[d] = sum_x f[x+d]*f[x], R[-d] = sum_x f[x-d]*f[x] = sum_y f[y]*f[y+d] (let y=x-d) = R[d]
    # So yes, autocorrelation is even! That means we can't distinguish positive from negative differences.

    # This approach doesn't directly give us the number of distinct differences.
    # We need a different approach for the difference set.

    # Alternative: just use the direct O(n^2) method for the difference set since it's bounded by n^2
    # and for the sumset we use FFT.

    # Actually, let's just use direct computation for both if n is moderate, and FFT for sumset when n is large
    # but difference set is still O(n^2) which is fine for n up to ~1000

    # Let's use direct for difference set
    diffset = set()
    U_list = list(U_set)
    for a in U_list:
        for b in U_list:
            diffset.add(a - b)
    diffset_size = len(diffset)

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    if n <= 1:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 0.0

    # For small sets, use direct computation
    if n <= 50:
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0.0
        return 1.0 + math.log(ratio) / denom
    else:
        return _score_set_fft(U_set)

def _golomb_ruler(k):
    """Generate a Golomb ruler with k marks using greedy construction."""
    # Simple greedy: start with 0, then add marks that maximize minimum distance
    marks = [0]
    for i in range(1, k):
        best_pos = None
        best_min_dist = -1
        # Try positions from current max + 1 onwards
        max_pos = marks[-1] + 1
        # Search for a good position
        for pos in range(max_pos, max_pos + 50):
            # Check if this creates a valid Golomb ruler (all pairwise distances distinct)
            diffs = set()
            valid = True
            for j in range(len(marks) + 1):
                if j < len(marks):
                    for m in marks:
                        d = pos - m
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                else:
                    for m in marks:
                        d = pos - m
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
            if valid:
                # Compute minimum distance
                all_marks = marks + [pos]
                min_d = min(all_marks[i+1] - all_marks[i] for i in range(len(all_marks)-1))
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = pos
                    if min_d >= 3:  # Good enough
                        break
        if best_pos is not None:
            marks.append(best_pos)
        else:
            # Fallback: just extend
            marks.append(marks[-1] + 1)
    return marks

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit

    best_set = {0, 1, 3}
    best_score = _score_set(best_set)

    # Try analytical constructions
    # Powers of 3
    for k in range(2, 20):
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
    for k in range(2, 30):
        U = {i * i for i in range(k)}
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)

    # Powers of 2
    for k in range(2, 15):
        U = {1 << i for i in range(k)}
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)

    # Golomb rulers
    for k in range(3, 15):
        marks = _golomb_ruler(k)
        U = set(marks)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = set(U)

    # Local search with FFT-enabled scoring
    improved = True
    while improved and time.time() < deadline:
        improved = False
        max_val = max(best_set) if best_set else 0
        search_range = max_val * 2 + 20

        # Try adding elements
        for new_elem in range(1, search_range + 1):
            if new_elem in best_set:
                continue
            candidate = best_set | {new_elem}
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
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
                improved = True
                if time.time() > deadline:
                    break

    return sorted(best_set)
# EVOLVE-BLOCK-END
