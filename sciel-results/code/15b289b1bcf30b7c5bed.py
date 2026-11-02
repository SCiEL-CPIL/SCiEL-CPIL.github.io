"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft


def _score_set(U):
    """Compute the score for a finite set U containing 0, using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # For very small sets, direct computation is faster
    if n <= 8:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom

    # FFT-based computation for larger sets
    # Sumset: U + U, values in [0, 2*max_u]
    sum_range = 2 * max_u + 1
    fft_len = 1
    while fft_len < 2 * sum_range:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_range] > 1e-9))

    # Difference set: U - U, values in [-max_u, max_u]
    # Use autocorrelation: place U at offset max_u, convolve with reversed
    diff_range = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_range:
        fft_len2 *= 2

    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0

    # Autocorrelation via FFT
    freq_d = fft(diff_vec)
    autocorr = ifft(freq_d * np.conj(freq_d)).real

    # The autocorrelation at lag k gives count of pairs with difference k
    # diff_vec has U at positions [max_u, max_u+u] for u in U
    # autocorr[k] = sum_i diff_vec[i] * diff_vec[i-k] (with wraparound handled by padding)
    # We need to count positions where autocorr > 0 in the valid range
    # Valid lags: -max_u to max_u, which in the autocorrelation array are at positions
    # fft_len2 - max_u to fft_len2 + max_u (wrapping)
    # Actually, for a zero-padded signal, autocorr[0] is at index 0,
    # positive lags at 1..max_u, negative lags at fft_len2-max_u..fft_len2-1

    # Count nonzero autocorrelation entries in valid range
    valid_mask = np.zeros(fft_len2, dtype=bool)
    valid_mask[0] = True
    valid_mask[1:max_u+1] = True
    valid_mask[fft_len2-max_u:fft_len2] = True
    diffset_size = int(np.sum(autocorr[valid_mask] > 1e-9))

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom


def _golomb_ruler(n):
    """Generate a Golomb ruler of order n."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }
    if n in known:
        return known[n]
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _mian_chowla(n):
    """Generate Mian-Chowla sequence (Sidon sequence) of length n."""
    if n <= 1:
        return [0]
    seq = [0, 1]
    while len(seq) < n:
        # Find smallest candidate that doesn't create duplicate sums
        cand = seq[-1] + 1
        while True:
            valid = True
            sums = set()
            for i in range(len(seq) + 1):
                for j in range(i, len(seq) + 1):
                    if i == len(seq):
                        a = cand
                    else:
                        a = seq[i]
                    if j == len(seq):
                        b = cand
                    else:
                        b = seq[j]
                    s = a + b
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                seq.append(cand)
                break
            cand += 1
    return seq


def _local_search(U, max_iters=3000, deadline=None):
    """Improve set U via local search with FFT scoring."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        # Try moving one element (not 0)
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try nearby values and some random ones
            candidates = set()
            for offset in range(-30, 31, 1):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(15):
                candidates.add(random.randint(1, upper_bound))

            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 200
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # Try adding an element
            if len(U) < 100:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            # Try removing an element (if size > 3)
            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

        if not improved:
            break

    return sorted(set(best_U)), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # Golomb rulers of various sizes
    for n in range(3, 15):
        if time.time() > deadline:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)

    # Mian-Chowla sequences
    for n in range(3, 15):
        if time.time() > deadline:
            break
        mc = _mian_chowla(n)
        candidates.append(mc)

    # Structured sparse sets: geometric-like spacing
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        n = random.randint(4, 20)
        # Start with 0, then add elements with increasing gaps
        U = [0]
        current = 1
        for i in range(n - 1):
            gap = random.randint(1, 5) * (i + 1)
            current += gap
            U.append(current)
        candidates.append(sorted(U))

    # Random sparse sets
    for trial in range(30):
        if time.time() > deadline:
            break
        n = random.randint(4, 25)
        max_val = random.randint(20, 300)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)

    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Local search on best candidate
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=deadline)

    # Try local search on a few other good candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
    scored_candidates.sort(key=lambda x: -x[0])

    for s, U in scored_candidates[:5]:
        if time.time() > deadline:
            break
        if sorted(set(U)) == best_U:
            continue
        if s > best_score - 0.05:
            improved_U, improved_score = _local_search(U, max_iters=1000, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
