"""Rewritten program: maximize 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
import time
import random
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - time.time()

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    # Direct computation (works well for n up to ~200)
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

def _score_set_fast(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    if n <= 150:
        return _score_set(U)

    # Use FFT for sumset
    arr = np.array(U, dtype=np.float64)
    size = 2 * max_u + 1
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2

    indicator = np.zeros(fft_size, dtype=np.float64)
    indicator[arr] = 1.0
    conv = np.fft.ifft(np.fft.fft(indicator) ** 2).real
    sumset_size = np.sum(conv[:size] > 0.5)

    # Difference set: |U-U| = |{a-b : a,b in U}|
    # Use the fact that U-U = U + (-U), and -U = {max_u - u for u in U} shifted
    # Actually simpler: compute directly for the difference
    # For large n this is still O(n^2), but we can try
    # Alternative: use the relation that for the difference set,
    # we can compute via autocorrelation
    rev = np.zeros(fft_size, dtype=np.float64)
    rev[max_u - arr + max_u] = 1.0  # place -U shifted by max_u
    # Actually: -u ranges from -max_u to 0. Shift by max_u: max_u - u ranges from 0 to max_u
    # U-U = {a-b} = {a + (-b)}. Let V = {max_u - b : b in U} (shifted -U)
    # Then U-U shifted by max_u = U + V
    V = np.zeros(fft_size, dtype=np.float64)
    V[max_u - arr] = 1.0  # positions max_u - u for u in U, range [0, max_u]
    conv_diff = np.fft.ifft(np.fft.fft(indicator) * np.fft.fft(V)).real
    # U-U shifted by max_u: values from 0 to 2*max_u
    diffset_size = np.sum(conv_diff[:2 * max_u + 1] > 0.5)

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
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 82, 94],
    }
    if n in known:
        return known[n][:n]
    if n <= 1:
        return [0]
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        pos = ruler[-1] + 1
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
            if pos > 2000:
                break
    return ruler

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 200

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if _time_left() < 5:
            break
        improved = False

        # Try moving one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            # Try nearby values and some random
            candidates = set()
            for offset in range(-30, 31):
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
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-10:
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
            if len(U) < 300:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            if not improved:
                # Try removing an element
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            max_u = max(U)
                            upper_bound = max_u + 200
                            improved = True
                            break

        if not improved:
            # Random perturbation
            if random.random() < 0.15:
                new_U = U[:]
                for _ in range(min(5, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                if len(set(new_U)) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True

            if not improved:
                # Try a bigger jump: replace several elements
                if random.random() < 0.05 and len(U) > 5:
                    new_U = U[:]
                    for _ in range(min(10, len(new_U))):
                        idx = random.randint(1, len(new_U) - 1)
                        new_U[idx] = random.randint(1, upper_bound * 2)
                    if len(set(new_U)) == len(new_U):
                        s = _score_set(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            max_u = max(U)
                            upper_bound = max_u + 200
                            improved = True

            if not improved:
                break

    return sorted(set(best_U)), best_score

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 140

    random.seed(12345)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # Generate diverse initial candidates
    candidates = []

    # Golomb rulers of various sizes
    for n in range(3, 25):
        if _time_left() < 30:
            break
        gr = _golomb_ruler(n)
        if len(gr) == n:
            candidates.append(gr)

    # Random sets with various sizes and ranges
    for trial in range(50):
        if _time_left() < 30:
            break
        n = random.randint(3, 80)
        max_val = random.randint(10, 1000)
        U = [0] + random.sample(range(1, max_val + 1), min(n - 1, max_val))
        U = sorted(U)
        candidates.append(U)

    # Structured sets: sparse arithmetic progressions
    for base in [2, 3, 4, 5, 7, 11, 13]:
        if _time_left() < 30:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U)

    # Power sets (exponential growth)
    for p in [2, 3]:
        if _time_left() < 30:
            break
        U = [0]
        val = 1
        while val < 5000:
            U.append(val)
            val *= p
        candidates.append(U)

    # Mixed: combine Golomb ruler elements with random additions
    if _time_left() > 30:
        gr = _golomb_ruler(15)
        for _ in range(5):
            U = gr[:]
            for _ in range(10):
                U.append(random.randint(1, 200))
            U = sorted(set(U))
            candidates.append(U)

    # Evaluate all candidates
    for U in candidates:
        if _time_left() < 20:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Local search on the best candidate
    if _time_left() > 10:
        best_U, best_score = _local_search(best_U, max_iters=8000, deadline=_DEADLINE - 5)

    # Local search on a few other good candidates
    scored = []
    for U in candidates:
        if _time_left() < 10:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        scored.append((s, U))

    scored.sort(key=lambda x: -x[0])
    for s, U in scored[:5]:
        if _time_left() < 8:
            break
        if s < best_score - 0.01:
            break
        improved_U, improved_score = _local_search(U, max_iters=3000, deadline=_DEADLINE - 5)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U

    # Final polish: try a few more local search iterations on the best
    if _time_left() > 5:
        best_U, best_score = _local_search(best_U, max_iters=2000, deadline=_DEADLINE - 2)

    return sorted(set(best_U))
