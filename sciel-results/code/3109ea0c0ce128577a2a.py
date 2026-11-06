import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1

    # Generate powerful numbers up to limit using the parametric form:
    # n = a^2 * b^3 where b is square-free
    # This ensures we generate each powerful number exactly once (up to the limit)
    limit = 60_000_000

    # Generate all powerful numbers <= limit
    pows_set = set()
    pows_set.add(1)

    max_a = math.isqrt(limit)
    max_b_cbrt = int(limit ** (1/3)) + 2

    # Precompute cubes
    cubes = [b ** 3 for b in range(1, max_b_cbrt + 1)]

    for a in range(1, max_a + 1):
        sq = a * a
        if sq > limit:
            break
        max_b = min(max_b_cbrt, int((limit // sq) ** (1/3)) + 1)
        for b in range(1, max_b + 1):
            val = sq * cubes[b - 1]
            if val <= limit:
                pows_set.add(val)
            else:
                break
        if time.time() > deadline - 30.0:
            break

    P_list = sorted(pows_set)
    K = len(P_list)

    if K == 0:
        return {'N': best_N, 'pairs': best_pairs}

    P = np.array(P_list, dtype=np.int64)
    pows_set_np = np.array(P_list, dtype=np.int64)

    # Use FFT to count sums
    try:
        size = 2 * limit + 1
        vec = np.zeros(size, dtype=np.float64)
        vec[P] = 1.0

        fft_vec = np.fft.rfft(vec)
        conv = np.fft.irfft(fft_vec * fft_vec, n=size)
        counts = np.rint(conv).astype(np.int64)

        # Find top 50 candidates
        c_slice = counts[2:]
        if len(c_slice) < 50:
            top_indices_local = np.argsort(c_slice)[-len(c_slice):]
        else:
            top_indices_local = np.argpartition(c_slice, -50)[-50:]
        top_indices = top_indices_local + 2

    except MemoryError:
        # Fallback: direct counting for smaller K
        if K < 8000:
            sums = P[:, None] + P[None, :]
            flat_sums = sums.ravel()
            counts = np.bincount(flat_sums, minlength=2 * limit + 1)
            c_slice = counts[2:]
            if len(c_slice) < 50:
                top_indices_local = np.argsort(c_slice)[-len(c_slice):]
            else:
                top_indices_local = np.argpartition(c_slice, -50)[-50:]
            top_indices = top_indices_local + 2
        else:
            return {'N': best_N, 'pairs': best_pairs}
    except Exception:
        return {'N': best_N, 'pairs': best_pairs}

    # Verify top candidates
    for s in top_indices:
        s_val = int(s)
        if s_val > 2 * limit or s_val < 2:
            continue

        pairs = []
        half = s_val // 2

        for a in P_list:
            if a > half:
                break
            b = s_val - a
            if b in pows_set:
                pairs.append([a, b])

        score = len(pairs)
        if score > best_score:
            best_score = score
            best_N = s_val
            best_pairs = pairs

        if time.time() > deadline - 5.0:
            break

    return {'N': best_N, 'pairs': best_pairs}
