import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    for limit in [100000000, 200000000, 400000000]:
        if time.time() > deadline - 35:
            break

        # Generate powerful numbers up to limit
        pows = set([1])
        max_i = math.isqrt(limit)
        max_j_global = int(limit ** (1.0 / 3.0)) + 2
        cubes = [j ** 3 for j in range(1, max_j_global + 1)]

        for i in range(1, max_i + 1):
            sq = i * i
            if sq > limit:
                break
            max_j = int((limit // sq) ** (1.0 / 3.0))
            if max_j > max_j_global:
                max_j = max_j_global
            for j in range(1, max_j + 1):
                val = sq * cubes[j - 1]
                if val <= limit:
                    pows.add(val)
                else:
                    break
            if time.time() > deadline - 25:
                break

        P = np.array(sorted(pows), dtype=np.int64)
        K = len(P)
        if K < 2:
            continue

        # Count representations
        if K * K <= 100000000:
            sums = P[:, None] + P[None, :]
            unique_sums, counts = np.unique(sums, return_counts=True)
            max_c = int(np.max(counts))
            idx = int(np.argmax(counts))
            best_sum = int(unique_sums[idx])
            D = 0
            if best_sum % 2 == 0:
                half = best_sum // 2
                pos = np.searchsorted(P, half)
                if pos < K and P[pos] == half:
                    D = 1
            actual_count = (max_c + D) // 2
        else:
            sum_counts = {}
            chunk_size = 50
            for i_start in range(0, K, chunk_size):
                i_end = min(i_start + chunk_size, K)
                chunk = P[i_start:i_end]
                for idx in range(len(chunk)):
                    a = int(chunk[idx])
                    j_min = np.searchsorted(P, a)
                    if j_min >= K:
                        continue
                    sums = a + P[j_min:]
                    vals, cnts = np.unique(sums, return_counts=True)
                    for v, c in zip(vals, cnts):
                        sum_counts[int(v)] = sum_counts.get(int(v), 0) + int(c)
                if time.time() > deadline - 15:
                    break
            if not sum_counts:
                continue
            best_sum = max(sum_counts, key=sum_counts.get)
            actual_count = sum_counts[best_sum]

        # Retrieve pairs
        pows_set = set(P.tolist())
        pairs = []
        for a in P:
            if a > best_sum:
                break
            b = best_sum - a
            if b < a:
                continue
            if b in pows_set:
                pairs.append([int(a), int(b)])

        if pairs and len(pairs) > best_score:
            best_score = len(pairs)
            best_result = {'N': int(best_sum), 'pairs': pairs}

        if time.time() > deadline - 5:
            break

    return best_result
