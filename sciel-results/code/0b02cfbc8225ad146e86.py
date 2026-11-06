import time
from math import isqrt
import numpy as np

def _is_squarefree(n):
    # quick check
    if n < 2:
        return True
    # factor
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            x //= p
            if x % p == 0:
                return False
        p += 1 if p == 2 else 2
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s (evaluator kills at 150s)

    # Generate powerful numbers up to L, cap count
    L = 10**10
    MAX_COUNT = 60000

    powerfuls = [1]
    # squares: a^2
    max_a = isqrt(L)
    # We'll collect all a^2 * b^3
    seen = {1}
    pows = [1]
    max_b = int(L ** (1/3)) + 1
    for b in range(2, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L and n not in seen:
                seen.add(n)
                pows.append(n)
                if len(pows) >= MAX_COUNT:
                    break
        if len(pows) >= MAX_COUNT:
            break
    # also add pure squares not covered? a^2*b^3 with b=1 gives a^2, covered by b loop? No, b starts at 2.
    # Add squares separately
    for a in range(1, isqrt(L) + 1):
        n = a * a
        if n not in seen:
            seen.add(n)
            pows.append(n)
            if len(pows) >= MAX_COUNT:
                break

    pows.sort()
    arr = np.array(pows, dtype=np.int64)
    n = len(arr)
    print("Generated", n, "powerful numbers up to", pows[-1] if n else 0)

    # Count sums using vectorized bincount per row
    # To save memory/time, do in chunks
    best_N = 2
    best_count = 1
    # We'll accumulate counts in a dict for only the sums we see? No, use bincount per i is O(n) each -> O(n^2)
    # Instead, do full pairwise but vectorized in blocks
    # Build count array sized 2*L
    # 2*L = 2*10^10 -> too big for int array. Use dict.
    from collections import defaultdict
    counts = defaultdict(int)

    # Process i in blocks, vectorize over j
    block = 2000
    for i in range(0, n, block):
        i_end = min(i + block, n)
        ai = arr[i:i_end]  # shape (block,)
        # for each i, j from i to n-1
        # We need pairs (ai[k], arr[j]) with j >= i+k roughly; to keep a<=b, j>=i+k
        # Simpler: allow all j, then we'd double count; but we want distinct pairs a<=b.
        # Let's just compute all sums ai[k] + arr[j] for j in range(i, n) and count, but that overcounts when i<j both processed.
        # To get a<=b exactly once: for each i, j from i to n-1.
        # Vectorize: for each k in block, sums = ai[k] + arr[i+k:]
        for k in range(i_end - i):
            j_start = i + k
            if j_start >= n:
                break
            sums = ai[k] + arr[j_start:]
            # count via bincount
            bc = np.bincount(sums)
            # accumulate into counts for nonzero
            # bincount returns array indexed by sum value; but sums up to 2*L huge -> bincount allocates huge array!
            # Problem: bincount size = max(sums)+1 ~ 2*10^10 -> impossible.
            break
        if time.time() > deadline - 20:
            break

    # bincount won't work due to huge range. Use dict accumulation with vectorized unique.
    # Redo: accumulate counts using np.unique per row.
    counts = defaultdict(int)
    for i in range(0, n):
        if time.time() > deadline - 20:
            break
        j_start = i
        sums = arr[i] + arr[j_start:]
        # unique with counts
        u, c = np.unique(sums, return_counts=True)
        for idx in range(len(u)):
            counts[int(u[idx])] += int(c[idx])
        if i % 5000 == 0:
            print("i", i, "elapsed", time.time() - start_time)

    if counts:
        best_N = max(counts, key=counts.get)
        best_count = counts[best_N]
    else:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Collect pairs for best_N
    pows_set = set(pows)
    pairs = []
    for a in pows:
        if 2 * a > best_N:
            break
        b_val = best_N - a
        if b_val in pows_set and b_val >= a:
            pairs.append([a, b_val])

    # verify
    assert len(pairs) == best_count or best_count <= len(pairs)
    return {'N': int(best_N), 'pairs': pairs}
