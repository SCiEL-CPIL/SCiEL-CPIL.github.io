import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)

    A = [L // (i + 1) for i in range(22)]

    def check_validity(indices):
        k = len(indices)
        if k == 0:
            return True, None

        vals = [A[i] for i in indices]
        size = 1 << k
        sums = [0] * size
        seen = {0: 0}

        for m in range(1, size):
            low_bit = m & -m
            idx = low_bit.bit_length() - 1
            prev = m ^ low_bit
            s = sums[prev] + vals[idx]
            sums[m] = s

            if s in seen:
                mask1 = m
                mask2 = seen[s]
                coll_idx = []
                for j in range(k):
                    if (mask1 >> j) & 1 or (mask2 >> j) & 1:
                        coll_idx.append(indices[j])
                return False, coll_idx
            else:
                seen[s] = m

        return True, None

    best_set = []

    def try_greedy(heuristic):
        S = list(range(22))
        while True:
            if time.time() > deadline:
                return S if len(S) > len(best_set) else None

            is_valid, coll = check_validity(S)
            if is_valid:
                return S
            if not coll:
                break

            if heuristic == 'small_idx':
                x = min(coll)
            elif heuristic == 'large_idx':
                x = max(coll)
            else:
                x = random.choice(coll)

            S.remove(x)
            if not S:
                return None

    for _ in range(20):
        if time.time() > deadline:
            break
        h = random.choice(['small_idx', 'large_idx', 'random'])
        res = try_greedy(h)
        if res and len(res) > len(best_set):
            best_set = res

    if not best_set:
        best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18] # Fallback

    return [i + 1 for i in best_set]
