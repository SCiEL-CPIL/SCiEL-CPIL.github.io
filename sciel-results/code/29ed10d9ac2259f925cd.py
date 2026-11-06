import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    N = 512
    TIME_LIMIT = 85  # seconds, leave 5s margin

    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT

    # Precompute LCM compatibility matrix
    # compat[i][j] = True if lcm(i,j) <= N
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            l = i // g * j
            if l <= N:
                compat[i][j] = True
                compat[j][i] = True
            else:
                compat[i][j] = False
                compat[j][i] = False

    # Precompute divisors for each number
    divisors = [set() for _ in range(N + 1)]
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            divisors[m].add(d)

    # Sort numbers by number of divisors (descending)
    nums_by_divs = sorted(range(1, N + 1), key=lambda x: -len(divisors[x]))

    def is_compatible(x, s):
        """Check if x is compatible with all elements in set s"""
        for y in s:
            if not compat[x][y]:
                return False
        return True

    def greedy_build(seed_set):
        """Greedy addition of elements"""
        current = set(seed_set)
        current.add(1)
        # Sort candidates by heuristic: prefer those with more divisors
        candidates = sorted(range(2, N + 1), key=lambda x: -len(divisors[x]))
        for x in candidates:
            if x not in current:
                if is_compatible(x, current):
                    current.add(x)
        return current

    def local_search_1for1(current):
        """Try to add elements, or swap 1 for 1+ to increase size"""
        current = set(current)
        improved = True
        while improved and not check_deadline():
            improved = False
            # First, try to add any element
            added = False
            for x in range(2, N + 1):
                if x not in current:
                    if is_compatible(x, current):
                        current.add(x)
                        improved = True
                        added = True
                        break
            if added:
                continue

            # If no direct addition, try 1-for-1+ swaps
            elems = sorted(current)
            for u in elems:
                if u == 1:
                    continue
                if check_deadline():
                    break
                trial = set(current)
                trial.discard(u)
                # Greedily add to trial
                for x in range(2, N + 1):
                    if x not in trial:
                        if is_compatible(x, trial):
                            trial.add(x)

                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break

        return current

    def local_search_2for3(current):
        """Try removing 2 elements and adding up to 3"""
        current = set(current)
        improved = True
        while improved and not check_deadline():
            improved = False
            elems = sorted(current)
            for i in range(len(elems)):
                if check_deadline():
                    break
                for j in range(i + 1, len(elems)):
                    if check_deadline():
                        break
                    u, v = elems[i], elems[j]
                    if u == 1 or v == 1:
                        continue
                    trial = set(current)
                    trial.discard(u)
                    trial.discard(v)
                    # Greedily add
                    for x in range(2, N + 1):
                        if x not in trial:
                            if is_compatible(x, trial):
                                trial.add(x)

                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        break
                if improved:
                    break
        return current

    best_set = set()
    best_size = 0

    # Strategy 1: Seeds from top divisor counts
    for m in nums_by_divs[:20]:
        if check_deadline():
            break
        seed = set(divisors[m])
        s = greedy_build(seed)
        s = local_search_1for1(s)
        s = local_search_2for3(s)

        if len(s) > best_size:
            best_set = set(s)
            best_size = len(s)

    # Strategy 2: Random restarts
    if not check_deadline():
        random.seed(42)
        for _ in range(10):
            if check_deadline():
                break
            m = random.randint(1, N)
            seed = set(divisors[m])
            # Add some random compatible numbers to diversify seed
            for _ in range(5):
                x = random.randint(1, N)
                if is_compatible(x, seed):
                    seed.add(x)

            s = greedy_build(seed)
            s = local_search_1for1(s)
            s = local_search_2for3(s)

            if len(s) > best_size:
                best_set = set(s)
                best_size = len(s)

    # Final aggressive local search on the best set found
    if not check_deadline():
        best_set = local_search_1for1(best_set)
        best_set = local_search_2for3(best_set)

    # Verification
    final_list = sorted(best_set)
    valid = len(final_list) > 0
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if not compat[final_list[i]][final_list[j]]:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback
        return list(range(1, int(math.isqrt(N)) + 1))

    return final_list
