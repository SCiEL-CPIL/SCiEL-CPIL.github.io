import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def check_validity(S):
        if len(S) <= 1:
            return True
        S_list = list(S)
        for i, x in enumerate(S_list):
            others = [y for y in S_list if y != x]
            if not others:
                continue
            L = reduce(_lcm, others, 1)
            target = L // x
            if target == 0:
                continue
            vals = [L // y for y in others]
            n = len(vals)
            half = n // 2
            left = vals[:half]
            right = vals[half:]
            left_sums = set()
            for mask in range(1 << len(left)):
                s = 0
                for j in range(len(left)):
                    if mask & (1 << j):
                        s += left[j]
                left_sums.add(s)
            right_sums = set()
            for mask in range(1 << len(right)):
                s = 0
                for j in range(len(right)):
                    if mask & (1 << j):
                        s += right[j]
                right_sums.add(s)
            for ls in left_sums:
                if (target - ls) in right_sums:
                    return False
        return True

    best = list(range(13, 25))
    best_size = len(best)

    for restart in range(30):
        if time.time() > deadline:
            break
        if restart == 0:
            order = list(range(24, 0, -1))
        elif restart == 1:
            order = list(range(1, 25))
        else:
            order = list(range(1, 25))
            random.seed(restart)
            random.shuffle(order)
        S = []
        for x in order:
            if time.time() > deadline:
                break
            new_S = S + [x]
            if check_validity(new_S):
                S = new_S
        if len(S) > best_size:
            best = S
            best_size = len(S)

    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        candidates = [x for x in range(1, 25) if x not in best]
        random.shuffle(candidates)
        for x in candidates:
            new_S = best + [x]
            if check_validity(new_S):
                best = new_S
                improved = True
                break
        if not improved:
            break

    return best
# EVOLVE-BLOCK-END
