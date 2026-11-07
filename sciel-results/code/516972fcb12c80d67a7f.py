import random
import itertools

def construct():
    def is_valid(S):
        n = len(S)
        sums = set()
        for r in range(1, n + 1):
            for combo in itertools.combinations(S, r):
                s = sum(combo)
                if s in sums:
                    return False
                sums.add(s)
        sum_list = list(sums)
        sum_list.sort()
        for i in range(len(sum_list)):
            a = sum_list[i]
            for j in range(i + 1, len(sum_list)):
                if sum_list[j] % a == 0:
                    return False
        return True

    # Try sizes from 8 down to 3
    for target_size in range(8, 2, -1):
        # Strategy 1: Narrow interval
        for M in range(30, 100, 5):
            for K in range(1, 10):
                if M + K > 256: continue
                for _ in range(20):
                    S = random.sample(range(M, M + K + 1), target_size)
                    if is_valid(S):
                        return S
        # Strategy 2: Random 1..256
        for _ in range(2000):
            S = random.sample(range(1, 257), target_size)
            if is_valid(S):
                return S

    return [31, 32, 34]
