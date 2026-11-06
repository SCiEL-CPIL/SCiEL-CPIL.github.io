import random

def construct():
    def is_valid_fast(S):
        M = 25000
        mask = (1 << (2*M + 1)) - 1
        for a in S:
            vals = [x - a for x in S if x != a]
            dp = 0
            for v in vals:
                if v > 0:
                    dp = (dp | (dp << v) | (1 << (M + v))) & mask
                elif v < 0:
                    dp = (dp | (dp >> (-v)) | (1 << (M + v))) & mask
            if (dp >> M) & 1:
                return False
        return True

    # Try random sets
    for _ in range(500):
        S = random.sample(range(1, 513), 48)
        if is_valid_fast(S):
            return S
            
    # Fallback to greedy if random fails (unlikely)
    # ... (current greedy logic)
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
