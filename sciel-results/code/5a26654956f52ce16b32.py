"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    def check_mean(S, a):
        """Return True if a is the arithmetic mean of some subset of S of size >=2."""
        others = [x for x in S if x != a]
        m = len(others)
        if m < 2:
            return False
        # dp[k] = bitset (Python int) where bit s is 1 if sum s achievable with k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= dp[k-1] << x
        max_sum = sum(others)
        for k in range(2, m + 1):
            t = k * a
            if t > max_sum:
                break
            if (dp[k] >> t) & 1:
                return True
        return False
    
    def valid(S):
        for a in S:
            if check_mean(S, a):
                return False
        return True
    
    best = []
    rng = random.Random(42)
    
    for _ in range(40):
        if time.time() > deadline:
            break
        cand = list(range(1, 513))
        rng.shuffle(cand)
        sel = []
        for x in cand:
            if len(sel) >= 48:
                break
            if x in sel:
                continue
            ns = sel + [x]
            if valid(ns):
                sel.append(x)
        if len(sel) > len(best):
            best = sel
    
    return sorted(best)

# EVOLVE-BLOCK-END
