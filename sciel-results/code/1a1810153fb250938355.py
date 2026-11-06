import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    def try_add(S, sums, x):
        """
        Check if x can be added to S.
        Returns (new_sums_set, success_bool)
        """
        new_sums = set()
        
        # (x, x, x)
        s = 3 * x
        if s in sums:
            return None, False
        new_sums.add(s)
        
        # (x, x, s_i) for s_i in S
        for s_i in S:
            s = 2 * x + s_i
            if s in sums or s in new_sums:
                return None, False
            new_sums.add(s)
            
        # (x, s_i, s_j) for s_i, s_j in S, i <= j
        n = len(S)
        for i in range(n):
            s_i = S[i]
            for j in range(i, n):
                s_j = S[j]
                s = x + s_i + s_j
                if s in sums or s in new_sums:
                    return None, False
                new_sums.add(s)
                
        return new_sums, True

    def run_greedy(candidates):
        S = []
        sums = set()
        for x in candidates:
            new_sums, valid = try_add(S, sums, x)
            if valid:
                S.append(x)
                sums.update(new_sums)
        return S

    # 1. Deterministic Greedy: Try 1 to 4096
    # This is fast and gives a baseline.
    candidates = list(range(1, 4097))
    S = run_greedy(candidates)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
        
    # 2. Randomized Greedy: Shuffle and try
    # We can also try biased shuffles or just uniform.
    # To make it faster, we might not need to try ALL 4096 in every restart if we are confident,
    # but greedy usually benefits from seeing all options.
    # However, 4096 is small enough.
    
    # Optimization: If we find a set of size 80 (the max), we can stop.
    # But 80 is likely impossible for n=4096.
    # Max size is approx n^(1/3) * constant.
    # 4096^(1/3) = 16.
    # So we are looking for improvements over the seed.
    
    while time.time() < deadline:
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        S = run_greedy(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Optional: Local search / Swap improvement on best_set
        # Given time, simple restarts are robust.
        
    return best_set
