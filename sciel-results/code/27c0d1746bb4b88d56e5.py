    def find_matching(L, W):
        available = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            start_m = L
            rem = start_m % d
            if rem != 0:
                start_m += d - rem
            
            m = start_m
            while m <= L + W:
                available[i].append(m)
                m += d
        
        order = sorted(range(n), key=lambda i: len(available[i]))
        
        used = [False] * (W + 1)
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in available[i]:
                idx_val = val - L
                if not used[idx_val]:
                    used[idx_val] = True
                    assignment[i] = val
                    if backtrack(idx + 1):
                        return True
                    used[idx_val] = False
            return False
        
        if backtrack(0):
            return assignment
        return None
