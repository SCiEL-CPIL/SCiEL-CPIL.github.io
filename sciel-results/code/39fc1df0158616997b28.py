import math
import time

def construct():
        start_time = time.time()
        deadline = start_time + 50  # 50s margin
        
        N = 12
        K = 27720 # lcm(1..12)
        
        # Precompute divisors or just use modulo
        # We need to find max L for each m in 0..K-1
        
        max_L_found = 0
        best_m = 0
        
        # Function to check if matching exists for given m and L
        # Returns True if there are distinct b_1..b_12 in (m, m+L] with i|b_i
        def has_matching(m, L):
            # available offsets for each i
            # offset j (1..L) is valid for i if (m+j) % i == 0
            # We need to assign distinct j's to 1..12
            
            # Precompute valid offsets for each i
            # valid[i] = list of j in 1..L such that (m+j)%i==0
            valid = [[] for _ in range(N+1)]
            for i in range(1, N+1):
                # First multiple of i > m
                # m < b <= m+L
                # b = k*i
                # k*i > m => k > m/i => k >= floor(m/i) + 1
                # b <= m+L => k*i <= m+L => k <= (m+L)/i
                # So k ranges from floor(m/i)+1 to floor((m+L)/i)
                # j = b - m = k*i - m
                k_start = m // i + 1
                k_end = (m + L) // i
                if k_end >= k_start:
                    # j values are k*i - m
                    # We can just list them
                    # But we need distinct j's.
                    # So for each i, we have a set of possible j's.
                    for k in range(k_start, k_end + 1):
                        valid[i].append(k * i - m)
            
            # Now check if we can pick distinct j for each i
            # Backtracking
            # Sort i by number of options to prune faster?
            # Or just 1..12
            used = [False] * (L + 1)
            
            def dfs(idx):
                if idx == N + 1:
                    return True
                for j in valid[idx]:
                    if not used[j]:
                        used[j] = True
                        if dfs(idx + 1):
                            return True
                        used[j] = False
                return False
            
            return dfs(1)

        # Iterate over residues
        for m in range(K):
            # Find minimal L
            # L is small, start from 1
            # Max L is likely < 200
            found = False
            for L in range(1, 200):
                if time.time() > deadline:
                    # Return best found so far
                    return int(best_m)
                if has_matching(m, L):
                    if L > max_L_found:
                        max_L_found = L
                        best_m = m
                    found = True
                    break
            
        return int(best_m)
