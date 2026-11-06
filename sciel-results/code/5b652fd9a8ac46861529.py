import time
import sys

def construct():
        start_time = time.time()
        deadline = start_time + 110  # 10s margin
        
        required = set(range(1, 31))
        
        # We know k must be at least 9.
        # Let's try to find a solution for increasing M.
        # Lower bound for M:
        # If we use k=9, max diff is M.
        # Is M=30 possible?
        
        def dfs(S, diffs, remaining, M):
            if time.time() > deadline:
                return None
            
            if not remaining:
                return S
            
            # Pruning:
            # If we can't cover the largest remaining diff, fail?
            # Max possible new diff from adding x <= M is M - min(S) = M.
            # So if max(remaining) > M, it's already handled by M loop.
            
            # Heuristic: try to cover large diffs first?
            # Or just iterate x.
            
            last = S[-1]
            # We can add x from last+1 to M
            # Optimization: If we have many elements left to add, we might want to pick x that
            # covers many remaining diffs.
            
            # Simple iteration
            for x in range(last + 1, M + 1):
                new_diffs = []
                valid = True
                for s in S:
                    d = x - s
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Check if any new diff is > M? No, x<=M, s>=0.
                
                new_diffs_set = set(new_diffs)
                new_remaining = remaining - new_diffs_set
                new_diffs_all = diffs | new_diffs_set
                
                # Recursive call
                res = dfs(S + [x], new_diffs_all, new_remaining, M)
                if res is not None:
                    return res
            
            return None

        # Start with S=[0]
        # To improve speed, we can try to fix the first element?
        # No, just let DFS decide.
        
        # But iterating M from 30 up might be slow if 30 fails.
        # Let's estimate M.
        # For k=9, average diff is ~ M/2.
        # We need 30 diffs.
        
        # Let's try M from 30 to 100.
        best_solution = None
        for M in range(30, 101):
            if time.time() > deadline:
                break
            # Reset state
            S = [0]
            diffs = set()
            remaining = set(required)
            
            res = dfs(S, diffs, remaining, M)
            if res is not None:
                best_solution = res
                break
        
        if best_solution is None:
            # Fallback to a known valid set or greedy if timeout
            # Greedy from seed
            a = [0]
            def diffs_list(s):
                return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
            for d in range(1, 31):
                if d in diffs_list(sorted(a)):
                    continue
                for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                    trial = sorted(a + [x])
                    if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                        a = trial
                        break
                else:
                    y = 2 * max(a) + 3 * d + 1
                    while True:
                        trial = sorted(a + [y, y + d])
                        if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                            a = trial
                            break
                        y += 1
            return a
            
        return best_solution
