import time
from math import lcm

def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        
        # Precompute validity
        N = 256
        valid = [[False] * (N + 1) for _ in range(N + 1)]
        for i in range(2, N + 1):
            for j in range(i + 1, N + 1):
                valid[i][j] = lcm(i, j) > 256
                valid[j][i] = valid[i][j]
                
        def is_valid_set(S):
            # S is a set of ints
            # Check all pairs
            lst = list(S)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if not valid[lst[i]][lst[j]]:
                        return False
            return True

        def score(S):
            return sum(1.0 / x for x in S)

        best_S = set()
        best_score = 0.0
        
        # Try different seeds
        # Seed 1: Greedy from 2
        # Seed 2: Greedy from 3
        # ...
        
        def greedy(start_seed):
            S = set()
            if start_seed:
                S.add(start_seed)
            for x in range(2, N + 1):
                if x in S:
                    continue
                # Check if x is valid with all in S
                if all(valid[x][y] for y in S):
                    S.add(x)
            return S

        candidates = []
        # Try starting with 0 (none), 2, 3, ..., 10
        for seed in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            S = greedy(seed)
            candidates.append(S)
            
        # Also try a random-ish greedy?
        # Sort by 1/x and pick? That's just ascending x.
        
        # Evaluate candidates
        for S in candidates:
            sc = score(S)
            if sc > best_score:
                best_score = sc
                best_S = set(S)
                
        # Local Search on best_S
        S = set(best_S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add
            for x in range(2, N + 1):
                if x in S:
                    continue
                if all(valid[x][y] for y in S):
                    S.add(x)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. Try to swap
            # We want to increase score, so replace x with y < x.
            # Iterate y from 2 to N
            # If y not in S, check if there exists x in S, x > y, such that S - {x} + {y} is valid.
            
            found_swap = False
            for y in range(2, N + 1):
                if y in S:
                    continue
                # Optimization: If y is incompatible with > 1 element in S, 
                # we might need to remove multiple? No, swap is 1-for-1.
                # So y must be compatible with S \ {x}.
                
                # Find candidates x in S such that x > y and y is compatible with S \ {x}
                # This means for all z in S, z != x, valid[y][z] is True.
                # Equivalently, the set of z in S where valid[y][z] is False must be subset of {x}.
                
                bad_z = [z for z in S if not valid[y][z]]
                
                if len(bad_z) == 0:
                    # y is compatible with all S.
                    # We can swap with any x > y.
                    # Best swap: remove largest x? No, we want to keep small x?
                    # We want to maximize sum(1/x).
                    # Replacing x with y (y < x) increases score.
                    # We should remove the x that "costs" the least in terms of future flexibility?
                    # Or just remove any x > y.
                    # Since we just added y, and y is compatible with all, 
                    # we can remove the largest x to free up space?
                    # Actually, removing a large x doesn't hurt compatibility much.
                    # Removing a small x might hurt.
                    # So remove max(S) if max(S) > y.
                    max_x = max(S)
                    if max_x > y:
                        S.remove(max_x)
                        S.add(y)
                        improved = True
                        break
                        
                elif len(bad_z) == 1:
                    # y is incompatible with exactly one z in S.
                    # We must remove that z.
                    x = bad_z[0]
                    if x > y:
                        # Check if removing x and adding y is valid.
                        # We know y is compatible with S \ {x} by definition of bad_z.
                        # And x is removed.
                        # So it's valid.
                        S.remove(x)
                        S.add(y)
                        improved = True
                        break
            if found_swap:
                continue
                
            # 3. Try to remove 1 and add 2?
            # If S has an element x that blocks many small numbers, maybe remove it.
            # This is harder to search.
            
            # If no improvement, break.
            
        return list(S)
