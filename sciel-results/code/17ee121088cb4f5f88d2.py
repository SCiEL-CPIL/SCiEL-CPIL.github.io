import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    num_colors = 5
    
    def solve(n, time_limit):
        # Random Initialization
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, num_colors-1)
                col[i][j] = c
                col[j][i] = c
        
        # Adjacency sets
        adj = [ [set() for _ in range(n)] for _ in range(num_colors) ]
        for i in range(n):
            for j in range(i+1, n):
                c = col[i][j]
                adj[c][i].add(j)
                adj[c][j].add(i)
                
        # Common neighbor counts
        # common[c][u][v]
        common = [ [[0]*n for _ in range(n)] for _ in range(num_colors) ]
        for c in range(num_colors):
            for u in range(n):
                for v in range(u+1, n):
                    # Count intersection
                    # Using sets is O(k)
                    common[c][u][v] = len(adj[c][u] & adj[c][v])
                    common[c][v][u] = common[c][u][v]
                    
        # Penalty
        penalty = 0
        for c in range(num_colors):
            for u in range(n):
                for v in range(u+1, n):
                    k = common[c][u][v]
                    if k >= 2:
                        penalty += k * (k - 1) // 2
                        
        best_penalty = penalty
        best_col = [row[:] for row in col]
        
        no_improve = 0
        max_no_improve = 20000 # Higher limit, delta is fast
        
        while time.time() < time_limit:
            if penalty == 0:
                return col
                
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c: continue
            
            delta_p = 0
            
            # Remove edge (i,j) from old_c
            # 1. x in N_old(i) \ {j}: common[old_c][j][x] decreases by 1
            #    Penalty change: -(common[old_c][j][x] - 1)
            # 2. x in N_old(j) \ {i}: common[old_c][i][x] decreases by 1
            #    Penalty change: -(common[old_c][i][x] - 1)
            
            # Note: We must use the CURRENT common values before update.
            
            for x in adj[old_c][i]:
                if x == j: continue
                a, b = (i, x) if i < x else (x, i)
                # We are updating common[old_c][j][x]? No.
                # If x in N(i), then i is a common neighbor of j and x?
                # No. i is a neighbor of x. Is i a neighbor of j? Yes (edge ij).
                # So i was a common neighbor of j and x.
                # After removing ij, i is no longer neighbor of j.
                # So common[j][x] decreases by 1.
                # Pair is (j, x).
                a, b = (j, x) if j < x else (x, j)
                delta_p -= (common[old_c][a][b] - 1)
                
            for x in adj[old_c][j]:
                if x == i: continue
                # x in N(j). j is neighbor of x. Is j neighbor of i? Yes.
                # So j was common neighbor of i and x.
                # After removing ij, j is no longer neighbor of i.
                # So common[i][x] decreases by 1.
                # Pair is (i, x).
                a, b = (i, x) if i < x else (x, i)
                delta_p -= (common[old_c][a][b] - 1)
                
            # Add edge (i,j) to new_c
            # 1. x in N_new(i) \ {j}: common[new_c][j][x] increases by 1
            #    Penalty change: + common[new_c][j][x]
            # 2. x in N_new(j) \ {i}: common[new_c][i][x] increases by 1
            #    Penalty change: + common[new_c][i][x]
            
            for x in adj[new_c][i]:
                if x == j: continue
                a, b = (j, x) if j < x else (x, j)
                delta_p += common[new_c][a][b]
                
            for x in adj[new_c][j]:
                if x == i: continue
                a, b = (i, x) if i < x else (x, i)
                delta_p += common[new_c][a][b]
                
            if delta_p < 0:
                # Accept
                col[i][j] = new_c
                col[j][i] = new_c
                
                # Update adj
                adj[old_c][i].remove(j)
                adj[old_c][j].remove(i)
                adj[new_c][i].add(j)
                adj[new_c][j].add(i)
                
                # Update common
                # old_c
                for x in adj[old_c][i]: # This is N(i) \ {j} now? 
                    # Wait, we removed j from adj[old_c][i]. 
                    # So adj[old_c][i] is now N(i) \ {j}.
                    # But we need to update common for x in N_old(i) \ {j}.
                    # The set adj[old_c][i] IS that set.
                    # However, we also need to update common[j][x] and common[x][j].
                    # And common[i][x] and common[x][i]?
                    # No, common[i][x] is NOT affected by removing ij?
                    # Let's re-verify.
                    # common[u][v] = |N(u) \cap N(v)|.
                    # Remove edge (i,j).
                    # common[j][x] for x in N(i) \ {j}:
                    #   N(j) loses i. N(x) unchanged.
                    #   Intersection loses i (if i in N(x)).
                    #   i in N(x) is true because x in N(i) (symmetric).
                    #   So common[j][x] decreases by 1.
                    # common[i][x] for x in N(j) \ {i}:
                    #   N(i) loses j. N(x) unchanged.
                    #   Intersection loses j (if j in N(x)).
                    #   j in N(x) is true because x in N(j).
                    #   So common[i][x] decreases by 1.
                    
                    # So:
                    # For x in N_old(i) \ {j}: decrement common[j][x] and common[x][j].
                    # For x in N_old(j) \ {i}: decrement common[i][x] and common[x][i].
                    
                    pass
                
                # Correct Update:
                # We need the OLD neighbors.
                # We can just iterate over the current adj sets which are already updated?
                # No, we need to know which x were in the OLD sets.
                # Since we only removed j from i and i from j, the rest of the set is the same.
                # So adj[old_c][i] (current) is N_old(i) \ {j}.
                # adj[old_c][j] (current) is N_old(j) \ {i}.
                
                for x in adj[old_c][i]:
                    # x in N_old(i) \ {j}
                    # Decrement common[j][x]
                    common[old_c][j][x] -= 1
                    common[old_c][x][j] -= 1
                    
                for x in adj[old_c][j]:
                    # x in N_old(j) \ {i}
                    # Decrement common[i][x]
                    common[old_c][i][x] -= 1
                    common[old_c][x][i] -= 1
                    
                # new_c
                # We added j to i and i to j.
                # adj[new_c][i] (current) is N_new(i) \cup {j}.
                # But we need to update for x in N_new(i) \ {j} (the ones that were there BEFORE).
                # This is adj[new_c][i] \ {j}.
                # Since we just added j, we can remove it for iteration?
                # Or just check if x == j.
                
                for x in adj[new_c][i]:
                    if x == j: continue
                    # x in N_new(i) \ {j}
                    # Increment common[j][x]
                    common[new_c][j][x] += 1
                    common[new_c][x][j] += 1
                    
                for x in adj[new_c][j]:
                    if x == i: continue
                    # x in N_new(j) \ {i}
                    # Increment common[i][x]
                    common[new_c][i][x] += 1
                    common[new_c][x][i] += 1
                    
                penalty += delta_p
                no_improve = 0
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_col = [row[:] for row in col]
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        if best_penalty == 0:
            return best_col
        return None

    # Try n=26
    # Allocate time wisely.
    # If we find 26, return it.
    # If not, try 25, etc.
    
    # We'll just try 26 with most of the time.
    # If it fails, we return a small valid one.
    
    # Small valid fallback: n=6?
    # Let's just return a random valid one if we find it quickly for small n.
    
    # For now, try 26.
    sol = solve(26, deadline - 5)
    if sol:
        colors = []
        n = 26
        for i in range(n):
            for j in range(i+1, n):
                colors.append(sol[i][j])
        return {'n': n, 'colors': colors}
        
    # Fallback: Try to find a small valid solution quickly
    # n=11 is known to be easy.
    sol = solve(11, deadline - 2)
    if sol:
        colors = []
        n = 11
        for i in range(n):
            for j in range(i+1, n):
                colors.append(sol[i][j])
        return {'n': n, 'colors': colors}
        
    # If all fails, return a trivial valid solution (n=1)
    return {'n': 1, 'colors': []}
