import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n from high to low
    # We know n=11 is valid.
    # Let's try 20 down to 11.
    for n in range(20, 10, -1):
        if time.time() > deadline - 5:
            break
        
        # Time limit per n
        limit_n = time.time() + 10
        
        # Try to find a coloring for n
        # Use multiple random restarts for each n
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > limit_n or time.time() > deadline - 2:
                break
            
            # Randomize seed for this restart
            random.seed(restart * 1000 + n)
            
            success, colors = solve_n(n, deadline)
            if success:
                return {'n': n, 'colors': colors}
        
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    A = [[0] * n for _ in range(5)]
    
    def backtrack(v):
        if time.time() > deadline:
            return False, None
            
        if v == n:
            # Construct final colors list
            res_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    for c in range(5):
                        if A[c][i] & (1 << j):
                            res_colors.append(c)
                            break
            return True, res_colors

        # Assign colors to edges (v, u) for u in v-1..0
        assigned = [-1] * v
        
        def assign(u):
            if time.time() > deadline:
                return False
            if u < 0:
                # All assigned for vertex v
                # Save state to allow undo
                old_A = []
                for c in range(5):
                    old_A.append(A[c][v])
                    for u2 in range(v):
                        old_A.append(A[c][u2])
                        
                # Update A
                for u2 in range(v):
                    c = assigned[u2]
                    A[c][u2] |= (1 << v)
                    A[c][v] |= (1 << u2)
                
                # Recurse
                if backtrack(v + 1):
                    return True
                
                # Undo
                idx = 0
                for c in range(5):
                    A[c][v] = old_A[idx]
                    idx += 1
                    for u2 in range(v):
                        A[c][u2] = old_A[idx]
                        idx += 1
                        
                return False
            
            # Try colors for edge (v, u)
            colors_to_try = list(range(5))
            random.shuffle(colors_to_try)
            
            # Heuristic: if v==1 and u==0, force color 0
            if v == 1 and u == 0:
                colors_to_try = [0]
                
            for c in colors_to_try:
                # Check validity
                valid = True
                for w in range(u + 1, v):
                    if assigned[w] == c:
                        if A[c][w] & A[c][u]:
                            valid = False
                            break
                if valid:
                    assigned[u] = c
                    if assign(u - 1):
                        return True
                    assigned[u] = -1
            return False
        
        return assign(v - 1)

    # Start with vertex 0
    success, colors = backtrack(0)
    return success, colors
