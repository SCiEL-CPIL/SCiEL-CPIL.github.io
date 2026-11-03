import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Use a structured algebraic construction
    # The matrix multiplication tensor has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We'll try to find a low-rank decomposition using a greedy approach
    
    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j, j*p+k, k*n+i)] = 1
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But we need at least enough to satisfy all constraints
    # Try ranks from 25 to 40
    
    for rank in range(25, 41):
        # Initialize with zeros
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Greedy: for each target entry, try to satisfy it with a new column
        # or adjust existing columns
        satisfied = set()
        
        for col in range(rank):
            # Find an unsatisfied target
            remaining = [(a,b,c) for (a,b,c) in target if (a,b,c) not in satisfied]
            if not remaining:
                break
            
            # Pick the first remaining target
            a, b, c = remaining[0]
            
            # Set U[a,col], V[b,col], W[c,col] to 1
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[c, col] = 1.0
            
            # Mark all targets that are now satisfied by this column
            # A target (a',b',c') is satisfied if U[a',col]*V[b',col]*W[c',col] = 1
            # which means U[a',col]=1, V[b',col]=1, W[c',col]=1
            for (a_t, b_t, c_t) in target:
                if U[a_t, col] * V[b_t, col] * W[c_t, col] == 1.0:
                    satisfied.add((a_t, b_t, c_t))
        
        # Check if all targets are satisfied
        if len(satisfied) == len(target):
            # Verify exactness: for all (a,b,c) in target, sum should be 1
            # and for non-target positions, sum should be 0
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for col in range(rank):
                            val += U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                        if abs(val - 1) > 1e-10:
                            error += (val - 1)**2
            
            # Check non-target positions
            all_positions = set()
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        all_positions.add((a,b,c))
            
            for pos in all_positions:
                if pos not in target:
                    a, b, c = pos
                    val = 0
                    for col in range(rank):
                        val += U[a, col] * V[b, col] * W[c, col]
                    if abs(val) > 1e-10:
                        error += val**2
            
            if error < 1e-10:
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
