import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    n, m, p = n_, m_, p_ = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def check(U, V, W):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_remove_term(U, V, W, col_to_remove):
        """Try to remove the term at col_to_remove by adjusting others."""
        rank = len(U[0])
        if rank <= 1:
            return None
        
        # Create a copy without the term
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        # Remove the column
        for row in U_new:
            del row[col_to_remove]
        for row in V_new:
            del row[col_to_remove]
        for row in W_new:
            del row[col_to_remove]
        
        new_rank = rank - 1
        
        # Now we need to adjust the remaining terms to compensate.
        # This is a hard problem. For now, just check if it's valid.
        if check(U_new, V_new, W_new):
            return U_new, V_new, W_new
        
        # Try to fix by adjusting one term.
        # For each remaining column c, try to adjust U, V, W entries.
        # This is a local search.
        
        # Simple approach: for each (i,j,k) that's now wrong, 
        # try to fix it by adjusting a nearby term.
        
        # For now, return None if we can't easily fix it.
        return None
    
    U, V, W = schoolbook()
    
    # Try to remove terms
    max_iter = 1000
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        rank = len(U[0])
        removed = False
        
        # Try each column
        for c in range(rank):
            result = try_remove_term(U, V, W, c)
            if result is not None:
                U, V, W = result
                removed = True
                break
        
        if not removed:
            break
    
    return {"U": U, "V": V, "W": W}
