"""
Strategy: Alternating Least Squares (ALS) with integer rounding.
Unlike the block-based combinatorial search, this treats the decomposition as a 
tensor fitting problem. We search for integer matrices U, V, W (scaled by 2 to 
represent half-integers) such that the tensor contraction matches the target.
We start from a low rank and increase it until a valid exact decomposition is found.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T is all 1s.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _als_step(U, V, W, n, m, p, target_val):
    """
    Perform one ALS step. 
    U, V, W are numpy arrays of shape (n*m, R), (m*p, R), (p*n, R).
    target_val is the value we want the sum of products to equal (usually 8 for integer scaling).
    Returns updated U, V, W.
    """
    R = U.shape[1]
    
    # 1. Update U
    # For each row i*m+j of U, we want:
    # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])_summed_over_k? No.
    # The equation is: sum_k sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = target_val * n? No.
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = target_val.
    # This is a linear system for U[i*m+j, :] given V, W.
    # Let A_c = V[j*p+k, c] * W[k*n+i, c]. This depends on k.
    # So for fixed i,j, we have a system of p equations (for k=0..p-1) and R unknowns (U[i*m+j, 0..R-1]).
    # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = target_val for all k.
    
    # Construct matrix M_U of shape (p, R)
    # M_U[k, c] = V[j*p+k, c] * W[k*n+i, c]
    # Solve M_U @ U[i*m+j, :] = target_val * ones(p)
    
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, R))
            for k in range(p):
                for c in range(R):
                    M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            b = np.full(p, target_val)
            # Solve M @ x = b
            # Use least squares if M is not square/invertible
            try:
                x = np.linalg.lstsq(M, b, rcond=None)[0]
            except:
                x = np.zeros(R)
            U[i*m+j, :] = x
            
    # 2. Update V
    # For each row j*p+k of V:
    # sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = target_val for all i.
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, R))
            for i in range(n):
                for c in range(R):
                    M[i, c] = U[i*m+j, c] * W[k*n+i, c]
            b = np.full(n, target_val)
            try:
                x = np.linalg.lstsq(M, b, rcond=None)[0]
            except:
                x = np.zeros(R)
            V[j*p+k, :] = x
            
    # 3. Update W
    # For each row k*n+i of W:
    # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = target_val for all j.
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, R))
            for j in range(m):
                for c in range(R):
                    M[j, c] = U[i*m+j, c] * V[j*p+k, c]
            b = np.full(m, target_val)
            try:
                x = np.linalg.lstsq(M, b, rcond=None)[0]
            except:
                x = np.zeros(R)
            W[k*n+i, :] = x
            
    return U, V, W

def _round_and_check(U, V, W, n, m, p, target_val):
    """
    Round U, V, W to nearest integers (representing 2 * half-integers).
    Check if the rounded version satisfies the equation exactly.
    Returns (rounded_U, rounded_V, rounded_W, is_valid)
    """
    U_r = np.rint(U).astype(int)
    V_r = np.rint(V).astype(int)
    W_r = np.rint(W).astype(int)
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(U_r.shape[1]):
                    s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                if s != target_val:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return U_r, V_r, W_r, True
    else:
        return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # We search for integer solutions to T = 8 (since 1/2 * 1/2 * 1/2 = 1/8, so 8 * (1/8) = 1)
    # Wait, if U,V,W are integers representing 2*x, 2*y, 2*z, then x=U/2, etc.
    # Product = (U/2)*(V/2)*(W/2) = U*V*W / 8.
    # Sum = 1 => Sum(U*V*W) = 8.
    target_val = 8
    
    # Start from a rank lower than the known best (88).
    # Theoretical lower bound for 3x4x8? 
    # Schoolbook is 96.
    # Let's try from 80 upwards.
    
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 80 to 96
    for rank in range(80, 97):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple random restarts for each rank
        for restart in range(20):
            if time.time() > deadline:
                break
                
            random.seed(42 + rank + restart)
            
            # Initialize U, V, W with small random integers
            # Scale: values around 1-3
            U = np.random.randint(-3, 4, (n*m, rank))
            V = np.random.randint(-3, 4, (m*p, rank))
            W = np.random.randint(-3, 4, (p*n, rank))
            
            # Run ALS for a few iterations
            for it in range(50):
                U, V, W = _als_step(U, V, W, n, m, p, target_val)
                
                # Periodically check if rounding works
                if it % 5 == 0:
                    Ur, Vr, Wr, valid = _round_and_check(U, V, W, n, m, p, target_val)
                    if valid:
                        # Convert back to half-integers for output? 
                        # The task says "integer/half-integer entries".
                        # If we found integers U_r, then the actual factors are U_r/2.
                        # Let's return U_r/2.
                        U_out = Ur / 2.0
                        V_out = Vr / 2.0
                        W_out = Wr / 2.0
                        
                        # Final verification with the returned values
                        if _verify(U_out, V_out, W_out, n, m, p):
                            if rank < best_rank:
                                best_rank = rank
                                best_U = U_out
                                best_V = V_out
                                best_W = W_out
                            found = True
                            break
            
            if found:
                break # Found solution for this rank, move to next (lower) rank? No, we are iterating 80->96. 
                # If we found rank 80, we should stop and return 80.
                # But we want the MINIMAL rank. So if we find 80, that's great.
                # If we don't find 80, we try 81, etc.
                # So if found, we can break the rank loop entirely if we are going up?
                # No, we are going up. The first one we find is the lowest.
                break

    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
        
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
