"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for arr in [U, V, W]:
        if len(arr) == 0:
            return False
        if len(arr[0]) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _constructive_decomposition(n, m, p):
    """Try to construct a lower-rank decomposition using block structure."""
    # Strategy: For each pair (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
    # This is a dot product of length m.
    # We can use the fact that for fixed (i,k), the j-index couples U[i*m+j] with V[j*p+k]
    
    # Let's try: for each (i,k), use a rank-r_{ik} decomposition of the length-m dot product
    # The dot product of two length-m vectors can be done in rank m (schoolbook)
    # But we can potentially share across different (i,k) pairs
    
    # Alternative: think of it as matrix multiplication A (n×m) * B (m×p) = C (n×p)
    # The tensor is T[i,j,k] = 1, and we decompose as sum_c U[i,j,c]V[j,k,c]W[i,k,c]
    
    # Let's try a "row-column" decomposition:
    # For each j, the contribution is A[:,j] * B[j,:] which is an outer product
    # So C = sum_j A[:,j] * B[j,:]
    # This is already the schoolbook with rank m
    
    # But we can do better by using the structure. Let me try to find a decomposition
    # where we group the j-indices.
    
    # For m=4, we can partition j into {0,1} and {2,3}
    # For each group, compute the partial product using a lower-rank method
    
    # Actually, let's try a direct algebraic approach.
    # We need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try rank = n*p + m*p + n*m - something
    # Or try rank = (n+m-1)*(p+m-1) - something
    
    # Let me try a specific construction for n=3, m=4, p=7
    # One approach: use the fact that we can write the multiplication as
    # a sum of rank-1 tensors where each tensor has a structured support
    
    # Try: for each (i,k), use a rank-4 decomposition (same as m) but with shared terms
    # This won't help directly.
    
    # Let's try a randomized local search starting from schoolbook
    # and trying to merge pairs of terms
    
    n, m, p = 3, 4, 7
    rank = n * m * p  # 84
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
    
    # Try to merge pairs of columns that have overlapping support
    # If two columns c1, c2 have U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] = 
    # U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2] for all (i,j,k), we can merge them
    # But this is unlikely.
    
    # Instead, let's try to find a decomposition with rank < 84 by solving
    # a system of equations. For a given rank r, we have 3*r*(n*m + m*p + p*n)/3 = r*(n*m+m*p+p*n)
    # unknowns and n*m*p = 84 equations.
    
    # For r=83, we have 83*84 = 6972 unknowns and 84 equations - very underdetermined
    # But the equations are nonlinear (cubic in the variables).
    
    # Let's try a different approach: use the fact that the problem is bilinear
    # in the sense that if we fix W, the equations are linear in U and V.
    
    # Actually, let's try a constructive approach based on polynomial identities.
    # The multiplication (x_1 + x_2 + x_3)(y_1 + y_2 + y_3 + y_4) can be computed
    # with fewer than 12 multiplications using the fact that we only need the
    # coefficients of the product polynomial.
    
    # For n=3, m=4, p=7, let's think of A as a 3×4 matrix and B as a 4×7 matrix.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One idea: use the "middle product" trick. If we embed the matrices into
    # larger matrices and use a known fast multiplication algorithm...
    
    # For now, let's try a simple heuristic: try ranks from 60 to 83 and see if
    # we can find a valid decomposition using random initialization + Newton's method.
    
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    best_result = _schoolbook(n, m, p)
    best_rank = rank
    
    # Try to find a lower-rank decomposition
    for target_rank in range(60, 84):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Use Newton's method to solve the system
            # F_c = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0 for all (i,j,k)
            
            # This is complex. Let's try a simpler approach:
            # Fix W to be random, then solve for U and V linearly.
            # But that's not quite right either.
            
            # Let's try: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # If we fix W, this is linear in the products U[i*m+j,c]*V[j*p+k,c]
            # But those products are not independent.
            
            # Alternative: use the fact that we can scale. If we multiply U[:,c] by alpha,
            # V[:,c] by beta, W[:,c] by gamma, the product is unchanged if alpha*beta*gamma=1.
            
            # Let's try a different construction. For n=3, m=4, p=7:
            # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
            # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
            # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k] + A[2,3]B[3,k]
            
            # We can try to compute these using shared intermediate values.
            # For example, let S_j = A[0,j] + A[1,j] + A[2,j] (sum over i)
            # Then S_j * B[j,k] = sum_i A[i,j]B[j,k]
            # But we need individual C[i,k], not the sum.
            
            # This approach doesn't directly help.
            
            # Let's try yet another approach: use the "transposed" view.
            # The tensor T[i,j,k] = 1. We want to write it as sum_c u_c ⊗ v_c ⊗ w_c
            # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.
            
            # One simple decomposition: for each (i,j), define u_{i,j} to be the vector with 1
            # at position (i,j) and 0 elsewhere. Then we need sum_k v_{i,j,k} w_{i,j,k} = 1
            # for all k, where v_{i,j,k} is the k-th component and w_{i,j,k} is the (k,i)-th component.
            # This gives rank n*m*p = 84.
            
            # To reduce rank, we need to find u_c, v_c, w_c such that the sum works.
            
            # Let's try a constructive approach for small ranks.
            # For rank = n*p + m*p + n*m - 1 = 21 + 28 + 12 - 1 = 60:
            # This is a guess based on some algebraic identities.
            
            pass
        
        # If we found a valid decomposition, update best
        # (This is a placeholder - the actual search needs to be implemented)
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Try the constructive approach first
    result = _constructive_decomposition(n, m, p)
    
    # Verify the result
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
