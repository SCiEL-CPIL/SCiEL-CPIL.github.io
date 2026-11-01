"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Return the verified rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B indices: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B10, B[5]=B11, B[6]=B12, B[7]=B13
    # C indices (W): C[0]=C00, C[1]=C10, C[2]=C01, C[3]=C11, C[4]=C02, C[5]=C12, C[6]=C03, C[7]=C13
    
    # Left block (cols 0,1) - Strassen
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Right block (cols 2,3) - Strassen
    # N1 = (A00+A11)(B02+B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2 = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3 = A00(B03-B13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4 = A11(B12-B02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5 = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6 = (A10-A00)(B02+B03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7 = (A01-A11)(B12+B13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 4
    rank = len(U[0])
    
    # Build the tensor from U, V, W: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # For each output element C[k], verify it equals sum over A[i]*B[j]
    # C[k] = sum_i,j T[i,j,k] * A[i] * B[j]
    # We need T[i,j,k] = 1 if C[k] = A[i]*B[j], 0 otherwise
    
    # A[i] where i = a*n + b (a=row, b=col)
    # B[j] where j = b'*p + c (b'=row, c=col)
    # C[k] where k = c'*n + a' (c'=col, a'=row)
    
    # C[a',c'] = sum_b A[a,b] * B[b,c']
    # So T[i,j,k] should be 1 iff:
    #   i = a*n + b, j = b*p + c', k = c'*n + a'
    #   i.e., a' = a, b' = b, c' = c'
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                # C[a,c] = A[a,b]*B[b,c]
                i = a * m + b  # A index
                j = b * p + c  # B index
                k = c * n + a  # C index
                
                # Check T[i,j,k]
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Also check that T[i,j,k] = 0 for non-matching triples
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                # Check if this is a valid multiplication term
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue  # Already checked above
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i][r] * V[j][r] * W[k][r]
                
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def _compute_error(U, V, W, target_rank):
    """Compute the squared error of the tensor decomposition."""
    n, m, p = 2, 2, 4
    error = 0.0
    # Correct terms: T[i,j,k] = 1 if i=a*m+b, j=b*p+c, k=c*n+a
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(target_rank))
                error += (t_val - 1.0) ** 2
    
    # Incorrect terms: T[i,j,k] = 0 otherwise
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                a = i // m
                b = i % m
                b_prime = j // p
                c = j % p
                c_prime = k // n
                a_prime = k % n
                
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(target_rank))
                error += t_val ** 2
    return error

def _try_optimize_rank(target_rank, start_time, deadline):
    """Attempt to find a valid decomposition of the given rank using optimization."""
    import numpy as np
    from scipy.optimize import minimize
    
    n, m, p = 2, 2, 4
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    # Flatten U, V, W into a single vector for optimization
    # We start from the rank-14 solution, zeroing out one term to force rank reduction
    # This is a heuristic: take rank-14, remove term 'drop_idx', and optimize the rest
    
    base = _build_rank14()
    U_base = np.array(base["U"], dtype=float)
    V_base = np.array(base["V"], dtype=float)
    W_base = np.array(base["W"], dtype=float)
    
    # We will try dropping each of the 14 terms one by one
    for drop_idx in range(14):
        if time.time() > deadline - 5:
            return None
            
        # Create initial guess: copy rank-14, but set column 'drop_idx' to 0
        # The remaining 13 columns will be optimized
        U_init = U_base.copy()
        V_init = V_base.copy()
        W_init = W_base.copy()
        U_init[:, drop_idx] = 0.0
        V_init[:, drop_idx] = 0.0
        W_init[:, drop_idx] = 0.0
        
        # Reshape to 1D
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[:size_U * target_rank].reshape((size_U, target_rank))
            V = x[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
            W = x[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
            # Convert to lists for _compute_error or just compute directly with numpy
            # Direct numpy computation is faster
            # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # We want T to match the multiplication tensor
            
            err = 0.0
            # Correct terms
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a * m + b
                        j = b * p + c
                        k = c * n + a
                        t_val = np.dot(U[i], np.multiply(V[j], W[k]))
                        err += (t_val - 1.0) ** 2
                        
            # Incorrect terms
            for i in range(size_U):
                for j in range(size_V):
                    for k in range(size_W):
                        a = i // m
                        b = i % m
                        b_prime = j // p
                        c = j % p
                        c_prime = k // n
                        a_prime = k % n
                        if a == a_prime and b == b_prime:
                            continue
                        t_val = np.dot(U[i], np.multiply(V[j], W[k]))
                        err += t_val ** 2
            return err
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
            if res.success and res.fun < 1e-8:
                # Extract solution
                x_opt = res.x
                U_opt = x_opt[:size_U * target_rank].reshape((size_U, target_rank))
                V_opt = x_opt[size_U * target_rank : size_U * target_rank + size_V * target_rank].reshape((size_V, target_rank))
                W_opt = x_opt[size_U * target_rank + size_V * target_rank :].reshape((size_W, target_rank))
                
                # Check if entries are close to integers or half-integers
                # The problem statement says "integer/half-integer entries"
                # We round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify the rounded solution
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Start with the known rank-14 solution
    best_result = _build_rank14()
    
    # Verify the base solution
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        # Try to find a lower-rank solution
        for target_rank in [13, 12]:
            if time.time() > deadline - 10:
                break
            
            result = _try_optimize_rank(target_rank, start_time, deadline)
            if result:
                return result
    
    # Return the best verified solution
    return best_result
# EVOLVE-BLOCK-END
