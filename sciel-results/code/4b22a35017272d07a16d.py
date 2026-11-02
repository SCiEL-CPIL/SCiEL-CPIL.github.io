"""Matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook decomposition: rank = n*m*p = 100 (guaranteed valid)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def build_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that the factorization U, V, W reconstructs T."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        T_rec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        error = np.max(np.abs(T_rec - T))
        return error < tol, error

    def round_to_half_integer(arr):
        """Round array entries to nearest half-integer (multiple of 0.5)."""
        return np.round(arr * 2) / 2

    def try_rank_als(R, T, max_iter=500, n_restarts=3, tol=1e-8):
        """Try to find a rank-R decomposition using ALS."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n * m, m * p, p * n
        
        best_U = None
        best_V = None
        best_W = None
        best_err = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(42 + restart)
            U = rng.randn(nm, R)
            V = rng.randn(mp, R)
            W = rng.randn(pn, R)
            
            U /= np.sqrt(R)
            V /= np.sqrt(R)
            W /= np.sqrt(R)
            
            prev_err = np.inf
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
                M = np.einsum('abc,br->acr', T, V)
                numer_U = np.einsum('acr,cr->ar', M, W)
                U_new = numer_U / denom_U[np.newaxis, :]
                
                # Update V
                denom_V = np.sum(U_new**2, axis=0) * np.sum(W**2, axis=0)
                M2 = np.einsum('abc,ar->bcr', T, U_new)
                numer_V = np.einsum('bcr,cr->br', M2, W)
                V_new = numer_V / denom_V[np.newaxis, :]
                
                # Update W
                denom_W = np.sum(U_new**2, axis=0) * np.sum(V_new**2, axis=0)
                M3 = np.einsum('abc,br->acr', T, V_new)
                numer_W = np.einsum('acr,ar->cr', M3, U_new)
                W_new = numer_W / denom_W[np.newaxis, :]
                
                T_rec = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                err = np.max(np.abs(T_rec - T))
                
                U, V, W = U_new, V_new, W_new
                
                if abs(prev_err - err) < 1e-10:
                    break
                prev_err = err
            
            # Round to half-integers and verify
            U_rounded = round_to_half_integer(U)
            V_rounded = round_to_half_integer(V)
            W_rounded = round_to_half_integer(W)
            
            T_rec_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            err_rounded = np.max(np.abs(T_rec_rounded - T))
            
            if err_rounded < 1e-6:
                U_frac = [[Fraction(int(U_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(nm)]
                V_frac = [[Fraction(int(V_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(mp)]
                W_frac = [[Fraction(int(W_rounded[i, r] * 2)) / 2 for r in range(R)] for i in range(pn)]
                
                if err < best_err:
                    best_U, best_V, best_W = U_frac, V_frac, W_frac
                    best_err = err
        
        if best_err < 1e-6:
            return best_U, best_V, best_W, best_err
        return None, None, None, np.inf

    # Try block-based decomposition using Strassen for 2x2 blocks
    # Strategy: Partition the matrices into blocks and use known low-rank decompositions
    # For 4x5 * 5x5:
    # A is 4x5, B is 5x5, C is 4x5
    # We can partition A into [A1 (2x5); A2 (2x5)]
    # B is 5x5 (keep as is)
    # C = A*B = [A1*B; A2*B]
    
    # Alternative: Use block structure
    # A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # Wait, this doesn't work cleanly. Let me think differently.
    
    # Better approach: Use the fact that 4x5 * 5x5 can be decomposed
    # by viewing it as multiple smaller multiplications
    
    # Method 1: Split by rows of A
    # C[0:2, :] = A[0:2, :] * B  (2x5 * 5x5 -> 2x5)
    # C[2:4, :] = A[2:4, :] * B  (2x5 * 5x5 -> 2x5)
    # Each 2x5 * 5x5 has rank at most 2*5*5 = 50, but we can do better
    # Actually, for 2x5 * 5x5, we can use:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, k in {0,...,4}
    # This is 2*5 = 10 independent bilinear forms
    # Each requires at most 5 multiplications (one for each j)
    # So total rank for this block is 2*5*5 = 50
    # But we can share: for a fixed j, A[0,j]*B[j,k] and A[1,j]*B[j,k] 
    # are two separate terms, so we still need 5 multiplications per row pair
    # Total: 2 * 5 * 5 = 50 for the first block, same for second = 100 total (schoolbook)
    
    # Method 2: Use Strassen-like decomposition
    # For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8
    # Can we apply this to our 4x5 * 5x5?
    
    # Let's try: A is 4x5, B is 5x5
    # Partition A into 2 blocks of 2x5
    # Partition B into 2 blocks: first 2 columns (5x2) and last 3 columns (5x3)
    # C[:, 0:2] = A * B[:, 0:2]  (4x5 * 5x2 -> 4x2)
    # C[:, 2:5] = A * B[:, 2:5]  (4x5 * 5x3 -> 4x3)
    
    # For 4x5 * 5x2:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..1
    # This is 4*2 = 8 bilinear forms, each sum of 5 products
    # Schoolbook: 4*5*2 = 40 multiplications
    
    # For 4x5 * 5x3:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..2
    # This is 4*3 = 12 bilinear forms, each sum of 5 products
    # Schoolbook: 4*5*3 = 60 multiplications
    
    # Total: 40 + 60 = 100 (schoolbook again)
    
    # To beat schoolbook, we need to find a genuinely lower-rank decomposition
    # Let's try ALS with more restarts and different initializations
    
    T = build_tensor(n, m, p)
    
    # Try a range of ranks with more aggressive search
    candidate_ranks = [25, 28, 30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    
    for R in candidate_ranks:
        if time.time() > deadline - 30:
            break
        if R >= best_rank:
            break
            
        U_try, V_try, W_try, err = try_rank_als(R, T, max_iter=300, n_restarts=5)
        
        if U_try is not None:
            best_rank = R
            best_result = {"U": U_try, "V": V_try, "W": W_try}
            break
    
    # If no better decomposition found via ALS, return schoolbook
    if best_rank == rank_school:
        return best_result
    
    # Convert Fractions to simple numbers
    def frac_to_num(f):
        if f.denominator == 1:
            return f.numerator
        elif f.denominator == 2:
            return f.numerator / 2.0
        else:
            return float(f)
    
    U_final = [[frac_to_num(f) for f in row] for row in best_result["U"]]
    V_final = [[frac_to_num(f) for f in row] for row in best_result["V"]]
    W_final = [[frac_to_num(f) for f in row] for row in best_result["W"]]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
