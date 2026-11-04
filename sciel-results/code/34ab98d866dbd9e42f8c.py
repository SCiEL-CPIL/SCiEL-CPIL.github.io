"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    r_school = n * m * p  # 100
    
    # Helper to build schoolbook decomposition
    def build_schoolbook(rank):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W
    
    # Helper to verify a decomposition
    def verify(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T_decomp = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t, optimize=True)
        T_target = np.ones((n, m, p))
        return np.allclose(T_decomp, T_target, atol=1e-9)
    
    # Establish baseline
    U_best, V_best, W_best = build_schoolbook(r_school)
    best_rank = r_school
    
    # Try block decomposition strategies to reduce rank
    # Strategy: decompose 4x5x5 into blocks and exploit structure
    
    # Approach 1: Row-block decomposition
    # Split A (4x5) into two 2x5 blocks: A1 (rows 0-1), A2 (rows 2-3)
    # C1 = A1 * B (2x5), C2 = A2 * B (2x5)
    # Each 2x5x5 multiplication has rank 2*5*5 = 50
    # Total: 100 (no improvement, but let's think differently)
    
    # Approach 2: Use the fact that 4x5x5 can be seen as 4*(5x5) 
    # But that's still 4*25 = 100
    
    # Approach 3: Try to find a lower rank via random search with verification
    # The key insight: for small matrices, exact decompositions may exist below n*m*p
    # Known bounds: R(4,5,5) is between some lower bound and 100
    
    # Let's try a structured approach using Kronecker products
    # 4x5x5: if we could write 4 = 2*2, we might use R(2,5,5)*something
    # But 2x5x5 has rank 50, so 2*50 = 100 again
    
    # Approach 4: Random search for rank < 100
    # Try ranks from 99 down, with limited time per rank
    np.random.seed(42)
    
    # Try a few random attempts at rank 99
    if time.time() - start_time < time_limit - 30.0:
        R_try = r_school - 1
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Initialize with schoolbook-like structure but with one rank removed
            # Try to merge two similar terms
            U_try = np.zeros((n * m, R_try))
            V_try = np.zeros((m * p, R_try))
            W_try = np.zeros((p * n, R_try))
            
            # Fill with schoolbook for first R_try terms
            for c in range(R_try):
                i = c // (m * p)
                j = (c // p) % m
                k = c % p
                U_try[i * m + j, c] = 1.0
                V_try[j * p + k, c] = 1.0
                W_try[k * n + i, c] = 1.0
            
            # Randomly perturb to try to compensate for the missing term
            if not verify(U_try, V_try, W_try, R_try):
                # Add random perturbation
                U_try += 0.1 * np.random.randn(n * m, R_try)
                V_try += 0.1 * np.random.randn(m * p, R_try)
                W_try += 0.1 * np.random.randn(p * n, R_try)
            
            if verify(U_try, V_try, W_try, R_try):
                best_rank = R_try
                U_best, V_best, W_best = U_try, V_try, W_try
                break
    
    # Approach 5: Use known results - R(4,4,4) = 49 (Strassen-like)
    # For 4x5x5, we might be able to embed a 4x4x4 solution
    # 4x5x5 = 4x4x5 + 4x1x5
    # R(4,4,5) <= 4*4*5 = 80 (schoolbook) or better
    # R(4,1,5) = 4*1*5 = 20 (can't do better for 1 in middle)
    # Total: 100 again unless we find better R(4,4,5)
    
    # Let's try to construct R(4,4,5) more efficiently
    # 4x4x5: decompose B (4x5) into columns
    # C[:,k] = A * B[:,k] for k=0..4
    # Each is 4x4x1, rank 16
    # Total: 5*16 = 80
    
    # Can we do better for 4x4x5?
    # R(4,4,4) is known to be 49 (best known)
    # R(4,4,5) <= R(4,4,4) + R(4,4,1) = 49 + 16 = 65
    # Or R(4,4,5) <= 5*R(4,4,1) = 80
    # So potentially 65 for 4x4x5, plus 20 for 4x1x5 = 85 total
    
    # Let's try to build a 4x4x5 decomposition of rank 65
    if time.time() - start_time < time_limit - 15.0:
        # First, try to find a 4x4x4 decomposition of rank 49
        # This is non-trivial; use known Strassen-based construction
        # For now, use schoolbook 4x4x4 (rank 64) as fallback
        
        # Build 4x4x4 schoolbook (rank 64)
        n4, m4, p4 = 4, 4, 4
        U_444 = np.zeros((n4 * m4, n4 * m4 * p4))
        V_444 = np.zeros((m4 * p4, n4 * m4 * p4))
        W_444 = np.zeros((p4 * n4, n4 * m4 * p4))
        for c, (i, j, k) in enumerate(product(range(n4), range(m4), range(p4))):
            U_444[i * m4 + j, c] = 1.0
            V_444[j * p4 + k, c] = 1.0
            W_444[k * n4 + i, c] = 1.0
        
        # Build 4x4x1 (rank 16)
        n41, m41, p41 = 4, 4, 1
        U_441 = np.zeros((n41 * m41, n41 * m41 * p41))
        V_441 = np.zeros((m41 * p41, n41 * m41 * p41))
        W_441 = np.zeros((p41 * n41, n41 * m41 * p41))
        for c, (i, j, k) in enumerate(product(range(n41), range(m41), range(p41))):
            U_441[i * m41 + j, c] = 1.0
            V_441[j * p41 + k, c] = 1.0
            W_441[k * n41 + i, c] = 1.0
        
        # Combine for 4x4x5 (rank 64+16=80)
        rank_445 = 80
        U_445 = np.zeros((n4 * m4, rank_445))
        V_445 = np.zeros((m4 * p4 + 5, rank_445))  # m4*p4 for 4x4 part, +5 for the 5th column
        # Wait, this isn't quite right. Let me reconsider.
        
        # Actually, 4x4x5 means A is 4x4, B is 4x5, C is 4x5
        # B has 5 columns. C[:,k] = A * B[:,k]
        # For k=0..3: C[:,k] = A * B4[:,k] where B4 is 4x4 (first 4 cols)
        # For k=4: C[:,4] = A * B4[:,4] where B4[:,4] is 4x1
        # So R(4,4,5) <= R(4,4,4) + R(4,4,1) = 64 + 16 = 80 (schoolbook)
        
        # For 4x1x5: A is 4x1, B is 1x5, C is 4x5
        # C[i,k] = A[i,0] * B[0,k]
        # This is rank 4*5 = 20 (each term is outer product)
        n415, m415, p415 = 4, 1, 5
        U_415 = np.zeros((n415 * m415, n415 * m415 * p415))
        V_415 = np.zeros((m415 * p415, n415 * m415 * p415))
        W_415 = np.zeros((p415 * n415, n415 * m415 * p415))
        for c, (i, j, k) in enumerate(product(range(n415), range(m415), range(p415))):
            U_415[i * m415 + j, c] = 1.0
            V_415[j * p415 + k, c] = 1.0
            W_415[k * n415 + i, c] = 1.0
        
        # Total rank: 80 + 20 = 100 (same as schoolbook)
        # No improvement from this approach.
    
    # Return the best found
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
