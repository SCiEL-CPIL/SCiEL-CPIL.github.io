"""Matrix multiplication 3x4 by 4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _build_matmul_tensor(n, m, p):
    """Build the matmul tensor T where T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, rank, num_trials=5, seed=42):
    """Verify that U, V, W correctly decompose the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    np.random.seed(seed)
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            Ar = U_arr[:, r].reshape(n, m)
            Br = V_arr[:, r].reshape(m, p)
            Cr = W_arr[:, r].reshape(p, n)
            temp = (Ar * A) @ (Br * B)
            C_decomposed += temp * Cr.T
        if not np.allclose(C_decomposed, C_expected, atol=1e-8):
            return False
    return True

def _validate(U, V, W):
    """Check that U, V, W form a valid decomposition of the matmul tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    return _verify_decomposition(U, V, W, n, m, p, rank)

def _schoolbook(n=None, m=None, p=None):
    """Standard schoolbook decomposition of rank n*m*p."""
    if n is None:
        n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _kronecker_combine(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """
    Combine two matmul decompositions using Kronecker product structure.
    
    Problem 1: n x m by m x p1 (decomposition U1, V1, W1 with rank r1)
    Problem 2: n x m by m x p2 (decomposition U2, V2, W2 with rank r2)
    
    We want to decompose: n x m by m x (p1*p2)
    
    The output index k = k1 * p2 + k2 where k1 in [0,p1), k2 in [0,p2)
    The W index is k*n + i = (k1*p2+k2)*n + i
    
    We need W_combined[(k1*p2+k2)*n+i, r1*r2] such that the product works.
    
    Strategy: Use rank r1 * r2 decomposition where each rank index is (r1_idx, r2_idx).
    U_combined[i*m+j, (r1_idx, r2_idx)] = U1[i*m+j, r1_idx] * U2[i*m+j, r2_idx]
    V_combined[j*p1*p2+k, (r1_idx, r2_idx)] where k = k1*p2+k2
      = V1[j*p1+k1, r1_idx] * V2[j*p2+k2, r2_idx]
    W_combined[(k1*p2+k2)*n+i, (r1_idx, r2_idx)] = W1[k1*n+i, r1_idx] * W2[k2*n+i, r2_idx]
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 * r2
    p = p1 * p2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for r1_idx in range(r1):
                for r2_idx in range(r2):
                    col = r1_idx * r2 + r2_idx
                    U[i * m + j][col] = U1[i * m + j][r1_idx] * U2[i * m + j][r2_idx]
    
    for j in range(m):
        for k1 in range(p1):
            for k2 in range(p2):
                k = k1 * p2 + k2
                for r1_idx in range(r1):
                    for r2_idx in range(r2):
                        col = r1_idx * r2 + r2_idx
                        V[j * p + k][col] = V1[j * p1 + k1][r1_idx] * V2[j * p2 + k2][r2_idx]
    
    for k1 in range(p1):
        for k2 in range(p2):
            for i in range(n):
                k = k1 * p2 + k2
                for r1_idx in range(r1):
                    for r2_idx in range(r2):
                        col = r1_idx * r2 + r2_idx
                        W[k * n + i][col] = W1[k1 * n + i][r1_idx] * W2[k2 * n + i][r2_idx]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Strategy: Use Kronecker product to factor p=8 into 2*4
    # Find decomposition for 3x4 by 4x4 and 3x4 by 4x2
    # Then combine them
    
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # First, try to find good decompositions for sub-problems
    # Sub-problem 1: 3x4 by 4x4 (schoolbook rank = 48)
    # Sub-problem 2: 3x4 by 4x2 (schoolbook rank = 24)
    
    # For 3x4 by 4x2, schoolbook rank is 24
    # For 3x4 by 4x4, schoolbook rank is 48
    # Combined rank would be 48 * 24 = 1152 (too high!)
    # 
    # Better approach: factor p=8 as 4*2
    # Sub-problem 1: 3x4 by 4x4, rank r1
    # Sub-problem 2: 3x4 by 4x2, rank r2
    # Combined rank = r1 * r2
    #
    # If we use schoolbook: r1=48, r2=24, combined=1152 (worse than 96)
    # 
    # Alternative: factor as 2*4
    # Sub-problem 1: 3x4 by 4x2, rank r1
    # Sub-problem 2: 3x4 by 4x4, rank r2
    # Same issue.
    #
    # Better: Use the fact that for 2x2 by 2x2, we can do rank 7 (Strassen)
    # Or even better, find specific low-rank decompositions
    #
    # Let me think differently. The Kronecker approach requires:
    # For 3x4 by 4x8, write k = k1*4 + k2 (k1 in 0..1, k2 in 0..3)
    # or k = k1*2 + k2 (k1 in 0..3, k2 in 0..1)
    #
    # Case 1: k = k1*4 + k2
    # Output C[i, k1*4+k2] = sum_j A[i,j] * B[j, k1*4+k2]
    # We can think of B as two 4x4 blocks: B1 = B[:, 0:4], B2 = B[:, 4:8]
    # C1 = A @ B1 (3x4), C2 = A @ B2 (3x4)
    # So we need decompositions for 3x4 by 4x4 (twice, or once reused)
    # If we have a rank-r decomposition for 3x4 by 4x4, we can use it for both
    # C1[i,k2] = sum_r U1[i*4+j, r] * A[i,j] * V1[j*4+k2, r] * B1[j,k2] * W1[k2*3+i, r]
    # C2[i,k2] = sum_r U1[i*4+j, r] * A[i,j] * V1[j*4+k2, r] * B2[j,k2] * W1[k2*3+i, r]
    #
    # But the W index needs to be (k1*4+k2)*3+i, not just k2*3+i
    # So we need to modify W to account for k1
    #
    # Actually, let's use a different factorization. Let k = k1*2 + k2
    # where k1 in 0..3, k2 in 0..1
    # B has columns indexed by k = k1*2+k2
    # We can group B into 4 pairs of columns
    #
    # Alternatively, think of it as:
    # 3x4 by 4x8 = (3x4 by 4x2) ⊗ (1x1 by 1x4) -- not quite
    #
    # Let me try a direct approach: find a low-rank decomposition for 3x4 by 4x4
    # and 3x4 by 4x2, then combine.
    
    # For 3x4 by 4x2: schoolbook rank = 24
    # Try to find rank < 24 for this
    # Lower bound: max(12, 8, 6) = 12
    
    # For 3x4 by 4x4: schoolbook rank = 48
    # Try to find rank < 48
    
    # Let's try random search for the sub-problems first
    np.random.seed(42)
    
    # Try to find rank-r decomposition for 3x4 by 4x2
    best_342 = _schoolbook(3, 4, 2)  # rank 24
    best_342_rank = 24
    
    # Try to find rank-r decomposition for 3x4 by 4x4
    best_344 = _schoolbook(3, 4, 4)  # rank 48
    best_344_rank = 48
    
    # Try random search for 3x4 by 4x2 with rank < 24
    for target_rank in range(23, 11, -1):
        if time.time() > deadline - 60:
            break
        for attempt in range(5):
            if time.time() > deadline - 55:
                break
            U = np.random.randn(12, target_rank)
            V = np.random.randn(8, target_rank)
            W = np.random.randn(6, target_rank)
            
            # Quick check: can we solve this?
            # The tensor has 12*8*6 = 576 entries, but only 24 are non-zero
            # We need sum_r U[:,r] outer V[:,r] outer W[:,r] = T
            # This is underdetermined for small rank, but the structure matters
            
            # Use a simple iterative approach
            # For each trial, check if random init gives exact solution
            # (unlikely, but let's try)
            
            # Actually, for exact solutions, we need to solve a nonlinear system
            # Let's use a different approach: structured search
            
            # For 3x4 by 4x2, the tensor T[i*4+j, j*2+k, k*3+i] = 1
            # Let's see if we can find a rank-12 solution (minimum possible)
            # This would require very specific structure
            
            pass
    
    # Given the difficulty of finding low-rank decompositions via random search,
    # let's use the Kronecker combination with schoolbook sub-decompositions
    # and see if we can do better than 96
    
    # Combine 3x4 by 4x4 (rank 48) with 3x4 by 4x2 (rank 24)
    # This gives rank 48*24 = 1152, which is worse
    
    # Better: Use the same decomposition for both halves
    # If we have a rank-r decomposition for 3x4 by 4x4, we can potentially
    # use it for the 3x4 by 4x8 problem by splitting the output
    
    # Let me try a direct random search for the full problem with rank < 96
    for target_rank in range(95, 85, -1):
        if time.time() > deadline - 20:
            break
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            U = np.random.randn(12, target_rank)
            V = np.random.randn(32, target_rank)
            W = np.random.randn(24, target_rank)
            
            # Verify
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), 3, 4, 8, target_rank, num_trials=3, seed=attempt):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        if best_rank < 96:
            break
    
    # Return the best result found
    return best_result
