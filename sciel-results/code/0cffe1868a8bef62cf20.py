import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 115  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact using vectorized check."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        errors.append((i,j,k,s))
        return len(errors) == 0
    
    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition(n, m, p)
    
    # === WINOGRAD-LIKE APPROACH ===
    # Key idea: Decompose M_{3,4,8} using block structure.
    # View A as 3x4, B as 4x8.
    # Split B into two 4x4 blocks: B = [B1 | B2]
    # Then C = A*B1 + A*B2 (block addition)
    # But we can use Winograd-style: for each row i of A, compute A[i,:]*B[:,k]
    # 
    # Alternative: Use the fact that M_{3,4,8} can be seen as:
    # For each (i,k) output, C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Winograd insight for small blocks:
    # Consider splitting the j-dimension (size 4) into blocks.
    # For a 1x4 * 4x1 multiplication, schoolbook needs 4 mults.
    # But if we have structure, we can do better.
    #
    # Let's try a different decomposition:
    # M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank = 4 schoolbook
    # But M_{1,2,2} might have rank 3 (it's like computing [a,b]*[[c,d],[e,f]] = [ac+be, ad+bf])
    # Actually M_{1,2,2} has rank 4 (no savings for this small case)
    #
    # Better: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # We need good rank for M_{3,4,4}.
    #
    # For M_{3,4,4}: 3x4 * 4x4 -> 3x4
    # Schoolbook: 48
    # 
    # Try: M_{3,4,4} = M_{3,2,4} ⊗ M_{1,2,2}? No, that gives M_{3,4,8}.
    #
    # Let's try direct Winograd on M_{3,4,8}:
    # Split the 4-dimension (j) into two 2-dimensions.
    # A[i, j] where j = 2*j1 + j2, j1 in {0,1}, j2 in {0,1}
    # B[j, k] where j = 2*j1 + j2
    # C[i,k] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, k]
    #        = sum_{j1} [A[i,2*j1] * B[2*j1,k] + A[i,2*j1+1] * B[2*j1+1,k]]
    #
    # This is just schoolbook rearranged. No savings.
    #
    # Winograd savings come from when you have convolutions or when
    # the same linear combinations appear multiple times.
    #
    # For matrix multiplication, the standard Winograd approach is:
    # If you have blocks of size BxB, you can use FMM (Fast Matrix Multiplication)
    # but that requires B >= 2 and the savings are asymptotic.
    #
    # For our small case, let's try a different approach:
    # Use the fact that we can share computations across the p=8 dimension.
    #
    # Actually, let's try: M_{3,4,8} = M_{3,4,2} ⊗ M_{1,1,4}
    # M_{1,1,4}: rank 4
    # M_{3,4,2}: 3x4 * 4x2 -> 3x2, schoolbook rank = 24
    # Total: 24*4 = 96 (no savings)
    #
    # The fundamental issue: for small matrices, tensor product decompositions
    # don't give savings unless the factors have sub-schoolbook rank.
    #
    # Let's try a randomized approach with more structure:
    # Fix U and V with small integer entries, solve for W, then round.
    
    def try_rank_decomposition(n, m, p, target_rank, max_attempts=50, seed_base=42):
        """Try to find a rank-target_rank decomposition."""
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 10:
                return None
            
            seed = seed_base + attempt * 1000 + target_rank
            np.random.seed(seed)
            
            # Generate U and V with small integer entries
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
            
            # Solve for W: W[k*n+i, c] = M[i,k,c] / ||M[i,k,:]||^2
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
        
        return None
    
    # Try to find better rank via randomized search
    # Start from schoolbook and try to reduce
    search_ranks = list(range(best_rank - 1, max(10, best_rank - 30), -1))
    
    for target_rank in search_ranks:
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_rank_decomposition(n, m, p, target_rank, max_attempts=40)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Continue searching for even lower rank
            search_ranks = list(range(target_rank - 1, max(10, target_rank - 20), -1))
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
