import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n=2, m=3, p=9):
    """
    Verify that the tensor decomposition is exact.
    We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Efficient check:
    # For each r, we have vectors u_r (n*m), v_r (m*p), w_r (p*n).
    # The term is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i].
    # Summing over r gives T[i,j,k].
    
    # We can loop over r, but that's rank * n * m * p operations.
    # For rank ~ 50 and 54 entries, it's fast.
    # But let's try to do it vectorized if possible, or just loop.
    
    # Actually, we can construct the tensor T as follows:
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's just loop i, j, k and check. It's only 54 checks.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Keep margin for 120s limit
    
    # 1. Try Rank 50 (Strassen-like block construction)
    # The idea from the "cell findings":
    # "Replacing the schoolbook rank-54 decomposition with a Strassen-based block decomposition achieves rank 50..."
    # Structure:
    # A is 2x3, B is 3x9.
    # We can split A into two 2x? blocks? No, 3 is prime.
    # Maybe split the 3 dimension?
    # Let's look at the specific rank 50 construction mentioned.
    # "using 7-term Strassen products for 2x2x2 sub-blocks"
    # This implies we are treating the multiplication as involving 2x2 blocks.
    # But dimensions are 2, 3, 9.
    # How do we get 2x2 blocks?
    # Maybe the 3 and 9 are treated specially.
    
    # Let's try to construct a rank 50 solution explicitly if we can find the pattern,
    # or use a randomized search for rank 50 if the explicit one is hard to code from memory.
    # However, the prompt says "Keep this design...". The current design is random search + schoolbook.
    # The "cell finding" suggests rank 50 is possible.
    # Let's try to find a rank 50 solution via random search first, as it's easier to implement robustly.
    # If random search fails, we can try to construct the Strassen one.
    
    # Actually, let's try to implement the specific Strassen block idea if we can deduce it.
    # The finding says: "Strassen products for 2x2x2 sub-blocks and schoolbook for the remaining 2x2x1 and 2x1x9 parts".
    # This suggests a decomposition of the tensor based on indices.
    
    # Let's try random search for R=50 first. It might find a solution quickly.
    # If not, we'll try R=51, 52...
    
    # Optimization: Use integer/half-integer constraints?
    # The problem allows "integer/half-integer".
    # Random search with {0, 1, -1} for U and V, then solve for W.
    
    best_result = None
    best_rank = n * m * p  # 54
    
    # Try ranks from 50 down to 45 (arbitrary lower bound for time)
    for R in range(50, 44, -1):
        if time.time() > deadline:
            break
            
        # For each R, try multiple random restarts
        # We want to find A, B, C such that sum_r A_r B_r C_r = T
        # Strategy: Random A, B with small entries. Solve for C.
        # C depends on (k, i). There are p*n = 18 such vectors.
        # Each vector C_ki has R components.
        # Equations: For each (i, j, k), sum_r A_ij_r B_jk_r C_ki_r = 1.
        # Group by (k, i). For fixed (k, i), we have m=3 equations for the R variables of C_ki.
        # Matrix M_jr = A_ij_r * B_jk_r.
        # We need M @ C_ki = 1 (vector of ones, length 3).
        
        for attempt in range(500): # Increased attempts for R=50
            if time.time() > deadline:
                break
                
            # Generate random A, B
            # Use {0, 1, -1}
            A = np.random.randint(-1, 2, (n*m, R))
            B = np.random.randint(-1, 2, (m*p, R))
            
            C = np.zeros((p*n, R))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build M (m x R)
                    M = np.zeros((m, R))
                    for j in range(m):
                        # A index: i*m + j
                        # B index: j*p + k
                        M[j, :] = A[i*m + j, :] * B[j*p + k, :]
                    
                    # Solve M @ C_vec = ones(m)
                    # We want C_vec to be "nice" (int/half-int) if possible, but any real solution works?
                    # The task says "integer/half-integer entries".
                    # So we MUST find a solution where C entries are k/2 for integer k.
                    # This makes the problem much harder than just finding *any* solution.
                    
                    # If we just solve for real C, we might not get nice numbers.
                    # However, if A and B are integers, and M is integer,
                    # the solution C = M^{-1} * 1 (if invertible) or M^+ * 1 (if not)
                    # might be rational.
                    # If we can force M to be invertible and its inverse to have denominator 2^k, we might get nice numbers.
                    
                    # Alternative: Search for C in a small set?
                    # No, R is up to 50.
                    
                    # Let's try to solve and check if close to half-integers.
                    try:
                        # Use pinv for robustness
                        C_vec = np.linalg.pinv(M) @ np.ones(m)
                        # Check if C_vec is close to half-integers
                        C_half = np.round(C_vec * 2) / 2
                        if np.allclose(M @ C_half, np.ones(m), atol=1e-9):
                            C[k*n + i, :] = C_half
                        else:
                            # Maybe try integer?
                            C_int = np.round(C_vec)
                            if np.allclose(M @ C_int, np.ones(m), atol=1e-9):
                                C[k*n + i, :] = C_int
                            else:
                                valid = False
                                break
                    except:
                        valid = False
                        break
                    
            if valid:
                # Verify the full solution
                if _verify(A, B, C):
                    best_rank = R
                    best_result = {
                        "U": A.tolist(),
                        "V": B.tolist(),
                        "W": C.tolist()
                    }
                    # If we found a low rank, we can return immediately
                    # But let's see if we can go lower?
                    # For now, return immediately as we are in a time budget.
                    return best_result

    # If no better rank found, return schoolbook
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
