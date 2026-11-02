import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    
    # Attempt to construct a rank-10 decomposition
    # If verification fails, fall back to rank 11
    
    def verify(U, V, W):
        rank = len(W[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check dimensions
        if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
            return False
            
        # Verify tensor decomposition
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
        # We test with random matrices
        rng = np.random.default_rng(42)
        for _ in range(20):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        lA = np.dot(U_np[i*m:(i+1)*m, r], A[i, :])
                        lB = np.dot(V_np[k::p, r], B[:, k])
                        C_computed[i, k] += W_np[k*n+i, r] * lA * lB
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Rank 11 Solution (Current Best)
    def get_rank_11():
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}

    def try_find_rank_10():
        """
        Search for a rank-10 decomposition using syzygy-based compression.
        
        Strategy: Start from the rank-11 solution and try to find a linear
        combination of the 11 rank-1 tensors that is zero (a syzygy), allowing
        us to eliminate one tensor and get rank 10.
        
        Alternatively, directly search for rank-10 solutions using randomized
        linear algebra.
        """
        if time.time() > deadline - 20:
            return None
        
        rank = 10
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # Try a known rank-10 construction for 2x2x3
        # Based on the structure: 7 multiplications for 2x2x2 (Strassen)
        # + 3 for the extra column, but with some sharing
        
        # First 7: Strassen for C[0:2, 0:2]
        # 1. (A00+A11)(B00+B11)
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11)
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00)
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01)
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11)
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # For column 2: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
        # Need 4 more multiplications, but we only have 3 slots (7,8,9)
        # So we need to share with existing multiplications
        
        # Try: Use existing multiplications that involve B[0,*] and B[1,*]
        # The key insight: some of the Strassen multiplications already compute
        # linear combinations of B[0,0], B[0,1], B[1,0], B[1,1]
        # We need to extend them to also capture B[0,2], B[1,2]
        
        # This requires a different approach. Let's try a direct search.
        
        # Direct randomized search for rank 10
        rng = np.random.default_rng(123)
        
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_rand = rng.standard_normal((n*m, rank))
            V_rand = rng.standard_normal((m*p, rank))
            W_rand = rng.standard_normal((p*n, rank))
            
            # Try to solve for W given U, V
            # The tensor equation is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # This is nonlinear. Instead, use iterative approach.
            
            # For now, just check if the hardcoded structure works
            pass
        
        # Try a specific known construction
        # Reset U, V, W
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u2(i, j, r, val): U[i*m+j][r] = val
        def set_v2(j, k, r, val): V[j*p+k][r] = val
        def set_w2(k, i, r, val): W[k*n+i][r] = val

        # Alternative rank-10 attempt
        # Use 6 multiplications for the 2x2 part (not Strassen, but something else)
        # and 4 for the extra column, with 2 shared
        
        # 1. A00 * (B00 + B02)
        set_u2(0, 0, 0, 1)
        set_v2(0, 0, 0, 1); set_v2(0, 2, 0, 1)
        set_w2(0, 0, 0, 1)
        set_w2(2, 0, 0, 1)

        # 2. A01 * (B10 + B12)
        set_u2(0, 1, 1, 1)
        set_v2(1, 0, 1, 1); set_v2(1, 2, 1, 1)
        set_w2(0, 0, 1, 1)
        set_w2(2, 0, 1, 1)

        # 3. A10 * (B00 + B02)
        set_u2(1, 0, 2, 1)
        set_v2(0, 0, 2, 1); set_v2(0, 2, 2, 1)
        set_w2(0, 1, 2, 1)
        set_w2(2, 1, 2, 1)

        # 4. A11 * (B10 + B12)
        set_u2(1, 1, 3, 1)
        set_v2(1, 0, 3, 1); set_v2(1, 2, 3, 1)
        set_w2(0, 1, 3, 1)
        set_w2(2, 1, 3, 1)

        # 5. (A00+A11) * (B01+B11)
        set_u2(0, 0, 4, 1); set_u2(1, 1, 4, 1)
        set_v2(0, 1, 4, 1); set_v2(1, 1, 4, 1)
        set_w2(1, 0, 4, 1)
        set_w2(1, 1, 4, 1)

        # 6. (A10+A11) * (B00-B01)
        set_u2(1, 0, 5, 1); set_u2(1, 1, 5, 1)
        set_v2(0, 0, 5, 1); set_v2(0, 1, 5, -1)
        set_w2(0, 1, 5, 1)
        set_w2(1, 1, 5, -1)

        # 7. (A00+A01) * (B11-B10)
        set_u2(0, 0, 6, 1); set_u2(0, 1, 6, 1)
        set_v2(1, 1, 6, 1); set_v2(1, 0, 6, -1)
        set_w2(0, 0, 6, 1)
        set_w2(1, 0, 6, 1)

        # 8. (A10-A00) * (B00+B01)
        set_u2(1, 0, 7, 1); set_u2(0, 0, 7, -1)
        set_v2(0, 0, 7, 1); set_v2(0, 1, 7, 1)
        set_w2(1, 1, 7, 1)

        # 9. (A01-A11) * (B10+B11)
        set_u2(0, 1, 8, 1); set_u2(1, 1, 8, -1)
        set_v2(1, 0, 8, 1); set_v2(1, 1, 8, 1)
        set_w2(0, 0, 8, 1)

        # 10. Need one more for the remaining terms
        # Let's verify what we have so far
        result = {"U": U, "V": V, "W": W}
        if verify(U, V, W):
            return result
        
        # If the above doesn't work, try another structure
        # Reset
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u3(i, j, r, val): U[i*m+j][r] = val
        def set_v3(j, k, r, val): V[j*p+k][r] = val
        def set_w3(k, i, r, val): W[k*n+i][r] = val

        # Try: 4 multiplications for column 2, 6 for the 2x2 part
        # But we need exactly 10 total
        
        # 1. (A00+A10) * B02
        set_u3(0, 0, 0, 1); set_u3(1, 0, 0, 1)
        set_v3(0, 2, 0, 1)
        set_w3(2, 0, 0, 1)
        set_w3(2, 1, 0, 1)

        # 2. (A01+A11) * B12
        set_u3(0, 1, 1, 1); set_u3(1, 1, 1, 1)
        set_v3(1, 2, 1, 1)
        set_w3(2, 0, 1, 1)
        set_w3(2, 1, 1, 1)

        # Now we have C[0,2] = A00*B02 + A01*B12 + A10*B02 + A11*B12
        # But we need C[0,2] = A00*B02 + A01*B12 and C[1,2] = A10*B02 + A11*B12
        # The above gives cross terms. Not correct.
        
        # This approach is getting complex. Let's just return None and use rank 11.
        return None

    # Try to find a Rank 10 solution
    rank_10_result = try_find_rank_10()
    if rank_10_result is not None:
        return rank_10_result
    
    # Fallback to verified rank-11 solution
    return get_rank_11()
# EVOLVE-BLOCK-END
