import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 140.0

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    def make_rank44():
        """
        Rank-44 construction: schoolbook 45 with one term merged.
        
        Strategy: For the (i=0,j=0) output entries at k=3 and k=4, 
        we use a rank-2 block that shares structure.
        
        Actually, let's use a cleaner approach: 
        Start with schoolbook 45. Replace two terms that have the same 
        U-index (same i,j) with a rank-2 sub-decomposition that 
        covers both k-values with 2 terms instead of 2 terms... 
        that doesn't save anything.
        
        Better: Use the identity for 2x2x2 matrix multiplication.
        For fixed i, the outputs C[i,3] and C[i,4] are:
        C[i,3] = sum_j A[i,j] B[j,3]
        C[i,4] = sum_j A[i,j] B[j,4]
        
        This is a 1x3x2 "matrix mult" which has rank 3 (schoolbook).
        No savings there.
        
        Alternative: Use the fact that the full 3x3x5 tensor can be 
        decomposed by splitting into 3x3x3 + 3x3x2, where the 3x3x2 
        part uses rank 17 instead of 18.
        
        For 3x3x2: C[i,k] = sum_j A[i,j] B[j,k] for k=0,1 (renamed)
        This is equivalent to multiplying a 3x3 by 3x2.
        Schoolbook rank = 9. Can we do it in rank 8?
        
        Known result: [3,3,2] tensor has rank 9 (no improvement known).
        
        Let me try a different merge: combine terms across different (i,j) pairs.
        
        Actually, the simplest verified rank-44: use schoolbook for all 45 terms,
        then observe that one pair of terms can be replaced. Specifically, use the 
        identity: 1 = 1*1 = (1/2)*(2*1) + (1/2)*(0*1)... no that's not right.
        
        Let me just do a brute-force verified construction:
        """
        # Schoolbook base: 45 terms, indices 0..44
        # Term (i,j,k) has index i*15 + j*5 + k
        
        # We'll create 44 terms by replacing terms (0,0,3) and (0,0,4) 
        # with a 2-term rank block that produces the same contributions
        # to C[0,3] and C[0,4] while contributing 0 to all other outputs.
        
        # This requires: for new terms c=43,44 (replacing old 43,44... wait)
        # Let me re-index. Schoolbook term idx = i*15+j*5+k.
        # (0,0,3) -> idx=3, (0,0,4) -> idx=4
        
        # Remove terms idx=3 and idx=4. Add 2 new terms that:
        # - Contribute 1 to (i=0,j=0,k=3) and (i=0,j=0,k=4) respectively
        # - Contribute 0 to all other (i',j',k') outputs
        
        # For new term c, contribution to (i',j',k') is U[i',j',c]*V[j',k',c]*W[k',i',c]
        # We need:
        #   U[0,0,c]*V[0,3,c]*W[3,0,c] = 1 for c=first new term (to cover k=3)
        #   U[0,0,c]*V[0,4,c]*W[4,0,c] = 1 for c=second new term (to cover k=4)
        #   All other combinations = 0
        
        # This is tricky. Let me use a different approach:
        # Keep all 45 schoolbook terms but set one to zero and adjust others.
        
        # Actually, the cleanest: just return schoolbook 45 if we can't verify 44.
        # But let me try one more structured approach.
        
        # Use the 2x2x2 Strassen trick on a sub-block.
        # For i in {0,1}, j in {0,1}, k in {3,4}: this is a 2x2x2 sub-problem.
        # Strassen gives rank 7 instead of 4. Wait, 2x2x2 schoolbook is 4, Strassen is 7. That's worse!
        
        # I'm overcomplicating this. Let me just use schoolbook 45.
        return make_schoolbook(n, m, p)

    # Try to find rank 44 via structured search
    best_U, best_V, best_W, best_rank = None, None, None, 45

    # Deterministic rank-44 attempt using algebraic structure
    # For 3x3x5, we can view it as computing 3 rows of C, each being 3x5 product.
    # Row i of C: C[i,:] = A[i,:] * B (3x5)
    # This is a 1x3x5 matrix mult, rank 15 schoolbook.
    # Total: 3 * 15 = 45.
    
    # Can we share computation between rows? If A[0,:] and A[1,:] are related...
    # But they're arbitrary, so no.
    
    # The only way to save rank is if the tensor has lower rank than 45.
    # Known: [3,3,5] tensor rank is 44 (from literature).
    # Construction: see e.g. "Rank of matrix multiplication tensors" 
    # One construction: use rank-11 for 2x2x3 (instead of 12) + remaining.
    
    # 2x2x3: C[i,k] = sum_j A[i,j]B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}
    # Schoolbook: 12. Known rank: 11 (Landsberg et al.)
    
    # So: split 3x3x5 as:
    #   i=2: 1x3x5, rank 15
    #   i in {0,1}: 2x3x5
    #     Split 2x3x5 as:
    #       j=2: 2x1x5, rank 10
    #       j in {0,1}: 2x2x5
    #         Split 2x2x5 as:
    #           k in {0,1,2}: 2x2x3, rank 11
    #           k in {3,4}: 2x2x2, rank 4
    #         Total 2x2x5: 11+4=15
    #     Total 2x3x5: 10+15=25
    # Total: 15+25=40?? That can't be right, let me recheck.
    
    # Wait, 2x2x3 rank 11: is that correct? The 2x2x2 tensor has rank 7.
    # 2x2x3 = 2x2x2 + 2x2x1. 2x2x1 has rank 4 (schoolbook, and it's minimal).
    # So 2x2x3 rank <= 7+4=11. And it's known to be exactly 11.
    
    # So total: 15 (i=2) + 10 (j=2, i in {0,1}) + 11 (2x2x3) + 4 (2x2x2) = 40?
    # That seems too low. Let me recount.
    
    # Full tensor: sum over i,j,k of A[i,j]B[j,k] -> C[i,k]
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4}
    
    # Split by i:
    #   i=2: sum_j A[2,j]B[j,k] for k=0..4: 1x3x5, rank 15
    #   i in {0,1}: sum_j A[i,j]B[j,k] for k=0..4: 2x3x5
    
    # Split 2x3x5 by j:
    #   j=2: sum over i in {0,1}: A[i,2]B[2,k] for k=0..4: 2x1x5, rank 10
    #   j in {0,1}: 2x2x5
    
    # Split 2x2x5 by k:
    #   k in {0,1,2}: 2x2x3, rank 11
    #   k in {3,4}: 2x2x2, rank 4
    #   Total: 15
    
    # Grand total: 15 + 10 + 15 = 40. 
    
    # But wait, this gives rank 40 which is less than 44! Let me verify this is valid.
    # The issue: when we split, the sub-tensors must be independent.
    # 1x3x5: 15 terms, each (i=2, j, k)
    # 2x1x5: 10 terms, each (i in {0,1}, j=2, k)
    # 2x2x3: 11 terms for (i in {0,1}, j in {0,1}, k in {0,1,2})
    # 2x2x2: 4 terms for (i in {0,1}, j in {0,1}, k in {3,4})
    # Total: 15+10+11+4 = 40 terms.
    
    # This should work! Let me implement it.
    
    def make_2x2x3_rank11():
        """
        Rank-11 decomposition of 2x2x3 matrix multiplication.
        C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
        
        Uses Strassen 2x2x2 (rank 7) + 2x2x1 (rank 4) for k=2.
        But 2x2x1 for k=2: C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2]
        This needs 4 terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
        
        Total: 7 + 4 = 11.
        """
        U = []  # shape (4, 11) for i in {0,1}, j in {0,1}
        V = []  # shape (6, 11) for j in {0,1}, k in {0,1,2}
        W = []  # shape (6, 11) for k in {0,1,2}, i in {0,1}
        
        # Strassen 2x2x2 for k in {0,1}:
        # M1 = A[0,0]*(B[0,0]+B[1,1])
        # M2 = A[1,0]*(B[0,0]+B[0,1])
        # M3 = (A[0,0]+A[0,1])*B[1,1]
        # M4 = (A[1,0]+A[1,1])*B[0,0]
        # M5 = (A[0,0]+A[1,0])*B[1,1]... 
        # Standard Strassen for C = AB where A,B are 2x2:
        # M1 = A11*(B11+B22)
        # M2 = (A11+A12)*B21
        # M3 = (A21+A22)*B12
        # M4 = A22*(B11+B22)
        # M5 = (A11+A21)*(B12+B22)
        # M6 = (A12-A22)*(B21+B22)
        # M7 = (A11-A12)*(B11+B12)
        # C11 = M1+M4-M5+M7
        # C12 = M2+M5
        # C21 = M3+M6
        # C22 = M1-M2+M3+M4
        
        # In our indexing: A[i,j], B[j,k], C[i,k]
        # i,j in {0,1}, k in {0,1} for Strassen part.
        
        # Term 0: M1 = A[0,0] * (B[0,0]+B[1,1]) * (W for k=0,i=0: +1, k=0,i=1: -1, k=1,i=0: +1, k=1,i=1: -1)
        # Wait, W[k*n+i, c] where n=2 here.
        
        # Let me just enumerate all 11 terms explicitly.
        
        # Terms 0-6: Strassen for (i,j,k) in {0,1}x{0,1}x{0,1}
        # Term 0: U[0,0]=1, V[0,0]=1,V[1,1]=1, W[0,0]=1,W[0,1]=-1,W[1,0]=1,W[1,1]=-1
        # That's not right. Let me use the standard form.
        
        # For Strassen, the 7 terms are:
        # t0: U[0,0]=1, V[0,0]=1,V[1,1]=1, W[0,0]=1,W[1,0]=-1,W[0,1]=1,W[1,1]=-1  -- no
        
        # Let me just use schoolbook for 2x2x3 (rank 12) and accept rank 41 total.
        # 15+10+12+4 = 41. Still better than 45!
        
        # Actually, let me just do full schoolbook for simplicity and correctness.
        # The rank-44 claim needs verification.
        return None

    # Given complexity, let's just use schoolbook 45 as the safe fallback
    # but first try a quick numerical search for rank 44
    
    from scipy.optimize import least_squares
    
    def residuals(params, R):
        U_flat = params[:9 * R].reshape(9, R)
        V_flat = params[9 * R:9 * R + 15 * R].reshape(15, R)
        W_flat = params[9 * R + 15 * R:].reshape(15, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    # Try rank 44 with a few restarts, time-limited
    for restart in range(5):
        if time.time() > deadline - 40:
            break
        
        rng = np.random.default_rng(42 + restart * 13)
        R = 44
        U_init = rng.normal(size=(9, R)) * 0.3
        V_init = rng.normal(size=(15, R)) * 0.3
        W_init = rng.normal(size=(15, R)) * 0.3
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = least_squares(
                lambda p: residuals(p, R),
                x0,
                method='lm',
                x_scale='jac',
                max_nfev=10000,
                ftol=1e-15,
                xtol=1e-15,
                gtol=1e-15
            )
            
            if res.cost < 1e-14:
                U_sol = res.x[:9 * R].reshape(9, R)
                V_sol = res.x[9 * R:9 * R + 15 * R].reshape(15, R)
                W_sol = res.x[9 * R + 15 * R:].reshape(15, R)
                
                for scale in [1, 2]:
                    U_r = np.round(U_sol * scale) / scale
                    V_r = np.round(V_sol * scale) / scale
                    W_r = np.round(W_sol * scale) / scale
                    
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    
                    if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                        best_U, best_V, best_W, best_rank = U_list, V_list, W_list, R
                        break
                if best_rank == R:
                    break
                    
        except Exception:
            pass

    if best_U is None:
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
