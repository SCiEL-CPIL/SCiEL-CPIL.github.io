import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = 145.0

    def verify(U, V, W, atol=1e-8):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b_idx in range(m):
                        for bp in range(m):
                            for c in range(p):
                                expected = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                s = 0.0
                                for l in range(r):
                                    s += W[k * n + i, l] * U[a * m + b_idx, l] * V[bp * p + c, l]
                                if abs(s - expected) > atol:
                                    return False
        return True

    # Schoolbook baseline (rank 36)
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_sb[ii * m + jj, col] = 1
                V_sb[jj * p + kk, col] = 1
                W_sb[kk * n + ii, col] = 1
                col += 1

    # Rank-11 Strassen-like decomposition for 2x3x3
    def get_2x3x3_rank11():
        # U: 6x11, V: 9x11, W: 6x11
        # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((6, 11))
        
        # A = [[a, b, c], [d, e, f]]
        # B = [[g, h, i], [j, k, l], [m, n, o]]
        # C = A*B
        # C[0,0] = ag+bj+cm
        # C[0,1] = ah+bk+dn
        # C[0,2] = ai+bl+eo
        # C[1,0] = dg+ej+fm
        # C[1,1] = dh+ek+fn
        # C[1,2] = di+el+fo
        
        # We want 11 multiplications.
        # Let's try to find a valid set of 11 terms.
        # This is a known result.
        
        # Term 1: (a+b) * (g+j) -> ag + aj + bg + bj
        # Term 2: (c) * (g+m) -> cg + cm
        # Term 3: (a-d) * (j) -> aj - dj
        # Term 4: (f) * (k+n) -> fk + fn
        # Term 5: (b) * (h+k) -> bh + bk
        # Term 6: (e) * (h-k) -> eh - ek
        # Term 7: (d+e) * (k) -> dk + ek
        # Term 8: (a+c) * (i) -> ai + ci  (Wait, C[0,2] needs ai+bl+eo. C[1,2] needs di+el+fo)
        # This manual construction is error-prone. 
        # Let's use a known valid rank 11 decomposition or fallback to rank 9 (schoolbook 3x3) if 11 is too complex to hardcode correctly without a solver.
        # Actually, schoolbook for 2x3x3 is rank 9.
        # Is there a rank < 9? No, lower bound is 9.
        # So rank 9 is optimal for 2x3x3.
        # Wait, the "inspiration" said "Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6".
        # If rank 9 is possible for 2x3x3, then 2*9 = 18 is possible for 2x3x6.
        # Why did the inspiration say 11? Maybe it was a different dimension or a suboptimal known algorithm?
        # Or maybe 2x3x3 rank is actually 9.
        # Let's check: 2x3x3 matrix multiplication.
        # Inputs: 2*3=6, 3*3=9. Output: 2*3=6.
        # Schoolbook: 2*3*3 = 18? No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # i in 0..1, k in 0..2, j in 0..2.
        # Total terms: 2*3*3 = 18.
        # So schoolbook rank is 18.
        # Can we do better?
        # Lower bound: max(6, 9, 6) = 9.
        # Is rank 9 possible? Unlikely for 2x3x3.
        # Is rank 11 possible?
        # If we use a 3x3x3 Strassen (rank 7), we can do 2x3x3 by treating the 2x3 matrix as 2 rows of 1x3.
        # This doesn't directly map.
        
        # Let's assume the "inspiration" is correct that a rank 11 exists or was found.
        # But if I can't find it, I should stick to what I can verify.
        # Schoolbook 2x3x3 is rank 18.
        # Two copies -> rank 36.
        
        # Let's try to find a better decomposition for 2x3x6 directly using the structure.
        # The matrix B is 3x6. We can split B into B1 (3x3) and B2 (3x3).
        # C = [A*B1, A*B2].
        # If we have a decomposition for A*B1 with rank r, and A*B2 with rank r, total rank 2r.
        # If we can share factors between B1 and B2 calculations, we might reduce rank.
        # But A is the same.
        
        # Let's try to use a known low-rank algorithm for 2x3x3.
        # Actually, for 2x3x3, the rank is known to be 18? No, lower bound is 9.
        # Let's check literature mentally. 2x2x2 is 7. 2x3x3 is likely > 9.
        # If I cannot guarantee a rank < 18 for 2x3x3, I cannot guarantee < 36 for 2x3x6 via this split.
        
        # However, the prompt says "Uses two copies of a Rank 11 decomposition for 2x3x3... Measured: 1 -> 1.059".
        # 1.059 corresponds to rank 34? No.
        # Score = seed/value. Seed = 36.
        # If score is 1.059, value = 36 / 1.059 = 33.99... ~ 34.
        # So the "Rank 11" claim in the inspiration might be misleading or I am misinterpreting "Rank 11".
        # If it used rank 11 for 2x3x3, then 2*11 = 22.
        # Score for rank 22 would be 36/22 = 1.636.
        # The inspiration says score 1.059. This implies rank ~34.
        # This is contradictory. "Uses two copies of a Rank 11... to construct a Rank 22... improving score from 34 to 22".
        # Wait, "improving score from 34 to 22" usually means the *rank* improved from 34 to 22?
        # If rank improved from 34 to 22, score improves from 36/34=1.058 to 36/22=1.636.
        # The text says "Measured: 1 -> 1.059". This suggests the score went from 1.0 to 1.059.
        # 1.059 * 36 = 38.1? No.
        # Maybe the seed is not 36?
        # "score is the rank" ... "seed/value, so 1.0 equals the seed program".
        # If seed program has rank 36, score 1.0.
        # If child has rank 34, score 36/34 = 1.0588.
        # So the "inspiration" program likely found rank 34, not 22.
        # The text "Uses two copies of a Rank 11... to construct a Rank 22" might be a hallucination or referring to a different sub-problem.
        # OR, it found a rank 11 for 2x3x3, but failed to combine them optimally, ending up at 34?
        # Or maybe it used a rank 17 for 2x3x3? 2*17 = 34.
        
        # Let's focus on finding a rank < 36.
        # Strategy: Use the block structure.
        # B = [B1, B2].
        # C1 = A B1, C2 = A B2.
        # We can compute C1 and C2 independently.
        # If we find a rank 17 decomposition for 2x3x3, we get rank 34 for 2x3x6.
        # Schoolbook 2x3x3 is 18.
        # Can we do 17?
        # Let's try to search for a rank 17 decomposition of 2x3x3.
        
        def search_2x3x3(target_rank):
            n_eq = 2 * 3 * 2 * 3 * 3 * 3 # 324
            n_U = 6
            n_V = 9
            n_W = 6
            
            def build_system(U, V, r):
                A = np.zeros((n_eq, n_W * r))
                b = np.zeros(n_eq)
                eq = 0
                for i in range(2):
                    for k in range(3):
                        for a in range(2):
                            for b_idx in range(3):
                                for bp in range(3):
                                    for c in range(3):
                                        b[eq] = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                        for l in range(r):
                                            col = (k * 2 + i) * r + l
                                            A[eq, col] = U[a * 3 + b_idx, l] * V[bp * 3 + c, l]
                                        eq += 1
                return A, b

            def solve_for_W(U, V, r):
                A, b = build_system(U, V, r)
                W_flat, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(n_W, r)
                if residuals and len(residuals) > 0:
                    res = residuals[0]
                else:
                    res = np.linalg.norm(A @ W_flat - b)
                return W, res

            # Try rank 17
            best_res = float('inf')
            best_U = best_V = best_W = None
            
            # Random restarts
            for restart in range(100):
                if time.time() - start_time > deadline - 20:
                    break
                U = np.random.randn(n_U, target_rank) * 0.5
                V = np.random.randn(n_V, target_rank) * 0.5
                
                # Coordinate descent
                for it in range(20):
                    if time.time() - start_time > deadline - 10:
                        break
                    # Update U
                    for l in range(target_rank):
                        U_new = U.copy()
                        U_new[:, l] = np.random.randn(n_U) * 0.5
                        W_trial, res_trial = solve_for_W(U_new, V, target_rank)
                        if res_trial < best_res:
                            best_res = res_trial
                            if res_trial < 1e-8:
                                best_U, best_V, best_W = U_new, V, W_trial
                                return best_U, best_V, best_W
                            U = U_new
                    # Update V
                    for l in range(target_rank):
                        V_new = V.copy()
                        V_new[:, l] = np.random.randn(n_V) * 0.5
                        W_trial, res_trial = solve_for_W(U, V_new, target_rank)
                        if res_trial < best_res:
                            best_res = res_trial
                            if res_trial < 1e-8:
                                best_U, best_V, best_W = U, V_new, W_trial
                                return best_U, best_V, best_W
                            V = V_new
                if best_res < 1e-8:
                    return best_U, best_V, best_W
            return None, None, None

        # Try to find rank 17 for 2x3x3
        U_233, V_233, W_233 = search_2x3x3(17)
        
        if U_233 is not None:
            # Construct 2x3x6 rank 34
            r_236 = 34
            U_236 = np.zeros((6, r_236))
            V_236 = np.zeros((18, r_236))
            W_236 = np.zeros((12, r_236))
            
            # Block 1: k=0,1,2
            # U_236[:, 0:17] = U_233
            U_236[:, 0:17] = U_233
            # V_236[0:9, 0:17] = V_233
            V_236[0:9, 0:17] = V_233
            # V_236[9:18, 0:17] = 0
            # W_236[0:6, 0:17] = W_233 (for k=0,1,2, i=0,1)
            # W index: k*n+i.
            # For k=0,1,2, i=0,1:
            # k=0: i=0 -> 0, i=1 -> 1
            # k=1: i=0 -> 2, i=1 -> 3
            # k=2: i=0 -> 4, i=1 -> 5
            # W_233 index: k*n+i.
            # So W_236[0:6, 0:17] = W_233
            
            # Block 2: k=3,4,5
            # Shift k by 3.
            # V_236[9:18, 17:34] = V_233
            V_236[9:18, 17:34] = V_233
            # V_236[0:9, 17:34] = 0
            # U_236[:, 17:34] = U_233
            U_236[:, 17:34] = U_233
            # W_236[6:12, 17:34] = W_233
            # k=3: i=0 -> 6, i=1 -> 7
            # k=4: i=0 -> 8, i=1 -> 9
            # k=5: i=0 -> 10, i=1 -> 11
            W_236[6:12, 17:34] = W_233
            
            if verify(U_236, V_236, W_236):
                return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
