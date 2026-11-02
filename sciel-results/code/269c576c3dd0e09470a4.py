"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank14():
        """Build the known rank-14 Strassen decomposition."""
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B)
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B)
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        return U, V, W

    def try_rank13_merge():
        """
        Try to find a rank-13 decomposition by merging one term from the left
        Strassen block with one from the right block.
        
        For each pair (li, ri) where li is a left-block term index and ri is a
        right-block term index, we check if there exists a single rank-1 tensor
        (u_new, v_new, w_new) such that the sum of all 14 terms minus the two
        removed terms plus the new term equals the target tensor.
        
        This is equivalent to checking if the difference tensor D = T_left_li + T_right_ri
        has rank 1, where T_left_li and T_right_ri are the rank-1 contributions.
        
        More precisely: we need to find u_new, v_new, w_new such that for all
        (i,j,k): u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i] equals the sum of
        the contributions from terms li and ri.
        """
        if time.time() > deadline - 5:
            return None
        
        U14, V14, W14 = build_rank14()
        
        # Extract term vectors
        U_terms = [np.array(U14[:, r], dtype=float) for r in range(14)]
        V_terms = [np.array(V14[:, r], dtype=float) for r in range(14)]
        W_terms = [np.array(W14[:, r], dtype=float) for r in range(14)]
        
        # For each pair (li, ri), compute the combined tensor and check if rank 1
        for li in range(7):
            for ri in range(7, 14):
                if time.time() > deadline - 3:
                    return None
                
                # Compute the combined contribution of terms li and ri
                # D[i,j,k] = U_li[i*m+j]*V_li[j*p+k]*W_li[k*n+i] + U_ri[i*m+j]*V_ri[j*p+k]*W_ri[k*n+i]
                # We need D to be rank 1, i.e., D = u_new ⊗ v_new ⊗ w_new
                
                # Build the 3D array D
                D = np.zeros((n, m, p), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            D[i,j,k] = (U_terms[li][i*m+j] * V_terms[li][j*p+k] * W_terms[li][k*n+i] +
                                       U_terms[ri][i*m+j] * V_terms[ri][j*p+k] * W_terms[ri][k*n+i])
                
                # Check if D is rank 1 using SVD
                # Reshape to (n, m*p) and check if rank 1
                D_2d = D.reshape(n, m*p)
                try:
                    U_svd, S, Vt = np.linalg.svd(D_2d, full_matrices=False)
                except:
                    continue
                
                if len(S) >= 2 and S[1] > 1e-10:
                    continue  # Not rank 1 in this mode
                
                # Also check other modes
                D_2d_mp = D.reshape(m, n*p)
                try:
                    U_svd2, S2, Vt2 = np.linalg.svd(D_2d_mp, full_matrices=False)
                except:
                    continue
                if len(S2) >= 2 and S2[1] > 1e-10:
                    continue
                
                D_2d_np = D.reshape(p, n*m)
                try:
                    U_svd3, S3, Vt3 = np.linalg.svd(D_2d_np, full_matrices=False)
                except:
                    continue
                if len(S3) >= 2 and S3[1] > 1e-10:
                    continue
                
                # If D is rank 1, extract the vectors
                # u_new from the first left singular vector of D_2d
                u_new = U_svd[:, 0] * S[0]
                # v_new from the first right singular vector of D_2d
                v_new = Vt[0, :] * S[0]
                # w_new needs to satisfy: u_new[i*m+j] * v_new[j*p+k] * w_new[k*n+i] = D[i,j,k]
                # For a given (i,j,k), w_new[k*n+i] = D[i,j,k] / (u_new[i*m+j] * v_new[j*p+k])
                
                # Check consistency
                w_new = np.zeros(pn, dtype=float)
                consistent = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            denom = u_new[i*m+j] * v_new[j*p+k]
                            if abs(denom) < 1e-12:
                                if abs(D[i,j,k]) > 1e-12:
                                    consistent = False
                                    break
                                else:
                                    w_new[k*n+i] = 0.0
                                    continue
                            val = D[i,j,k] / denom
                            if k*n+i >= 0 and k*n+i < pn:
                                if abs(w_new[k*n+i]) < 1e-12:
                                    w_new[k*n+i] = val
                                else:
                                    if abs(w_new[k*n+i] - val) > 1e-9:
                                        consistent = False
                                        break
                        if not consistent:
                            break
                    if not consistent:
                        break
                
                if not consistent:
                    continue
                
                # Build the rank-13 solution
                R = 13
                U_new = [[0]*R for _ in range(4)]
                V_new = [[0]*R for _ in range(8)]
                W_new = [[0]*R for _ in range(8)]
                
                # Copy all terms except li and ri
                new_idx = 0
                for r in range(14):
                    if r == li or r == ri:
                        continue
                    for row in range(4):
                        U_new[row][new_idx] = U14[row][r]
                    for row in range(8):
                        V_new[row][new_idx] = V14[row][r]
                        W_new[row][new_idx] = W14[row][r]
                    new_idx += 1
                
                # Add the new merged term
                for row in range(4):
                    U_new[row][new_idx] = u_new[row]
                for row in range(8):
                    V_new[row][new_idx] = v_new[row]
                    W_new[row][new_idx] = w_new[row]
                
                if verify(U_new, V_new, W_new):
                    return {"U": U_new, "V": V_new, "W": W_new}
        
        return None

    # Try rank-13 first
    result = try_rank13_merge()
    if result is not None:
        return result
    
    # Fallback to rank-14
    U14, V14, W14 = build_rank14()
    if verify(U14, V14, W14):
        return {"U": U14, "V": V14, "W": W14}
    
    # Final fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
