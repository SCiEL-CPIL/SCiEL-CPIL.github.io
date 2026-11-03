import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4
    
    def build_2x2x3():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x2.
        U: 4x11 (indices i*2+j), V: 6x11 (indices j*3+k), W: 6x11 (indices k*2+i)
        """
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                # W2 is indexed by k*2+i for k=0,1 and i=0,1
                # In 2x2x3, W is indexed by k*2+i for k=0,1,2 and i=0,1
                # So for k=0,1: W[k*2+i] = W2[k*2+i]
                W[idx][r] = W2[idx][r]
        
        # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1  # k=2, i in {0,1}, so W[4+i]
        
        return U, V, W
    
    def kronecker_decompose_4x4x2():
        """
        Build a rank-28 decomposition for 4x4x2 using Kronecker product
        of two 2x2x2 Strassen decompositions, each contributing rank 14.
        U: 16x28, V: 8x28, W: 8x28
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*28 for _ in range(16)]
        V4 = [[0]*28 for _ in range(8)]
        W4 = [[0]*28 for _ in range(8)]
        
        # For each of the 2 output columns (k=0,1), use a 2x2x2 Strassen
        # per 2x2 block, total 4 blocks * 7 = 28
        for k_out in range(2):
            for row_group in range(2):
                for col_group in range(2):
                    for r_local in range(7):
                        r_global = k_out * 14 + (row_group * 2 + col_group) * 7 + r_local
                        
                        for i_local in range(2):
                            for j_local in range(2):
                                i = 2 * row_group + i_local
                                j = 2 * col_group + j_local
                                U4[i*4 + j][r_global] = U2[i_local*2 + j_local][r_local]
                        
                        for j_local in range(2):
                            j = 2 * col_group + j_local
                            V4[j*2 + k_out][r_global] = V2[j_local*2 + 0][r_local]
                        
                        for i_local in range(2):
                            i = 2 * row_group + i_local
                            W4[k_out*4 + i][r_global] = W2[0*2 + i_local][r_local]
        
        return U4, V4, W4
    
    def build_4x4x7_decomposition():
        """
        Build a rank-93 decomposition for 4x4x7.
        
        Strategy: Use Kronecker Strassen (rank 49) for the 4x4x4 part.
        For the 4x4x3 part (columns 4,5,6 of B):
          - 4x4x2 via Kronecker Strassen (rank 28)
          - 4x4x1 via schoolbook (rank 16)
        Grand total: 49 + 28 + 16 = 93.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        U42, V42, W42 = kronecker_decompose_4x4x2()
        r42 = len(U42[0])  # 28
        
        r_total = r4 + r42 + 16  # 49 + 28 + 16 = 93
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Part 1: 4x4x4 Kronecker (columns 0-3 of B)
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Part 2a: 4x4x2 via Kronecker Strassen (columns 4-5 of B)
        col_offset = r4
        for i in range(16):
            for r in range(r42):
                U[i][col_offset + r] = U42[i][r]
        
        for j in range(4):
            for k in range(2):
                for r in range(r42):
                    V[j*7 + (4 + k)][col_offset + r] = V42[j*2 + k][r]
        
        for k in range(2):
            for i in range(4):
                for r in range(r42):
                    W[(4 + k)*4 + i][col_offset + r] = W42[k*4 + i][r]
        
        # Part 2b: 4x4x1 schoolbook (column 6 of B)
        col_offset2 = r4 + r42
        for i in range(4):
            for j in range(4):
                r = col_offset2 + i*4 + j
                U[i*4 + j][r] = 1
                V[j*7 + 6][r] = 1
                W[6*4 + i][r] = 1
        
        return U, V, W
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_opt, V_opt, W_opt = build_4x4x7_decomposition()
            r_opt = len(U_opt[0])
            
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
        except Exception as e:
            pass
    
    # Try to reduce rank by identifying and merging redundant terms
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 93:
        pass  # Already at 93, skip
    
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN and best_rank == 93:
        try:
            U_arr = np.array(best_result["U"], dtype=float)
            V_arr = np.array(best_result["V"], dtype=float)
            W_arr = np.array(best_result["W"], dtype=float)
            r_cur = U_arr.shape[1]
            
            # Try to find linearly dependent rank-1 terms and remove them
            # Build the tensor and check for dependencies
            reduced = False
            for attempt in range(min(10, r_cur - 92)):
                if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                    break
                
                # Try removing each term and checking validity
                for r_remove in range(r_cur):
                    if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                        break
                    
                    # Skip if this term is already zero
                    if (np.all(U_arr[:, r_remove] == 0) or 
                        np.all(V_arr[:, r_remove] == 0) or 
                        np.all(W_arr[:, r_remove] == 0)):
                        continue
                    
                    # Try to compensate by adjusting other terms
                    # This is a linear system: for each (i,j,k), 
                    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
                    # If we remove term r_remove, we need to adjust others
                    # to maintain the constraint.
                    
                    # Simple check: can we set this term to zero and still satisfy?
                    # Only if the term contributes zero to all (i,j,k)
                    contributes = False
                    for i in range(4):
                        for j in range(4):
                            for k in range(7):
                                contrib = 0.0
                                for c in range(r_cur):
                                    if c == r_remove:
                                        continue
                                    contrib += U_arr[i*4+j, c] * V_arr[j*7+k, c] * W_arr[k*4+i, c]
                                full = 0.0
                                for c in range(r_cur):
                                    full += U_arr[i*4+j, c] * V_arr[j*7+k, c] * W_arr[k*4+i, c]
                                if abs(full - 1.0) > 1e-9:
                                    break
                                if abs(contrib - full) > 1e-12:
                                    contributes = True
                                    break
                            if contributes:
                                break
                        if contributes:
                            break
                    if not contributes:
                        # This term is redundant
                        U_arr = np.delete(U_arr, r_remove, axis=1)
                        V_arr = np.delete(V_arr, r_remove, axis=1)
                        W_arr = np.delete(W_arr, r_remove, axis=1)
                        r_cur -= 1
                        reduced = True
                        break
                
                if not reduced:
                    break
            
            if reduced:
                best_result = {
                    "U": U_arr.tolist(),
                    "V": V_arr.tolist(),
                    "W": W_arr.tolist()
                }
                best_rank = r_cur
        except Exception as e:
            pass
    
    return best_result
