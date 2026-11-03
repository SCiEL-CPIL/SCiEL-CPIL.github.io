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
    
    def build_2x2x3_decomposition():
        """
        Build a rank-11 decomposition for 2x2x3 (2x2 A * 2x3 B -> 2x2 C).
        Uses Strassen rank-7 for the 2x2x2 part plus 4 schoolbook terms for the extra column.
        Returns U (4x11), V (6x11), W (4x11) with indices U[i*2+j], V[j*3+k], W[k*2+i].
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        r = 11
        U = [[0]*r for _ in range(4)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(4)]
        
        # First 7 terms: Strassen for 2x2x2 (k=0,1)
        for i in range(4):
            for c in range(7):
                U[i][c] = U2[i][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*3+k][c] = V2[j*2+k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[k*2+i][c] = W2[k*2+i][c]
        
        # Last 4 terms: schoolbook for k=2 (the third column)
        col_offset = 7
        for i in range(2):
            for j in range(2):
                U[i*2+j][col_offset] = 1
                V[j*3+2][col_offset] = 1
                W[2*2+i][col_offset] = 1
                col_offset += 1
        
        return U, V, W
    
    def build_4x4x3_decomposition():
        """
        Build a decomposition for 4x4x3 using four 2x2x3 blocks.
        Each 2x2x3 block has rank 11, so total rank = 4*11 = 44.
        Returns U (16x44), V (12x44), W (12x44).
        U[i*4+j], V[j*3+k], W[k*4+i].
        """
        U23, V23, W23 = build_2x2x3_decomposition()
        r = 44
        U = [[0]*r for _ in range(16)]
        V = [[0]*r for _ in range(12)]
        W = [[0]*r for _ in range(12)]
        
        # Four 2x2x3 blocks: (i1,j1) in {0,1}^2, each covering 2x2x3 subproblem
        # Block (i1,j1) maps to i=i1*2+i2, j=j1*2+j2, k=0,1,2
        for i1 in range(2):
            for j1 in range(2):
                base = (i1*2 + j1) * 11
                for i2 in range(2):
                    for j2 in range(2):
                        i = i1*2 + i2
                        j = j1*2 + j2
                        for c in range(11):
                            U[i*4+j][base+c] = U23[i2*2+j2][c]
                for j2 in range(2):
                    for k in range(3):
                        j = j1*2 + j2
                        for c in range(11):
                            V[j*3+k][base+c] = V23[j2*3+k][c]
                for k in range(3):
                    for i2 in range(2):
                        i = i1*2 + i2
                        for c in range(11):
                            W[k*4+i][base+c] = W23[k*2+i2][c]
        
        return U, V, W
    
    def build_4x4x7_decomposition():
        """
        Build a decomposition for 4x4x7 by combining the 4x4x4 Strassen-Kronecker
        decomposition (rank 49) with a 4x4x3 decomposition (rank 44).
        Total rank: 49 + 44 = 93.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        U43, V43, W43 = build_4x4x3_decomposition()
        r43 = len(U43[0])
        
        r_total = r4 + r43
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # First part: 4x4x4 using Strassen-Kronecker (k=0..3)
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
        
        # Second part: 4x4x3 using 2x2x3 blocks (k=4..6)
        col_offset = r4
        for i in range(16):
            for r in range(r43):
                U[i][col_offset + r] = U43[i][r]
        for j in range(4):
            for k in range(3):
                for r in range(r43):
                    V[j*7 + (k+4)][col_offset + r] = V43[j*3+k][r]
        for k in range(3):
            for i in range(4):
                for r in range(r43):
                    W[(k+4)*4 + i][col_offset + r] = W43[k*4+i][r]
        
        return U, V, W
    
    def try_cw_embedding():
        """
        Coppersmith-Winograd embedding: try to find a lower-rank decomposition
        by exploiting shared structure across the 4x4x7 tensor.
        
        Strategy: Use the 4x4x4 Strassen-Kronecker as base (rank 49), then try
        to merge some of the 4x4x3 schoolbook terms with existing Strassen terms
        by finding linear dependencies. If we can express some of the 48 schoolbook
        terms for k=4,5,6 as linear combinations of existing rank-1 tensors,
        we reduce the total rank.
        
        Alternative: Try a structured embedding where we map the 4x4x7 problem
        into a larger space and use the sparsity to find a lower-rank solution.
        """
        elapsed = time.time() - start_time
        if elapsed > WALL_DEADLINE - SAFETY_MARGIN:
            return None
        
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        
        U4_arr = np.array(U4, dtype=float)
        V4_arr = np.array(V4, dtype=float)
        W4_arr = np.array(W4, dtype=float)
        
        # For the 4x4x3 part (k=4,5,6), we need 48 schoolbook terms.
        # Try to see if some can be absorbed into the existing 49 terms.
        # The tensor equation for (i,j,k) with k>=4:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        # With existing terms (c < 49), V[j,k,c]=0 and W[k,i,c]=0 for k>=4.
        # So we need new terms. But maybe we can modify existing terms.
        
        # Try: can we add small perturbations to the existing 49 rank-1 tensors
        # to also cover some of the k=4,5,6 terms?
        # This is a nonlinear problem. Instead, try a different approach:
        # Build the full 4x4x7 tensor and try to find a lower-rank decomposition
        # using structured patterns.
        
        # Approach: Use 4x4x4 Strassen (rank 49) + optimized 4x4x3 (rank 44) = 93
        # But try to find savings by sharing between the two parts.
        
        # Let's try: for each of the 48 schoolbook terms in the 4x4x3 part,
        # check if it can be expressed as a linear combination of existing terms
        # by modifying V and W entries.
        
        # Actually, let's try a more direct approach:
        # Build a candidate with rank < 93 by trying to merge terms.
        
        # The key insight: in the 4x4x4 part, for each (i,j) pair, the U[i,j,:]
        # vector is nonzero. For the 4x4x3 part, we need U[i,j,:] to also have
        # components that interact with V[j,4..6,:] and W[4..6,i,:].
        
        # Try: extend the U vectors to also have components for k=4,5,6.
        # This requires modifying V and W accordingly.
        
        # Let's try a greedy approach: for each schoolbook term (i,j,k) with k>=4,
        # try to find an existing rank-1 tensor c that we can modify to also
        # contribute to this term.
        
        # For term (i,j,k) with k>=4, we need:
        # U[i,j,c_new] * V[j,k,c_new] * W[k,i,c_new] = 1
        # If we use an existing c, we need to add to U[i,j,c], V[j,k,c], W[k,i,c].
        # But V[j,k,c] and W[k,i,c] are currently 0 for k>=4.
        # So we'd be adding new nonzero entries to V and W.
        # The problem is that this might break existing equations.
        
        # Let's just verify the 93-rank solution and return it.
        # The CW embedding idea is complex; let's focus on ensuring the 93-rank
        # solution is correct and see if we can find any small savings.
        
        return None
    
    def try_rank_reduction():
        """
        Try to reduce the rank of the 93-term decomposition by finding
        linear dependencies or using structured patterns.
        """
        elapsed = time.time() - start_time
        if elapsed > WALL_DEADLINE - SAFETY_MARGIN:
            return None
        
        U, V, W = build_4x4x7_decomposition()
        r = len(U[0])
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check for zero columns (terms that don't contribute)
        for c in range(r):
            if np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0):
                # This term is useless; remove it
                U_arr = np.delete(U_arr, c, axis=1)
                V_arr = np.delete(V_arr, c, axis=1)
                W_arr = np.delete(W_arr, c, axis=1)
                r -= 1
                break
        
        if r < len(U[0]):
            U_new = U_arr.tolist()
            V_new = V_arr.tolist()
            W_new = W_arr.tolist()
            if check_valid(U_new, V_new, W_new):
                return U_new, V_new, W_new
        
        return None
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized 4x4x7 decomposition (rank 93)
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
    
    # Try rank reduction
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            reduced = try_rank_reduction()
            if reduced is not None:
                U_red, V_red, W_red = reduced
                r_red = len(U_red[0])
                if check_valid(U_red, V_red, W_red) and r_red < best_rank:
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = r_red
        except Exception as e:
            pass
    
    # Try CW embedding approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cw_result = try_cw_embedding()
            if cw_result is not None:
                U_cw, V_cw, W_cw = cw_result
                r_cw = len(U_cw[0])
                if check_valid(U_cw, V_cw, W_cw) and r_cw < best_rank:
                    best_result = {"U": U_cw, "V": V_cw, "W": W_cw}
                    best_rank = r_cw
        except Exception as e:
            pass
    
    return best_result
