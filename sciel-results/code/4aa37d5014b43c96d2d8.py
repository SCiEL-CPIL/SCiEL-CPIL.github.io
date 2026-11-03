import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 135.0
SAFETY_MARGIN = 10.0

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
        
        # Indexing:
        # U[i*2+j, r] for A[i,j]
        # V[j*2+k, r] for B[j,k]
        # W[k*2+i, r] for C[i,k] (note: C[i,k] corresponds to W[k*2+i])
        
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1  # a11, a22
        V[0][0] = 1; V[3][0] = 1  # b11, b22
        W[0][0] = 1; W[3][0] = 1  # c11, c22
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1  # a21, a22
        V[0][1] = 1  # b11
        W[1][1] = 1; W[3][1] = -1  # c21, -c22
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1  # a11
        V[1][2] = 1; V[3][2] = -1  # b12, -b22
        W[2][2] = 1; W[3][2] = 1  # c12, c22
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1  # a22
        V[2][3] = 1; V[0][3] = -1  # b21, -b11
        W[0][3] = 1; W[1][3] = 1  # c11, c21
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1  # a11, a12
        V[3][4] = 1  # b22
        W[0][4] = -1; W[2][4] = 1  # -c11, c12
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1  # a21, -a11
        V[0][5] = 1; V[1][5] = 1  # b11, b12
        W[3][5] = 1  # c22
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1  # a12, -a22
        V[2][6] = 1; V[3][6] = 1  # b21, b22
        W[0][6] = 1  # c11
        
        return U, V, W
    
    def build_4x4x7_optimized():
        """
        Build an optimized decomposition for 4x4x7.
        
        Strategy: Split the 4x4x7 problem into blocks.
        View A as 2x2 blocks of 2x2 matrices, B as 2x2 blocks of 2x2x2 matrices (since p=7, 
        we split into 2x2 and 2x1 blocks), and C similarly.
        
        Actually, let's use a different approach:
        - For the first 4 columns of B (k=0..3), use a 4x4x4 decomposition via Kronecker
        - For the last 3 columns of B (k=4..6), use schoolbook
        - But optimize the 4x4x4 part using Strassen at the block level
        
        Better approach: Use the fact that 4x4x7 can be decomposed by treating it as
        multiplying 4x4 by 4x7. We can split B into B1 (4x4) and B2 (4x3).
        C = A*B1 + A*B2
        
        For A*B1 (4x4x4): Use Kronecker product of two Strassen 2x2x2 decompositions
        For A*B2 (4x4x3): Use schoolbook = 4*4*3 = 48 terms
        Total: 49 + 48 = 97
        
        But we can do better. Let's try a direct block approach:
        Split A into 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
        Split B into 2x2 blocks of 2x2 and 2x1: B = [[B11, B12, B13], [B21, B22, B23]]
        where B11,B21 are 2x4 and B12,B22 are 2x4 and B13,B23 are 2x2... 
        This is getting complicated. Let me just fix the Kronecker approach properly.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        # Build 4x4x4 via Kronecker product of two 2x2x2 Strassen
        # U4 has 16 rows (4*4), V4 has 16 rows (4*4), W4 has 16 rows (4*4)
        # Rank = 7*7 = 49
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                # U4[i*4+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                # where i = i1*2 + i2, j = j1*2 + j2
                for i1 in range(2):
                    for i2 in range(2):
                        for j1 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                    
                # V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                # where j = j1*2 + j2, k = k1*2 + k2
                for j1 in range(2):
                    for j2 in range(2):
                        for k1 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                    
                # W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
                # where k = k1*2 + k2, i = i1*2 + i2
                for k1 in range(2):
                    for k2 in range(2):
                        for i1 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        # Now build 4x4x7 by extending with 3 more columns using schoolbook
        r4 = 49
        r_extra = 4 * 4 * 3  # 48
        r_total = r4 + r_extra
        
        U = [[0] * r_total for _ in range(16)]   # 4*4 rows
        V = [[0] * r_total for _ in range(28)]   # 4*7 rows
        W = [[0] * r_total for _ in range(28)]   # 7*4 rows
        
        # Copy 4x4x4 part
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
        
        # Add schoolbook for k=4,5,6
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j][col_offset] = 1
                    V[j*7 + k][col_offset] = 1
                    W[k*4 + i][col_offset] = 1
                    col_offset += 1
        
        return U, V, W
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized Kronecker approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_optimized()
            r_kron = len(U_kron[0])
            
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception as e:
            pass
    
    # Try random search to reduce rank further
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        U, V, W = best_result["U"], best_result["V"], best_result["W"]
        r = len(U[0])
        
        # Convert to numpy for efficiency
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            # Try to merge two random columns
            if r <= 1:
                break
                
            c1 = random.randint(0, r-1)
            c2 = random.randint(0, r-1)
            
            if c1 == c2:
                continue
            
            # Try to find a linear combination that eliminates one column
            alpha = random.uniform(-1, 1)
            beta = random.uniform(-1, 1)
            
            # New column c1 = alpha * old_c1 + beta * old_c2
            # New column c2 = 0 (removed)
            
            U_new = U_np.copy()
            V_new = V_np.copy()
            W_new = W_np.copy()
            
            U_new[:, c1] = alpha * U_new[:, c1] + beta * U_new[:, c2]
            V_new[:, c1] = alpha * V_new[:, c1] + beta * V_new[:, c2]
            W_new[:, c1] = alpha * W_new[:, c1] + beta * W_new[:, c2]
            
            # Remove column c2
            mask = [i for i in range(r) if i != c2]
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            # Check validity
            r_new = len(U_new[0])
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r_new):
                            s += U_new[i*m + j, c] * V_new[j*p + k, c] * W_new[k*n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid and r_new < best_rank:
                U_list = U_new.tolist()
                V_list = V_new.tolist()
                W_list = W_new.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r_new
                U_np = U_new
                V_np = V_new
                W_np = W_new
                r = r_new
    
    return best_result
