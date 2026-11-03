import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomp(n, m, p):
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

def _strassen_rank28():
    """
    Construct a rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Structure:
    - Block 1: Strassen-7 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    - Block 2: Strassen-7 for A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    - Block 3: Schoolbook-4 for A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    - Block 4: Schoolbook-6 for A[0:2, 2:3] * B[2:3, 0:5] -> C[0:2, 0:5]
    
    Total rank: 7 + 7 + 4 + 6 = 24... wait, let me recount.
    
    Actually, let me use the exact structure from inspiration 2:
    - Block 1: Strassen-7, j=0,1, k=0,1 -> 7 ranks
    - Block 2: Strassen-7, j=0,1, k=2,3 -> 7 ranks  
    - Block 3: 4 ranks for j=0,1, k=4 (2x1 block)
    - Block 4: 6 ranks for j=2, k=0..4 (1x5 block)
    Total: 7+7+4+6 = 24? But inspiration says rank 28.
    
    Let me re-examine. The inspiration uses rank=28. Let me trace through more carefully.
    
    Actually, looking at inspiration 2 more carefully, it seems like the blocks might overlap
    in a way that requires more ranks. Let me just use the exact code from inspiration 2
    and verify it.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(rank_offset, j_offset, k_offset):
        # A indices: i in 0..1, j in j_offset..j_offset+1
        a_idx = 0 * m + j_offset       # A[0, j_offset]
        b_idx = 0 * m + (j_offset + 1) # A[0, j_offset+1]
        c_idx = 1 * m + j_offset       # A[1, j_offset]
        d_idx = 1 * m + (j_offset + 1) # A[1, j_offset+1]
        
        # B indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1
        e_idx = j_offset * p + k_offset             # B[j_offset, k_offset]
        f_idx = j_offset * p + (k_offset + 1)       # B[j_offset, k_offset+1]
        g_idx = (j_offset + 1) * p + k_offset       # B[j_offset+1, k_offset]
        h_idx = (j_offset + 1) * p + (k_offset + 1) # B[j_offset+1, k_offset+1]
        
        r0 = rank_offset
        r1 = rank_offset + 1
        r2 = rank_offset + 2
        r3 = rank_offset + 3
        r4 = rank_offset + 4
        r5 = rank_offset + 5
        r6 = rank_offset + 6
        
        # U (L_A) - Strassen's 7 products for A:
        # M1 = (a+d)(e+h)
        U[a_idx][r0] += 1; U[d_idx][r0] += 1
        # M2 = d(e+f)
        U[d_idx][r1] += 1
        # M3 = (b-d)(g+h)
        U[b_idx][r2] += 1; U[d_idx][r2] -= 1
        # M4 = a(g+h)
        U[a_idx][r3] += 1
        # M5 = (c+a)g
        U[c_idx][r4] += 1; U[a_idx][r4] += 1
        # M6 = (b+c)(e-f)
        U[b_idx][r5] += 1; U[c_idx][r5] += 1
        # M7 = (a-c)(h-f)
        U[a_idx][r6] += 1; U[c_idx][r6] -= 1
        
        # V (L_B)
        V[e_idx][r0] += 1; V[h_idx][r0] += 1
        V[e_idx][r1] += 1; V[f_idx][r1] += 1
        V[g_idx][r2] += 1; V[h_idx][r2] += 1
        V[g_idx][r3] += 1; V[h_idx][r3] += 1
        V[g_idx][r4] += 1
        V[e_idx][r5] += 1; V[f_idx][r5] -= 1
        V[h_idx][r6] += 1; V[f_idx][r6] -= 1
        
        # W (Coefficients) - C[i,k] = sum of products
        # C[0,0] = M1 - M4 - M5 + M7
        W[k_offset * n + 0][r0] += 1
        W[k_offset * n + 0][r3] -= 1
        W[k_offset * n + 0][r4] -= 1
        W[k_offset * n + 0][r6] += 1
        
        # C[0,1] = M2 + M4
        W[(k_offset + 1) * n + 0][r1] += 1
        W[(k_offset + 1) * n + 0][r3] += 1
        
        # C[1,0] = M3 + M5
        W[k_offset * n + 1][r2] += 1
        W[k_offset * n + 1][r4] += 1
        
        # C[1,1] = M1 - M2 - M3 + M6
        W[(k_offset + 1) * n + 1][r0] += 1
        W[(k_offset + 1) * n + 1][r1] -= 1
        W[(k_offset + 1) * n + 1][r2] -= 1
        W[(k_offset + 1) * n + 1][r5] += 1

    # Block 1: j=0,1, k=0,1 (2x2 * 2x2 -> 2x2)
    add_strassen_block(0, 0, 0)
    
    # Block 2: j=0,1, k=2,3 (2x2 * 2x2 -> 2x2)
    add_strassen_block(7, 0, 2)
    
    # Block 3: j=0,1, k=4 (2x2 * 2x1 -> 2x1) - schoolbook 4 products
    r0 = 14
    r1 = 15
    r2 = 16
    r3 = 17
    
    # A[0,0]*B[0,4], A[0,1]*B[1,4], A[1,0]*B[0,4], A[1,1]*B[1,4]
    U[0 * m + 0][r0] += 1  # A[0,0]
    U[0 * m + 1][r1] += 1  # A[0,1]
    U[1 * m + 0][r2] += 1  # A[1,0]
    U[1 * m + 1][r3] += 1  # A[1,1]
    
    V[0 * p + 4][r0] += 1  # B[0,4]
    V[1 * p + 4][r1] += 1  # B[1,4]
    V[0 * p + 4][r2] += 1  # B[0,4]
    V[1 * p + 4][r3] += 1  # B[1,4]
    
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    W[4 * n + 0][r0] += 1
    W[4 * n + 0][r1] += 1
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    W[4 * n + 1][r2] += 1
    W[4 * n + 1][r3] += 1
    
    # Block 4: j=2, k=0..4 (2x1 * 1x5 -> 2x5) - schoolbook 10 products... 
    # But we only have 28-18=10 ranks left. 2x5 = 10 products. Good.
    r_base = 18
    for k in range(5):
        for i in range(2):
            r = r_base + k * 2 + i
            # A[i,2] * B[2,k]
            U[i * m + 2][r] += 1
            V[2 * p + k][r] += 1
            W[k * n + i][r] += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin from 150s limit
    
    # Try the rank-28 Strassen-based decomposition
    decomp28 = _strassen_rank28()
    if _verify_decomp(decomp28["U"], decomp28["V"], decomp28["W"], n, m, p):
        return decomp28
    
    # If verification fails, try a numerical polish approach
    # Use the rank-28 structure as initial guess and refine
    if time.time() < deadline - 20:
        R = 28
        U = np.array(decomp28["U"], dtype=float)
        V = np.array(decomp28["V"], dtype=float)
        W = np.array(decomp28["W"], dtype=float)
        
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    if np.linalg.matrix_rank(A) > 0:
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    if np.linalg.matrix_rank(A) > 0:
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    if np.linalg.matrix_rank(A) > 0:
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
            
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += abs(val - 1)
            
            if error < 1e-9:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = [[float(x) for x in row] for row in U_rounded]
                V_list = [[float(x) for x in row] for row in V_rounded]
                W_list = [[float(x) for x in row] for row in W_rounded]
                
                if _verify_decomp(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try higher ranks with numerical search
        for R in range(29, 31):
            if time.time() > deadline - 10:
                break
            
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Initialize with schoolbook pattern spread across ranks
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        if r < R:
                            U[i * m + j, r] = 1
                            V[j * p + k, r] = 1
                            W[k * n + i, r] = 1
            
            for iteration in range(300):
                if time.time() > deadline - 5:
                    break
                
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                            except:
                                pass
                
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R))
                        for i in range(n):
                            for r in range(R):
                                A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                            except:
                                pass
                
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R))
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        if np.linalg.matrix_rank(A) > 0:
                            try:
                                W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                            except:
                                pass
                
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += abs(val - 1)
                
                if error < 1e-9:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    U_list = [[float(x) for x in row] for row in U_rounded]
                    V_list = [[float(x) for x in row] for row in V_rounded]
                    W_list = [[float(x) for x in row] for row in W_rounded]
                    if _verify_decomp(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    return _schoolbook_decomp(n, m, p)
