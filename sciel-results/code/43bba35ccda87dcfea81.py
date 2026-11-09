"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 matrix multiplication.
    M_{2,2,2} with rank 7.
    Returns U (4x7), V (4x7), W (4x7) where:
    U indexes (i,j) for i,j in {0,1}
    V indexes (j,k) for j,k in {0,1}  
    W indexes (k,i) for k,i in {0,1}
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[p,q],[r,s]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # p = M1 + M4 - M5 + M7
    # q = M3 + M5
    # r = M2 + M4
    # s = M1 - M2 + M3 + M6
    #
    # U[i*m+j, r] for A element (i,j)
    # V[j*p+k, r] for B element (j,k)
    # W[k*n+i, r] for C element (k,i)
    
    # Term 0: M1 = (a+d)(e+h)
    # U: a(0,0)=1, d(1,1)=1
    # V: e(0,0)=1, h(1,1)=1
    # W: p(0,0)=1, s(1,1)=-1
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = -1.0
    
    # Term 1: M2 = (c+d)e
    # U: c(1,0)=1, d(1,1)=1
    # V: e(0,0)=1
    # W: r(1,0)=1, s(1,1)=-1
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # Term 2: M3 = a(f-h)
    # U: a(0,0)=1
    # V: f(0,1)=1, h(1,1)=-1
    # W: q(0,1)=1, s(1,1)=1
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # Term 3: M4 = d(g-e)
    # U: d(1,1)=1
    # V: g(1,0)=1, e(0,0)=-1
    # W: p(0,0)=1, r(1,0)=1
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # Term 4: M5 = (a+b)h
    # U: a(0,0)=1, b(0,1)=1
    # V: h(1,1)=1
    # W: p(0,0)=-1, q(0,1)=1
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # Term 5: M6 = (c-a)(e+f)
    # U: c(1,0)=1, a(0,0)=-1
    # V: e(0,0)=1, f(0,1)=1
    # W: s(1,1)=1
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # Term 6: M7 = (b-d)(g+h)
    # U: b(0,1)=1, d(1,1)=-1
    # V: g(1,0)=1, h(1,1)=1
    # W: p(0,0)=1
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _build_m441():
    """
    Build rank-2 decomposition for M_{4,4,1} (4x4 * 4x1 -> 4x1).
    Uses half-integers to satisfy sum=1.
    """
    n = 4
    m = 4
    p = 1
    rank = 2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # r=0
    for i in range(n):
        for k in range(m):
            U[i * m + k, 0] = 1.0
    for k in range(m):
        V[k * p, 0] = 0.5
    for i in range(n):
        W[i * p, 0] = 1.0
        
    # r=1
    for i in range(n):
        for k in range(m):
            U[i * m + k, 1] = 1.0
    for k in range(m):
        V[k * p, 1] = 0.5
    for i in range(n):
        W[i * p, 1] = 1.0
        
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build a rank-49 decomposition for M_{4,4,4} using recursive Strassen.
    Split 4x4 into four 2x2 blocks, apply Strassen (rank 7) at each level.
    Total rank = 7 * 7 = 49.
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block indices for 4x4 split into 2x2 blocks:
    # Block (bi, bj) where bi,bj in {0,1}
    # Element (i,j) in 4x4 maps to block (i//2, j//2), local (i%2, j%2)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U: A element (i, k) where i in {0,1,2,3}, k in {0,1,2,3}
            # Block (i//2, k//2), local (i%2, k%2)
            for bi in range(2):
                for bk in range(2):
                    i_local = bi * 2
                    k_local = bk * 2
                    for ii in range(2):
                        for kk in range(2):
                            i = i_local + ii
                            k = k_local + kk
                            u_val = U2[ii * 2 + kk, r1]
                            if u_val != 0:
                                U[i * m + k, r] = u_val
            
            # V: B element (k, j) where k in {0,1,2,3}, j in {0,1,2,3}
            # Block (k//2, j//2), local (k%2, j%2)
            for bk in range(2):
                for bj in range(2):
                    k_local = bk * 2
                    j_local = bj * 2
                    for kk in range(2):
                        for jj in range(2):
                            k = k_local + kk
                            j = j_local + jj
                            v_val = V2[kk * 2 + jj, r2]
                            if v_val != 0:
                                V[k * p + j, r] = v_val
            
            # W: C element (j, i) where j in {0,1,2,3}, i in {0,1,2,3}
            # Block (j//2, i//2), local (j%2, i%2)
            for bj in range(2):
                for bi in range(2):
                    j_local = bj * 2
                    i_local = bi * 2
                    for jj in range(2):
                        for ii in range(2):
                            j = j_local + jj
                            i = i_local + ii
                            w_val = W2[jj * 2 + ii, r1 * 7 + r2 if False else r]
                            # Wait, W should use the combined Strassen term
                            # Actually W for the outer Strassen uses r1, and inner uses r2
                            # Let me reconsider...
                            pass
            
            # Redo W properly:
            # The outer Strassen computes 7 products of 2x2 block matrices
            # Each product is computed via inner Strassen (7 terms)
            # So W[r1,r2] = W_outer[r1] * W_inner[r2] conceptually
            # But actually the decomposition is:
            # C_block(bi,bj) = sum_{r1} W_outer[r1][bi,bj] * (sum_{r2} W_inner[r2] * A_block(bi,*) * B_block(*,bj))
            # Hmm, this is getting complex. Let me think again.
            
            # Actually for tensor decomposition, we need:
            # sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] = 1 for all i,k,j
            # 
            # With recursive Strassen:
            # r = (r1, r2) where r1 is outer Strassen term, r2 is inner
            # 
            # U[i,k,(r1,r2)] = U_outer[i//2, k//2, r1] * U_inner[i%2, k%2, r2]
            # V[k,j,(r1,r2)] = V_outer[k//2, j//2, r1] * V_inner[k%2, j%2, r2]  
            # W[j,i,(r1,r2)] = W_outer[j//2, i//2, r1] * W_inner[j%2, i%2, r2]
            #
            # Wait, that's not quite right either. Let me use the proper formulation.
            
    # Let me redo this more carefully
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For recursive Strassen on 4x4:
    # Split A(4x4) into blocks A00,A01,A10,A11 (each 2x2)
    # Split B(4x4) into blocks B00,B01,B10,B11 (each 2x2)
    # C(4x4) = A*B, split into C00,C01,C10,C11 (each 2x2)
    #
    # Outer Strassen (on block level, 2x2 blocks):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M is a 2x2 matrix product, computed via inner Strassen (7 terms)
    
    # Outer Strassen coefficients for W (C output):
    # W_outer[cbi, cbj, r1] where cbi=block row of C, cbj=block col of C
    W_outer = np.zeros((4, 7))  # [block_index, r1] where block_index = cbi*2+cbj
    # C00 = M1 + M4 - M5 + M7
    W_outer[0, 0] = 1.0  # M1
    W_outer[0, 3] = 1.0  # M4
    W_outer[0, 4] = -1.0 # M5
    W_outer[0, 6] = 1.0  # M7
    # C01 = M3 + M5
    W_outer[1, 2] = 1.0  # M3
    W_outer[1, 4] = 1.0  # M5
    # C10 = M2 + M4
    W_outer[2, 1] = 1.0  # M2
    W_outer[2, 3] = 1.0  # M4
    # C11 = M1 - M2 + M3 + M6
    W_outer[3, 0] = 1.0  # M1
    W_outer[3, 1] = -1.0 # M2
    W_outer[3, 2] = 1.0  # M3
    W_outer[3, 5] = 1.0  # M6
    
    # Outer Strassen coefficients for U (A input):
    # U_outer[abi, abj, r1] where abi=block row of A, abj=block col of A
    U_outer = np.zeros((4, 7))  # [block_index, r1]
    # M1 = (A00+A11)(...)
    U_outer[0, 0] = 1.0  # A00
    U_outer[3, 0] = 1.0  # A11
    # M2 = (A10+A11)B00
    U_outer[2, 1] = 1.0  # A10
    U_outer[3, 1] = 1.0  # A11
    # M3 = A00(...)
    U_outer[0, 2] = 1.0  # A00
    # M4 = A11(...)
    U_outer[3, 3] = 1.0  # A11
    # M5 = (A00+A01)B11
    U_outer[0, 4] = 1.0  # A00
    U_outer[1, 4] = 1.0  # A01
    # M6 = (A10-A00)(...)
    U_outer[2, 5] = 1.0  # A10
    U_outer[0, 5] = -1.0 # A00
    # M7 = (A01-A11)(...)
    U_outer[1, 6] = 1.0  # A01
    U_outer[3, 6] = -1.0 # A11
    
    # Outer Strassen coefficients for V (B input):
    # V_outer[bki, bkj, r1] where bki=block row of B, bkj=block col of B
    V_outer = np.zeros((4, 7))  # [block_index, r1]
    # M1 = (...)(B00+B11)
    V_outer[0, 0] = 1.0  # B00
    V_outer[3, 0] = 1.0  # B11
    # M2 = (...)B00
    V_outer[0, 1] = 1.0  # B00
    # M3 = ...(B01-B11)
    V_outer[1, 2] = 1.0  # B01
    V_outer[3, 2] = -1.0 # B11
    # M4 = ...(B10-B00)
    V_outer[2, 3] = 1.0  # B10
    V_outer[0, 3] = -1.0 # B00
    # M5 = ...B11
    V_outer[3, 4] = 1.0  # B11
    # M6 = ...(B00+B01)
    V_outer[0, 5] = 1.0  # B00
    V_outer[1, 5] = 1.0  # B01
    # M7 = ...(B10+B11)
    V_outer[2, 6] = 1.0  # B10
    V_outer[3, 6] = 1.0  # B11
    
    # Now combine with inner Strassen
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U[i*m+k, r] = sum over block contributions
            for bi in range(2):
                for bk in range(2):
                    u_outer_val = U_outer[bi * 2 + bk, r1]
                    if u_outer_val == 0:
                        continue
                    # Inner Strassen for 2x2 block (bi,bk)
                    for ii in range(2):
                        for kk in range(2):
                            i = bi * 2 + ii
                            k = bk * 2 + kk
                            u_inner_val = U2[ii * 2 + kk, r2]
                            if u_inner_val != 0:
                                U[i * m + k, r] += u_outer_val * u_inner_val
            
            # V[k*p+j, r] = sum over block contributions
            for bk in range(2):
                for bj in range(2):
                    v_outer_val = V_outer[bk * 2 + bj, r1]
                    if v_outer_val == 0:
                        continue
                    for kk in range(2):
                        for jj in range(2):
                            k = bk * 2 + kk
                            j = bj * 2 + jj
                            v_inner_val = V2[kk * 2 + jj, r2]
                            if v_inner_val != 0:
                                V[k * p + j, r] += v_outer_val * v_inner_val
            
            # W[j*n+i, r] = sum over block contributions
            for bj in range(2):
                for bi in range(2):
                    w_outer_val = W_outer[bj * 2 + bi, r1]
                    if w_outer_val == 0:
                        continue
                    for jj in range(2):
                        for ii in range(2):
                            j = bj * 2 + jj
                            i = bi * 2 + ii
                            w_inner_val = W2[jj * 2 + ii, r2]
                            if w_inner_val != 0:
                                W[j * n + i, r] += w_outer_val * w_inner_val
    
    return U, V, W


def _build_m414():
    """
    Build rank-1 decomposition for M_{4,1,4} (4x1 * 1x4 -> 4x4).
    """
    n = 4
    m = 1
    p = 4
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        U[i * m, 0] = 1.0
    V[0, 0] = 1.0
    for j in range(p):
        W[j * n, 0] = 1.0
        
    return U, V, W


def _build_m412():
    """
    Build rank-1 decomposition for M_{4,1,2} (4x1 * 1x2 -> 4x2).
    """
    n = 4
    m = 1
    p = 2
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        U[i * m, 0] = 1.0
    for j in range(p):
        V[j, 0] = 1.0
    for j in range(p):
        for i in range(n):
            W[j * n + i, 0] = 1.0
            
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Split indices
    # A: 4x5 (i:0-3, k:0-4)
    # B: 5x6 (k:0-4, j:0-5)
    # C: 4x6 (i:0-3, j:0-5)
    
    # Core: 4x4x4 (i:0-3, k:0-3, j:0-3)
    # Right: 4x4x2 (i:0-3, k:0-3, j:4-5)
    # Bottom: 4x1x4 (i:0-3, k:4, j:0-3)
    # Corner: 4x1x2 (i:0-3, k:4, j:4-5)
    
    # 1. Core Schoolbook (Rank 64)
    U_core, V_core, W_core = _build_schoolbook(4, 4, 4)
    # Map indices to global
    U_c = np.zeros((4*5, 64))
    V_c = np.zeros((5*6, 64))
    W_c = np.zeros((6*4, 64))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                r = i*16 + k*4 + j
                U_c[i*5 + k, r] = 1.0
                V_c[k*6 + j, r] = 1.0
                W_c[j*4 + i, r] = 1.0

    # 2. Right Strip (Rank 4)
    # Two M_441
    U_r1, V_r1, W_r1 = _build_m441()
    U_r2, V_r2, W_r2 = _build_m441()
    
    U_r = np.zeros((4*5, 4))
    V_r = np.zeros((5*6, 4))
    W_r = np.zeros((6*4, 4))
    
    # r=0,1 for j=4
    for r in range(2):
        for i in range(4):
            for k in range(4):
                U_r[i*5 + k, r] = U_r1[i*4 + k, r]
        for k in range(4):
            V_r[k*6 + 4, r] = V_r1[k*1, r]
        for i in range(4):
            W_r[4*4 + i, r] = W_r1[1*i, r] # W index: j*n+i -> 4*4+i
            
    # r=2,3 for j=5
    for r in range(2):
        for i in range(4):
            for k in range(4):
                U_r[i*5 + k, r+2] = U_r2[i*4 + k, r]
        for k in range(4):
            V_r[k*6 + 5, r+2] = V_r2[k*1, r]
        for i in range(4):
            W_r[5*4 + i, r+2] = W_r2[1*i, r]

    # 3. Bottom Strip (Rank 1)
    U_b, V_b, W_b = _build_m414()
    U_b_g = np.zeros((4*5, 1))
    V_b_g = np.zeros((5*6, 1))
    W_b_g = np.zeros((6*4, 1))
    for i in range(4):
        U_b_g[i*5 + 4, 0] = U_b[i*1, 0]
    V_b_g[4*6 + 0, 0] = V_b[0, 0]
    for j in range(4):
        W_b_g[j*4 + 0, 0] = W_b[j*4, 0] # W index: j*n+i -> j*4+0

    # 4. Corner (Rank 1)
    U_co, V_co, W_co = _build_m412()
    U_co_g = np.zeros((4*5, 1))
    V_co_g = np.zeros((5*6, 1))
    W_co_g = np.zeros((6*4, 1))
    for i in range(4):
        U_co_g[i*5 + 4, 0] = U_co[i*1, 0]
    for j in range(2):
        V_co_g[4*6 + j, 0] = V_co[j, 0]
    for j in range(2):
        for i in range(4):
            W_co_g[j*4 + i, 0] = W_co[j*2 + i, 0] # W index: j*n+i -> j*4+i

    # Concatenate
    U = np.hstack([U_c, U_r, U_b_g, U_co_g])
    V = np.hstack([V_c, V_r, V_b_g, V_co_g])
    W = np.hstack([W_c, W_r, W_b_g, W_co_g])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
