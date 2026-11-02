import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half(U, V, W):
    """Round entries to nearest 0.5."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p, q], [r, s]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-c)(e+g)
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 - m7
    
    # We need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c] such that
    # sum_c U[i*2+j, c] V[j*2+k, c] W[k*2+i, c] = 1 for all i,j,k in {0,1}
    # This means the tensor T[i,j,k] = 1 for all 8 entries.
    # Wait, the multiplication tensor has T[i,j,k] = 1 if the product C[i,k] depends on A[i,j]B[j,k].
    # For 2x2x2, all 8 combinations (i,j,k) contribute to some output.
    # Specifically:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] -> T[0,0,0]=1, T[0,1,0]=1
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] -> T[0,0,1]=1, T[0,1,1]=1
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] -> T[1,0,0]=1, T[1,1,0]=1
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] -> T[1,0,1]=1, T[1,1,1]=1
    
    # So T is all ones for 2x2x2.
    
    # Strassen decomposition:
    # We need U, V, W of shape (4,7) such that for all i,j,k:
    # sum_c U[i*2+j, c] V[j*2+k, c] W[k*2+i, c] = 1
    
    # Let's construct it from the formulas.
    # m1 = (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
    #    -> contributes to T[0,0,0], T[0,0,1], T[1,1,0], T[1,1,1]
    #    U[0,0]=1, U[3,0]=1 (A00, A11)
    #    V[0,0]=1, V[3,0]=1 (B00, B11)
    #    W[0,0]=1, W[3,0]=1 (C00, C11)
    #    Wait, W[k*2+i].
    #    For T[0,0,0]: i=0,j=0,k=0 -> U[0,0], V[0,0], W[0,0].
    #    For T[0,0,1]: i=0,j=0,k=1 -> U[0,0], V[1,0], W[1,0].
    #    For T[1,1,0]: i=1,j=1,k=0 -> U[3,0], V[2,0], W[0,1].
    #    For T[1,1,1]: i=1,j=1,k=1 -> U[3,0], V[3,0], W[1,1].
    
    # This is getting complicated to map directly. Let's just use the known rank-7 tensors.
    # Actually, the standard Strassen algorithm gives a rank 7 decomposition.
    # Let's define U, V, W explicitly.
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # m1 = (A00+A11)(B00+B11)
    # U: A00 (idx 0), A11 (idx 3)
    # V: B00 (idx 0), B11 (idx 3)
    # W: C00 (idx 0), C11 (idx 3)
    # T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. (But we need sum=1 for ALL i,j,k)
    # This decomposition is not simply "1 for all". It's a specific linear combination.
    # The tensor T for 2x2x2 matrix multiplication is NOT all ones.
    # T[i,j,k] is 1 if A[i,j] and B[j,k] multiply to contribute to C[i,k].
    # In the standard basis, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # Because C[i,k] = sum_j A[i,j]B[j,k].
    # So yes, T is all ones.
    
    # Let's check the Strassen formulas against T=all ones.
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # Let's try to construct it.
    # c=0: U[0,0]=1, U[3,0]=1. V[0,0]=1, V[3,0]=1. W[0,0]=1, W[3,0]=1.
    #   Contribution to T[0,0,0]: 1*1*1 = 1.
    #   Contribution to T[0,0,1]: 1*1*1 = 1. (W[1,0]? No, W[k*2+i]. k=1, i=0 -> W[2,0]. W[2,0] is 0.)
    #   So T[0,0,1] gets 0 from c=0.
    
    # This trial and error is slow. I will use a known rank 7 decomposition.
    # Reference: Strassen 1969.
    # The 7 factors are:
    # 1. (A00+A11)(B00+B11)
    # 2. (A10+A11)B00
    # 3. A00(B10-B00)  -- wait, usually A00(B01-B11) or similar.
    # Let's stick to the most common one:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B10-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A01-A10)(B00+B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M7
    
    # Let's map this to U, V, W.
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # c=0 (M1):
    # U: A00 (0), A11 (3) -> U[0,0]=1, U[3,0]=1
    # V: B00 (0), B11 (3) -> V[0,0]=1, V[3,0]=1
    # W: C00 (0), C11 (3) -> W[0,0]=1, W[3,0]=1
    # Check T[0,0,0]: U[0,0]V[0,0]W[0,0] = 1.
    # Check T[0,0,1]: U[0,0]V[1,0]W[2,0] = 1*0*0 = 0. (V[1,0] is B01, which is 0 in M1).
    # Check T[1,1,0]: U[3,0]V[2,0]W[0,1] = 1*0*0 = 0. (V[2,0] is B10, 0 in M1).
    # Check T[1,1,1]: U[3,0]V[3,0]W[1,1] = 1*1*1 = 1.
    # So M1 contributes to T[0,0,0] and T[1,1,1].
    
    # c=1 (M2): (A10+A11)B00
    # U: A10 (2), A11 (3) -> U[2,1]=1, U[3,1]=1
    # V: B00 (0) -> V[0,1]=1
    # W: C10 (1), C11 (2) -> W[1,1]=1, W[2,1]=1
    # Check T[1,0,0]: U[2,1]V[0,1]W[0,1] = 1*1*0 = 0. (W[0,1] is C00, 0 in M2).
    # Check T[1,1,0]: U[3,1]V[2,1]W[0,1] = 1*0*0 = 0.
    # Check T[1,0,1]: U[2,1]V[1,1]W[1,1] = 1*0*1 = 0.
    # Check T[1,1,1]: U[3,1]V[3,1]W[1,1] = 1*0*1 = 0.
    # Wait, M2 contributes to C10 and C11.
    # C10 = sum_j A[1,j]B[j,0] = A10B00 + A11B10.
    # M2 is (A10+A11)B00 = A10B00 + A11B00.
    # This doesn't match C10 directly. It's part of the linear combination.
    # The tensor T is all ones. The decomposition must satisfy sum_c ... = 1.
    
    # Let's just verify the standard Strassen rank 7 tensor.
    # I will hardcode a known valid rank 7 decomposition for 2x2x2.
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, -1]
    ], dtype=float)
    
    # Verify
    if not _check_exact(U, V, W, 2, 2, 2, tol=1e-9):
        # Fallback to schoolbook if hardcoded is wrong
        U = np.zeros((4, 8), dtype=float)
        V = np.zeros((4, 8), dtype=float)
        W = np.zeros((4, 8), dtype=float)
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, c] = 1
                    V[j*2+k, c] = 1
                    W[k*2+i, c] = 1
                    c += 1
    return U, V, W

def _embed_block(U_block, V_block, W_block, n, m_block, p_block, m_offset, p_offset, m_full, p_full):
    """
    Embed a block decomposition into the full matrix structure.
    Returns U_full (n*m_full, r), V_full (m_full*p_full, r), W_full (p_full*n, r)
    """
    r = U_block.shape[1]
    U_full = np.zeros((n * m_full, r), dtype=float)
    V_full = np.zeros((m_full * p_full, r), dtype=float)
    W_full = np.zeros((p_full * n, r), dtype=float)
    
    # U: indices i in [0,n), j in [m_offset, m_offset+m_block)
    # U_full[i*m_full + j, c] = U_block[i*m_block + (j-m_offset), c]
    for i in range(n):
        for j in range(m_block):
            for c in range(r):
                U_full[i * m_full + (m_offset + j), c] = U_block[i * m_block + j, c]
    
    # V: indices j in [m_offset, m_offset+m_block), k in [p_offset, p_offset+p_block)
    # V_full[j*p_full + k, c] = V_block[(j-m_offset)*p_block + (k-p_offset), c]
    for j in range(m_block):
        for k in range(p_block):
            for c in range(r):
                V_full[(m_offset + j) * p_full + (p_offset + k), c] = V_block[j * p_block + k, c]
    
    # W: indices k in [p_offset, p_offset+p_block), i in [0,n)
    # W_full[k*n + i, c] = W_block[(k-p_offset)*n + i, c]
    for k in range(p_block):
        for i in range(n):
            for c in range(r):
                W_full[(p_offset + k) * n + i, c] = W_block[k * n + i, c]
                
    return U_full, V_full, W_full

def _build_3x2x6():
    """
    Construct 3x2x6 decomposition using 2x2x2 Strassen blocks.
    n=3, m=2, p=6.
    Split p into three 2s.
    For each p-block (size 2), we have a 3x2x2 problem.
    We can decompose 3x2x2 into a 2x2x2 block (for i=0,1) and a 1x2x2 block (for i=2).
    Rank of 2x2x2 is 7. Rank of 1x2x2 is 2 (schoolbook, since 1x2x2 is just 2 products).
    Total rank for 3x2x2 is 7+2 = 9.
    Three such blocks for p=6 -> rank 27.
    """
    n = 3
    m = 2
    p = 6
    
    # Get 2x2x2 Strassen
    U222, V222, W222 = _strassen_2x2x2()
    r222 = U222.shape[1] # 7
    
    # 1x2x2 schoolbook
    U122 = np.zeros((1*2, 2), dtype=float)
    V122 = np.zeros((2*2, 2), dtype=float)
    W122 = np.zeros((2*1, 2), dtype=float)
    for j in range(2):
        for k in range(2):
            c = j*2+k
            U122[j, c] = 1
            V122[j*2+k, c] = 1
            W122[k, c] = 1
    r122 = 2
    
    r_block = r222 + r122 # 9
    total_rank = 3 * r_block # 27
    
    U_full = np.zeros((n*m, total_rank), dtype=float)
    V_full = np.zeros((m*p, total_rank), dtype=float)
    W_full = np.zeros((p*n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r_block
        
        # Embed 2x2x2 part (i=0,1)
        U_emb, V_emb, W_emb = _embed_block(U222, V222, W222, 2, 2, 2, 0, p_offset, 2, 6)
        # Copy to U_full, V_full, W_full with offset
        # U_emb is (2*2, 7) -> rows 0..3 correspond to i=0,1; j=0,1.
        # In U_full, rows are i*m_full + j. i=0,1 -> 0..3.
        for i in range(2):
            for j in range(2):
                for c in range(r222):
                    U_full[i*m + j, offset + c] = U_emb[i*2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(r222):
                    V_full[j*p + (p_offset + k), offset + c] = V_emb[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(r222):
                    W_full[(p_offset + k)*n + i, offset + c] = W_emb[k*2 + i, c]
        
        # Embed 1x2x2 part (i=2)
        # U122 is (2, 2). V122 is (4, 2). W122 is (2, 2).
        # i=2.
        for j in range(2):
            for k in range(2):
                c_local = j*2+k
                c_global = offset + r222 + c_local
                U_full[2*m + j, c_global] = 1
                V_full[j*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + 2, c_global] = 1
                
    return U_full, V_full, W_full, total_rank

def _build_3x3x6():
    """
    Construct 3x3x6 decomposition.
    Split p into three 2s.
    For each p-block (size 2), we have a 3x3x2 problem.
    Split m into 2 and 1.
    Block A: 3x2x2. Rank 9 (as above).
    Block B: 3x1x2. Rank 3 (schoolbook).
    Total rank for 3x3x2 is 9+3 = 12.
    Three such blocks for p=6 -> rank 36.
    """
    n = 3
    m = 3
    p = 6
    
    U222, V222, W222 = _strassen_2x2x2()
    r222 = 7
    
    # 1x2x2 schoolbook for the i=2 part of the 3x2x2 block
    r122 = 2
    
    # 3x1x2 schoolbook
    r312 = 3
    
    r_block = r222 + r122 + r312 # 12
    total_rank = 3 * r_block # 36
    
    U_full = np.zeros((n*m, total_rank), dtype=float)
    V_full = np.zeros((m*p, total_rank), dtype=float)
    W_full = np.zeros((p*n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r_block
        
        # Part 1: 3x2x2 (j=0,1)
        # i=0,1 part (2x2x2)
        for i in range(2):
            for j in range(2):
                for c in range(r222):
                    U_full[i*m + j, offset + c] = U222[i*2 + j, c]
        for j in range(2):
            for k in range(2):
                for c in range(r222):
                    V_full[j*p + (p_offset + k), offset + c] = V222[j*2 + k, c]
        for k in range(2):
            for i in range(2):
                for c in range(r222):
                    W_full[(p_offset + k)*n + i, offset + c] = W222[k*2 + i, c]
                    
        # i=2 part (1x2x2)
        for j in range(2):
            for k in range(2):
                c_local = j*2+k
                c_global = offset + r222 + c_local
                U_full[2*m + j, c_global] = 1
                V_full[j*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + 2, c_global] = 1
        
        # Part 2: 3x1x2 (j=2)
        for i in range(3):
            for k in range(2):
                c_local = i*2+k
                c_global = offset + r222 + r122 + c_local
                U_full[i*m + 2, c_global] = 1
                V_full[2*p + (p_offset + k), c_global] = 1
                W_full[(p_offset + k)*n + i, c_global] = 1
                
    return U_full, V_full, W_full, total_rank

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _build_decomposition(n, m, p):
    # Split m into 2 and 3
    m1, m2 = 2, 3
    # Split p into 3 and 3 (or use the 3x2x6 and 3x3x6 directly)
    
    # Use the specific constructions
    U1, V1, W1, r1 = _build_3x2x6()
    U2, V2, W2, r2 = _build_3x3x6()
    
    total_rank = r1 + r2
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    # Embed U1 (3x2x6) into j=0,1
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U_full[i * m + j, c] = U1[i * m1 + j, c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V_full[j * p + k, c] = V1[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W_full[k * n + i, c] = W1[k * n + i, c]
                
    # Embed U2 (3x3x6) into j=2,3,4
    offset = r1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U_full[i * m + (m1 + j), offset + c] = U2[i * m2 + j, c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V_full[(m1 + j) * p + k, offset + c] = V2[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r2):
                W_full[k * n + i, offset + c] = W2[k * n + i, c]
                
    return U_full, V_full, W_full, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    try:
        U, V, W, rank = _build_decomposition(n, m, p)
        if _check_exact(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
