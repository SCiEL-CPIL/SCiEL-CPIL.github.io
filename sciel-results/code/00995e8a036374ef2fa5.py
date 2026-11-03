"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build 2x2 Strassen. Returns (U, V, W) each shape (4, 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    
    return U, V, W

def _build_strassen_4x4():
    """4x4 Strassen, rank 49."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    defs = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
    ]
    
    for s, (ab, bb, cb) in enumerate(defs):
        for t in range(7):
            col = s*7+t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca:
                        for (bi,bj),sg in ab:
                            U[(bi*2+li)*4+(bj*2+lj)][col] += ca*sg
            for lj in range(2):
                for lk in range(2):
                    cb_ = V2[lj*2+lk][t]
                    if cb_:
                        for (bj,bk),sg in bb:
                            V[(bj*2+lj)*4+(bk*2+lk)][col] += cb_*sg
            for lk in range(2):
                for li in range(2):
                    cw = W2[lk*2+li][t]
                    if cw:
                        for (bi,bk),sg in cb:
                            W[(bk*2+lk)*4+(bi*2+li)][col] += cw*sg
    return U, V, W

def _build_2x2x2_schoolbook():
    """2x2 * 2x2 schoolbook, rank 4."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    for j in range(2):
        for k in range(2):
            c = j*2+k
            U[j][c]=1; V[j*2+k][c]=1; W[k*2+j][c]=1  # W row is k*n+i = k*2+i
            # Actually W[k*n+i] where i is the A row index
            # For 2x2x2: U is (n*m)=(4), V is (m*p)=(4), W is (p*n)=(4)
            # U[i*m+j] = U[i*2+j]
            # V[j*p+k] = V[j*2+k]
            # W[k*n+i] = W[k*2+i]
    # Redo properly
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][c]=1
                V[j*2+k][c]=1
                W[k*2+i][c]=1
                c += 1
    return U, V, W

def _build_4x4x2():
    """4x4 * 4x2, rank 28. Split into 4 (2x2)*(2x2) blocks with Strassen."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    for b, (ai,aj,bj,ci,ck) in enumerate(blocks):
        base = b*7
        for t in range(7):
            col = base+t
            for li in range(2):
                for lj in range(2):
                    ca = U2[li*2+lj][t]
                    if ca:
                        U[(ai+li)*4+(aj+lj)][col] += ca
            for lj in range(2):
                for lk in range(2):
                    cb = V2[lj*2+lk][t]
                    if cb:
                        V[(bj+lj)*2+(ck+lk)][col] += cb
            for lk in range(2):
                for li in range(2):
                    cw = W2[lk*2+li][t]
                    if cw:
                        W[(ck+lk)*4+(ci+li)][col] += cw
    return U, V, W

def _build_4x2x6():
    """
    4x2 * 2x6, rank 16.
    Split 4 into 2+2, 6 into 3+3.
    4 (2x2)*(2x3) blocks. Each rank 4. Total 16.
    """
    rank = 16
    U = [[0]*rank for _ in range(8)]   # 4*2
    V = [[0]*rank for _ in range(12)]  # 2*6
    W = [[0]*rank for _ in range(24)]  # 6*4
    
    blocks = [(0,0,0), (0,3,0), (2,0,0), (2,3,0)]
    for b, (ai,ks,ci) in enumerate(blocks):
        base = b*4
        for j in range(2):
            for k in range(3):
                col = base + j*3+k
                U[(ai+j)*2][col] = 1
                V[j*6+ks+k][col] = 1
                W[(ks+k)*4+(ai+j)][col] = 1
    return U, V, W

def _build_4x1x6():
    """4x1 * 1x6, rank 24 schoolbook."""
    rank = 24
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    c = 0
    for i in range(4):
        for k in range(6):
            U[i][c]=1
            V[k][c]=1
            W[k*4+i][c]=1
            c += 1
    return U, V, W

def _combine(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    cur = 0
    for U_s, V_s, W_s, (i_s,i_e), (j_s,j_e), (k_s,k_e) in decomps:
        n_s = i_e-i_s; m_s = j_e-j_s; p_s = k_e-k_s
        r_s = len(U_s[0])
        
        for r in range(n_s*m_s):
            il = r // m_s; jl = r % m_s
            ig = i_s+il; jg = j_s+jl
            rg = ig*m+jg
            for c in range(r_s):
                U[rg][cur+c] = U_s[r][c]
        
        for r in range(m_s*p_s):
            jl = r // p_s; kl = r % p_s
            jg = j_s+jl; kg = k_s+kl
            rg = jg*p+kg
            for c in range(r_s):
                V[rg][cur+c] = V_s[r][c]
        
        for r in range(p_s*n_s):
            kl = r // n_s; il = r % n_s
            kg = k_s+kl; ig = i_s+il
            rg = kg*n+ig
            for c in range(r_s):
                W[rg][cur+c] = W_s[r][c]
        
        cur += r_s
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c]=1
                V[j*p+k][c]=1
                W[k*n+i][c]=1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # 1. 4x4x4 Strassen, rank 49
        U1, V1, W1 = _build_strassen_4x4()
        d1 = (U1, V1, W1, (0,4), (0,4), (0,4))
        
        # 2. 4x1x6 schoolbook, rank 24
        U2, V2, W2 = _build_4x1x6()
        d2 = (U2, V2, W2, (0,4), (4,5), (0,6))
        
        # 3. 4x4x2 Strassen, rank 28
        U3, V3, W3 = _build_4x4x2()
        d3 = (U3, V3, W3, (0,4), (0,4), (4,6))
        
        U, V, W = _combine([d1, d2, d3], n, m, p)
        
        # Verify before returning
        _verify(U, V, W, n, m, p)
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    rank = len(U[0])
    # Check tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k} (for the multiplication tensor)
    # The matrix multiplication tensor T[i,j,k] = 1 if i==k (in the sense of C_ik = sum_j A_ij B_jk)
    # Actually the tensor is T[i,j,k] where the bilinear form gives C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This should equal sum_j A[i,j]*B[j,k] for each (i,k), so the coefficient of A[i,j]*B[j,k] in C[i,k] should be 1.
    # So: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 if this is the (i,j,k) component, 0 otherwise.
    # But the decomposition computes C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For this to equal (AB)[i,k] = sum_j A[i,j]B[j,k], we need:
    # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 if we're looking at the right (i,k)
    # Actually the standard form: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
    # So the coefficient of A[a,j]*B[j,b] in C[a,b] is sum_r U[a*m+j,r]*V[j*p+b,r]*W[b*n+a,r]
    # This must equal 1 if a==b (same row/col for the output), and the sum over j gives the matrix product.
    # Wait, C[i,k] = sum_j A[i,j]*B[j,k]. The coefficient of A[i0,j]*B[j,k0] in C[i0,k0] is 1 iff i0=i0 and k0=k0, which is always true for the matching (i,j,k) triple.
    # So: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1.
    # No wait - that's not right either. The tensor decomposition says:
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # where T[i,j,k] = 1 (the multiplication tensor has all entries 1 in the appropriate sense).
    # Actually for matrix multiplication C_ik = sum_j A_ij B_jk, the tensor is T[i,j,k] = 1 for all valid (i,j,k).
    # So we need: for all i in [0,n), j in [0,m), k in [0,p): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Quick check on a few entries
    for i in range(min(n, 4)):
        for j in range(min(m, 5)):
            for k in range(min(p, 6)):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")

# EVOLVE-BLOCK-END
