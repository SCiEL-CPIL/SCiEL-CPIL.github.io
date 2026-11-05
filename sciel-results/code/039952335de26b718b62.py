"""Improved rank-minimal exact decomposition for 3x4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U,V,W) for 2x2x2 matrix mult, rank 7, with W index [k*2+i, r]."""
    # Standard Strassen: M1..M7
    # A = [[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1=(a+b)(e+h) ; M2=(c+d)e ; M3=a(f-h) ; M4=d(g-e)
    # M5=(a+c)(e+h) ; M6=(b-d)(g+h) ; M7=(a-c)(f+h)
    # C00=M1+M4-M5+M7 ; C01=M3+M5 ; C10=M2+M5 ; C11=M1-M2+M3+M6
    # U indexed by [i*2+j] (row i of A, col j of A), V by [j*2+k] (row j of B, col k of B),
    # W by [k*2+i] (col k of C, row i of C).
    U = []
    V = []
    W = []
    # each factor: (4,7)
    U = np.zeros((4,7)); V = np.zeros((4,7)); W = np.zeros((4,7))
    # a=U[0], b=U[1], c=U[2], d=U[3]
    # e=V[0], f=V[1], g=V[2], h=V[3]
    # M1=(a+b)(e+h): U col: a=1,b=1; V col: e=1,h=1; W: C00+=1, C11+=1
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(c+d)e: U: c=1,d=1; V: e=1; W: C10+=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3=a(f-h): U: a=1; V: f=1,h=-1; W: C01+=1, C11+=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4=d(g-e): U: d=1; V: g=1,e=-1; W: C00+=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1
    # M5=(a+c)(e+h): U: a=1,c=1; V: e=1,h=1; W: C00-=1, C01+=1, C10+=1
    U[0,4]=1; U[2,4]=1; V[0,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1; W[2,4]=1
    # M6=(b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C11+=1
    U[1,5]=1; U[3,5]=-1; V[2,5]=1; V[3,5]=1; W[3,5]=1
    # M7=(a-c)(f+h): U: a=1,c=-1; V: f=1,h=1; W: C00+=1
    U[0,6]=1; U[2,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_block_decomposition(n, m, p, deadline):
    """Build decomposition using Strassen for 2x2x2 sub-blocks where possible,
    schoolbook for the rest. Returns (U,V,W) as lists or None."""
    SU, SV, SW = _strassen_222()
    U = []
    V = []
    W = []
    # We'll accumulate columns.
    # Partition:
    #   rows of A (n=3): row_blocks = [[0,1],[2]]
    #   cols of A / rows of B (m=4): col_blocks = [[0,1],[2,3]]
    #   cols of B (p=8): p_blocks = [[0,1],[2,3],[4,5],[6,7]]
    row_blocks = [[0,1],[2]]
    col_blocks = [[0,1],[2,3]]
    p_blocks = [[0,1],[2,3],[4,5],[6,7]]

    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U.append([0.0]*1); V.append([0.0]*1); W.append([0.0]*1)
        U[-1][0]=u_val; V[-1][0]=v_val; W[-1][0]=w_val
        # store index mapping
        return len(U)-1

    # We'll build full matrices with a running rank.
    # Use numpy for accumulation for speed, then convert.
    # Max possible rank: n*m*p = 96
    max_rank = n*m*p
    Umat = np.zeros((n*m, max_rank))
    Vmat = np.zeros((m*p, max_rank))
    Wmat = np.zeros((p*n, max_rank))
    col = 0

    for ri, rblock in enumerate(row_blocks):
        for cj, cblock in enumerate(col_blocks):
            for pk, pblock in enumerate(p_blocks):
                # sub-tensor: A[rblock, cblock] (size ra x ca) times B[cblock, pblock] (ca x pa)
                # = C[rblock, pblock] (ra x pa)
                ra = len(rblock); ca = len(cblock); pa = len(pblock)
                if ra==2 and ca==2 and pa==2:
                    # Strassen rank 7
                    for t in range(7):
                        # U factor: maps (i,j) in rblock x cblock -> value
                        for li, i in enumerate(rblock):
                            for lj, j in enumerate(cblock):
                                Umat[i*m+j, col] = SU[li*2+lj, t]
                        for lj, j in enumerate(cblock):
                            for lk, k in enumerate(pblock):
                                Vmat[j*p+k, col] = SV[lj*2+lk, t]
                        for lk, k in enumerate(pblock):
                            for li, i in enumerate(rblock):
                                Wmat[k*n+i, col] = SW[lk*2+li, t]
                        col += 1
                        if time.time() > deadline:
                            return None
                else:
                    # schoolbook for this sub-block: ra*ca*pa terms
                    for li, i in enumerate(rblock):
                        for lj, j in enumerate(cblock):
                            for lk, k in enumerate(pblock):
                                Umat[i*m+j, col] = 1.0
                                Vmat[j*p+k, col] = 1.0
                                Wmat[k*n+i, col] = 1.0
                                col += 1
                                if time.time() > deadline:
                                    return None
    # Trim to col
    Umat = Umat[:, :col]
    Vmat = Vmat[:, :col]
    Wmat = Wmat[:, :col]
    # Verify identity: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    # Check: (U @ diag? ) Actually C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]
    # and must equal sum_c W[k*n+i,c].
    # Compute both sides.
    A_idx = np.arange(n*m)
    B_idx = np.arange(m*p)
    # Left side: for each (i,k), sum_j U[i*m+j,:] dot V[j*p+k,:]
    # Build via einsum
    # U shape (n*m, col) -> reshape (n, m, col)
    U3 = Umat.reshape(n, m, col)
    V3 = Vmat.reshape(m, p, col)
    # LHS[i,k] = sum_j sum_c U3[i,j,c]*V3[j,k,c]
    LHS = np.einsum('ijc,jkc->ik', U3, V3)
    W3 = Wmat.reshape(p, n, col)
    RHS = np.einsum('kic->ki', W3)  # sum_c W[k*n+i,c] -> (p,n) -> transpose (n,p)? 
    # W3[k,i,c]; sum over c gives (p,n) = W3.sum(axis=2) shape (p,n); we need (n,p)
    RHS = W3.sum(axis=2).T  # (n,p)
    if not np.allclose(LHS, RHS, atol=1e-9):
        return None
    # Convert to lists of ints where possible
    def to_list(M):
        out = []
        for row in M:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(float(v))
            out.append(r)
        return out
    return {"U": to_list(Umat), "V": to_list(Vmat), "W": to_list(Wmat), "rank": col}

def _schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": rank}

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best = None
    # Try block Strassen decomposition
    res = _build_block_decomposition(n, m, p, deadline)
    if res is not None:
        best = res
    # Fallback to schoolbook if needed
    if best is None:
        best = _schoolbook(n, m, p)
    # Strip rank key
    out = {"U": best["U"], "V": best["V"], "W": best["W"]}
    return out
# EVOLVE-BLOCK-END
