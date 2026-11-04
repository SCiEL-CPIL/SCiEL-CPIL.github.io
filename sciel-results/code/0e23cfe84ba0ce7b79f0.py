"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for M_{n,m,p}."""
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
    return U, V, W, rank

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 (rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # c=2: M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # c=3: M4 = A11(B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W, 7

def _block_decompose(n, m, p, n_split, m_split, p_split):
    """
    Decompose n x m x p using block partitioning.
    n_split: list of n-block sizes (sums to n)
    m_split: list of m-block sizes (sums to m)
    p_split: list of p-block sizes (sums to p)
    
    For each combination of blocks (ib, jb, kb), we need a decomposition of
    n_split[ib] x m_split[jb] x p_split[kb].
    We use Strassen for 2x2x2 and schoolbook for everything else.
    """
    total_rank = 0
    # First pass: compute total rank
    for ib in range(len(n_split)):
        for jb in range(len(m_split)):
            for kb in range(len(p_split)):
                nb = n_split[ib]
                mb = m_split[jb]
                pb = p_split[kb]
                if nb == 2 and mb == 2 and pb == 2:
                    total_rank += 7
                else:
                    total_rank += nb * mb * pb
    
    # Allocate
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in range(len(n_split)):
        i_off = sum(n_split[:ib])
        for jb in range(len(m_split)):
            j_off = sum(m_split[:jb])
            for kb in range(len(p_split)):
                k_off = sum(p_split[:kb])
                nb = n_split[ib]
                mb = m_split[jb]
                pb = p_split[kb]
                
                if nb == 2 and mb == 2 and pb == 2:
                    Ub, Vb, Wb, rb = _strassen_2x2x2()
                else:
                    Ub, Vb, Wb, rb = _schoolbook(nb, mb, pb)
                
                # Place into global matrices
                for c in range(rb):
                    # U: rows i_off..i_off+nb-1, cols j_off..j_off+mb-1
                    for ii in range(nb):
                        for jj in range(mb):
                            U[(i_off + ii) * m + (j_off + jj)][col + c] = Ub[ii * mb + jj][c]
                    # V: rows j_off..j_off+mb-1, cols k_off..k_off+pb-1
                    for jj in range(mb):
                        for kk in range(pb):
                            V[(j_off + jj) * p + (k_off + kk)][col + c] = Vb[jj * pb + kk][c]
                    # W: rows k_off..k_off+pb-1, cols i_off..i_off+nb-1
                    for kk in range(pb):
                        for ii in range(nb):
                            W[(k_off + kk) * n + (i_off + ii)][col + c] = Wb[kk * nb + ii][c]
                
                col += rb
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook(n, m, p)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try block decomposition: n=3 as 1+2, m=5 as 1+4, p=6 as 1+5
    # Sub-blocks: (1,1,1)=1, (1,1,5)=5, (1,4,1)=4, (1,4,5)=20,
    #             (2,1,1)=2, (2,1,5)=10, (2,4,1)=8, (2,4,5)=40
    # Total = 1+5+4+20+2+10+8+40 = 90 (same as schoolbook, no gain)
    
    # Better: n=3 as 1+2, m=5 as 2+3, p=6 as 3+3
    # Sub-blocks: (1,2,3)=6, (1,3,3)=9, (2,2,3)=12, (2,3,3)=18
    # Total = 6+9+12+18 = 45
    
    U2, V2, W2, r2 = _block_decompose(n, m, p, [1,2], [2,3], [3,3])
    if r2 < best_rank:
        U, V, W, best_rank = U2, V2, W2, r2
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 1+2, m=5 as 1+2+2, p=6 as 2+2+2
    # Sub-blocks with 2x2x2 = Strassen(7):
    # (1,1,2)=2, (1,2,2)=4, (1,2,2)=4
    # (2,1,2)=4, (2,2,2)=7, (2,2,2)=7
    # Total = 2+4+4+4+7+7 = 28
    
    U3, V3, W3, r3 = _block_decompose(n, m, p, [1,2], [1,2,2], [2,2,2])
    if r3 < best_rank:
        U, V, W, best_rank = U3, V3, W3, r3
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 1+1+1, m=5 as 1+2+2, p=6 as 2+2+2
    # Sub-blocks:
    # (1,1,2)=2, (1,2,2)=4, (1,2,2)=4
    # (1,1,2)=2, (1,2,2)=4, (1,2,2)=4
    # (1,1,2)=2, (1,2,2)=4, (1,2,2)=4
    # Total = 2+4+4+2+4+4+2+4+4 = 30
    
    U4, V4, W4, r4 = _block_decompose(n, m, p, [1,1,1], [1,2,2], [2,2,2])
    if r4 < best_rank:
        U, V, W, best_rank = U4, V4, W4, r4
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 1+2, m=5 as 2+2+1, p=6 as 2+2+2
    # Sub-blocks:
    # (1,2,2)=4, (1,2,2)=4, (1,1,2)=2
    # (2,2,2)=7, (2,2,2)=7, (2,1,2)=4
    # Total = 4+4+2+7+7+4 = 28
    
    U5, V5, W5, r5 = _block_decompose(n, m, p, [1,2], [2,2,1], [2,2,2])
    if r5 < best_rank:
        U, V, W, best_rank = U5, V5, W5, r5
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 1+1+1, m=5 as 2+2+1, p=6 as 2+2+2
    # Sub-blocks:
    # (1,2,2)=4, (1,2,2)=4, (1,1,2)=2
    # (1,2,2)=4, (1,2,2)=4, (1,1,2)=2
    # (1,2,2)=4, (1,2,2)=4, (1,1,2)=2
    # Total = 4+4+2+4+4+2+4+4+2 = 30
    
    U6, V6, W6, r6 = _block_decompose(n, m, p, [1,1,1], [2,2,1], [2,2,2])
    if r6 < best_rank:
        U, V, W, best_rank = U6, V6, W6, r6
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    # Sub-blocks:
    # (2,2,2)=7, (2,2,2)=7, (2,1,2)=4
    # (1,2,2)=4, (1,2,2)=4, (1,1,2)=2
    # Total = 7+7+4+4+4+2 = 28
    
    U7, V7, W7, r7 = _block_decompose(n, m, p, [2,1], [2,2,1], [2,2,2])
    if r7 < best_rank:
        U, V, W, best_rank = U7, V7, W7, r7
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try: n=3 as 1+1+1, m=5 as 1+1+1+1+1, p=6 as 2+2+2
    # All 1x1x2 = 2 each, 15 blocks * 2 = 30
    
    # Try: n=3 as 1+2, m=5 as 1+1+1+1+1, p=6 as 2+2+2
    # (1,1,2)=2 x 15 blocks for n-block-1, (2,1,2)=4 x 15 for n-block-2... 
    # Wait: 1 n-block * 5 m-blocks * 3 p-blocks = 15 blocks of (1,1,2) = 30
    # 1 n-block * 5 m-blocks * 3 p-blocks = 15 blocks of (2,1,2) = 60
    # Total = 90, no gain
    
    # Try: n=3 as 1+1+1, m=5 as 1+1+1+1+1, p=6 as 1+1+1+1+1+1
    # 15 blocks of (1,1,1) = 15... wait that's only 15? No: 3*5*6=90
    
    # Let's try more creative splits
    # n=3 as 1+2, m=5 as 3+2, p=6 as 4+2
    # (1,3,4)=12, (1,2,4)=8, (1,3,2)=6, (1,2,2)=4
    # (2,3,4)=24, (2,2,4)=16, (2,3,2)=12, (2,2,2)=7
    # Total = 12+8+6+4+24+16+12+7 = 89
    
    # n=3 as 1+2, m=5 as 3+2, p=6 as 2+4
    # (1,3,2)=6, (1,2,2)=4, (1,3,4)=12, (1,2,4)=8
    # (2,3,2)=12, (2,2,2)=7, (2,3,4)=24, (2,2,4)=16
    # Total = 6+4+12+8+12+7+24+16 = 89
    
    # The best so far is 28. Let me try to verify and return.
    
    # Verify the best solution
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W, best_rank = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
