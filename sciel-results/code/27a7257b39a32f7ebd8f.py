"""Optimized block decomposition for 3x5x6 matrix multiplication."""
import itertools

DIMENSIONS = [3, 5, 6]

def _strassen_2x2_2x2():
    """Return U,V,W for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen: 7 multiplications
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2x2: n=2,m=2,p=2
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_block(nb, mb, pb):
    """Return U,V,W for nb x mb x pb schoolbook decomposition."""
    rank = nb * mb * pb
    U = [[0]*rank for _ in range(nb*mb)]
    V = [[0]*rank for _ in range(mb*pb)]
    W = [[0]*rank for _ in range(pb*nb)]
    col = 0
    for i in range(nb):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*nb+i][col] = 1
                col += 1
    return U, V, W

def _block_rank(nb, mb, pb):
    """Return the rank of the best known decomposition for nb x mb x pb."""
    if nb == 2 and mb == 2 and pb == 2:
        return 7
    return nb * mb * pb

def _get_block_decomp(nb, mb, pb):
    """Get the best decomposition for a block."""
    if nb == 2 and mb == 2 and pb == 2:
        return _strassen_2x2_2x2()
    return _schoolbook_block(nb, mb, pb)

def _compute_total_rank(n_splits, m_splits, p_splits):
    """Compute total rank for given splits."""
    total = 0
    for nb in n_splits:
        for mb in m_splits:
            for pb in p_splits:
                total += _block_rank(nb, mb, pb)
    return total

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Search over all possible block splits
    best_rank = float('inf')
    best_splits = None
    
    # Generate all possible splits for n=3, m=5, p=6
    def all_splits(total):
        """Generate all compositions of total."""
        if total == 1:
            yield [1]
            return
        for first in range(1, total):
            for rest in all_splits(total - first):
                yield [first] + rest
    
    n_splits_list = list(all_splits(3))
    m_splits_list = list(all_splits(5))
    p_splits_list = list(all_splits(6))
    
    for ns in n_splits_list:
        for ms in m_splits_list:
            for ps in p_splits_list:
                r = _compute_total_rank(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_splits = (ns, ms, ps)
    
    n_splits, m_splits, p_splits = best_splits
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for ni, nb in enumerate(n_splits):
        for mi, mb in enumerate(m_splits):
            for pi, pb in enumerate(p_splits):
                ub, vb, wb = _get_block_decomp(nb, mb, pb)
                U_blocks.append((ni, nb, ub))
                V_blocks.append((mi, mb, vb))
                W_blocks.append((pi, pb, wb))
    
    # Now assemble full matrices
    total_rank = sum(len(ub[0]) for _, _, ub in U_blocks)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for (ni, nb, ub), (mi, mb, vb), (pi, pb, wb) in zip(U_blocks, V_blocks, W_blocks):
        r = len(ub[0])
        n_off = sum(n_splits[:ni])
        m_off = sum(m_splits[:mi])
        p_off = sum(p_splits[:pi])
        
        for ri in range(nb):
            for rj in range(mb):
                for c in range(r):
                    U[(n_off+ri)*m + (m_off+rj)][col_offset+c] = ub[ri*mb+rj][c]
        
        for rj in range(mb):
            for rk in range(pb):
                for c in range(r):
                    V[(m_off+rj)*p + (p_off+rk)][col_offset+c] = vb[rj*pb+rk][c]
        
        for rk in range(pb):
            for ri in range(nb):
                for c in range(r):
                    W[(p_off+rk)*n + (n_off+ri)][col_offset+c] = wb[rk*nb+ri][c]
        
        col_offset += r
    
    return {"U": U, "V": V, "W": W}
