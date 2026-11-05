"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _strassen_222():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication using Strassen (rank 7).
    Indices: U[0:4, :] (n*m=4), V[0:4, :] (m*p=4), W[0:4, :] (p*n=4)
    Convention: W[k*n+i, c]
    """
    # Strassen formulas for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # p = M1+M4-M5+M7
    # q = M3+M5
    # r = M2+M4
    # s = M1-M2+M3+M6
    
    # U: maps A entries [a,b,c,d] (row-major: a=0,b=1,c=2,d=3) to Strassen inputs
    # V: maps B entries [e,f,g,h] (row-major: e=0,f=1,g=2,h=3) to Strassen inputs
    # W: maps C entries [p,q,r,s] (row-major: p=0,q=1,r=2,s=3) from Strassen outputs
    # Convention: U[i*m+j, c] where i=row, j=col of A. So A[0,0]=a is U[0], A[0,1]=b is U[1], A[1,0]=c is U[2], A[1,1]=d is U[3]
    # V[j*p+k, c] where j=row, k=col of B. B[0,0]=e is V[0], B[0,1]=f is V[1], B[1,0]=g is V[2], B[1,1]=h is V[3]
    # W[k*n+i, c] where k=col of C, i=row of C. C[0,0]=p is W[0*2+0]=W[0], C[0,1]=q is W[1*2+0]=W[2], C[1,0]=r is W[0*2+1]=W[1], C[1,1]=s is W[1*2+1]=W[3]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: p=1,s=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1  # p and s
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: r=1,s=-1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1  # r and s
    
    # M3 = a(f-g): U: a=1; V: f=1,g=-1; W: q=1,r=1
    U[0,2] = 1
    V[1,2] = 1; V[2,2] = -1
    W[2,2] = 1; W[1,2] = 1  # q and r
    
    # M4 = d(g-h): U: d=1; V: g=1,h=-1; W: p=1,r=1
    U[3,3] = 1
    V[2,3] = 1; V[3,3] = -1
    W[0,3] = 1; W[1,3] = 1  # p and r
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: p=-1,q=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1  # p and q
    
    # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: s=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # s
    
    # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: p=1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # p
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return U, V, W for n x m times m x p schoolbook multiplication.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _block_decompose(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build a block-structured decomposition.
    n_blocks, m_blocks, p_blocks: lists of block sizes summing to n, m, p.
    Uses Strassen for (2,2,2) blocks, schoolbook for others.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank).
    """
    total_rank = 0
    # Pre-compute all block decompositions
    block_data = []
    for ni in n_blocks:
        for mj in m_blocks:
            for pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2:
                    U_b, V_b, W_b = _strassen_222()
                else:
                    U_b, V_b, W_b = _schoolbook_block(ni, mj, pk)
                block_data.append((ni, mj, pk, U_b, V_b, W_b))
                total_rank += U_b.shape[1]
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    row_offset_n = 0  # offset in n dimension
    for idx, (ni, mj, pk, U_b, V_b, W_b) in enumerate(block_data):
        row_offset_m = 0
        for mi_idx, mj_size in enumerate(m_blocks):
            row_offset_p = 0
            for pi_idx, pk_size in enumerate(p_blocks):
                if mi_idx == m_blocks.index(mj) and pi_idx == p_blocks.index(pk) and n_blocks.index(ni) == idx // (len(m_blocks)*len(p_blocks)):
                    pass
                row_offset_p += pk_size
            row_offset_m += mj_size
        # This indexing is getting complicated. Let me redo it properly.
        break
    
    # Redo properly
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    n_off = 0
    for ni in n_blocks:
        m_off = 0
        for mj in m_blocks:
            p_off = 0
            for pk in p_blocks:
                # Find the block data for (ni, mj, pk)
                # We need to find the right block_data entry
                # Let's restructure: iterate in the same order
                pass
            p_off += pk
        m_off += mj
    n_off += ni
    
    # Simpler approach: build in order
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    n_off = 0
    for ni in n_blocks:
        m_off = 0
        for mj in m_blocks:
            p_off = 0
            for pk in p_blocks:
                if ni == 2 and mj == 2 and pk == 2:
                    U_b, V_b, W_b = _strassen_222()
                else:
                    U_b, V_b, W_b = _schoolbook_block(ni, mj, pk)
                
                # Place U_b into U at rows [n_off*m_off : (n_off+ni)*m_off + ... ]
                # U row index: (i_global * m + j_global) where i_global = n_off + i_local, j_global = m_off + j_local
                # U_b has shape (ni*mj, rank_b)
                # Row i_local*mj + j_local in U_b maps to (n_off + i_local)*m + (m_off + j_local) in U
                for i_l in range(ni):
                    for j_l in range(mj):
                        src_row = i_l * mj + j_l
                        dst_row = (n_off + i_l) * m + (m_off + j_l)
                        U[dst_row, col_offset:col_offset + U_b.shape[1]] = U_b[src_row, :]
                
                # V row index: (j_global * p + k_global) where j_global = m_off + j_local, k_global = p_off + k_local
                # V_b has shape (mj*pk, rank_b)
                for j_l in range(mj):
                    for k_l in range(pk):
                        src_row = j_l * pk + k_l
                        dst_row = (m_off + j_l) * p + (p_off + k_l)
                        V[dst_row, col_offset:col_offset + V_b.shape[1]] = V_b[src_row, :]
                
                # W row index: (k_global * n + i_global) where k_global = p_off + k_local, i_global = n_off + i_local
                # W_b has shape (pk*ni, rank_b)
                for k_l in range(pk):
                    for i_l in range(ni):
                        src_row = k_l * ni + i_l
                        dst_row = (p_off + k_l) * n + (n_off + i_l)
                        W[dst_row, col_offset:col_offset + W_b.shape[1]] = W_b[src_row, :]
                
                col_offset += U_b.shape[1]
            p_off += pk
        m_off += mj
    n_off += ni
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Use einsum for speed
    # T_reconstructed = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    if np.max(np.abs(T_rec - T_target)) > tol:
        return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Try different block decompositions
    # n=3: possible splits: [3], [2,1], [1,2], [1,1,1]
    # m=5: possible splits: [5], [2,2,1], [2,1,2], [1,2,2], [3,2], [2,3], [1,4], [4,1], [1,1,3], [3,1,1], etc.
    # p=5: same as m
    
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate candidate block splits
    def gen_splits(total, max_parts=3):
        """Generate all compositions of 'total' into at most max_parts positive integers."""
        results = []
        def helper(remaining, parts, current):
            if not parts:
                if remaining > 0:
                    results.append(current + [remaining])
                return
            for v in range(1, remaining + 1):
                helper(remaining - v, parts - 1, current + [v])
        helper(total, max_parts, [])
        return results
    
    n_splits = gen_splits(n, 3)
    m_splits = gen_splits(m, 3)
    p_splits = gen_splits(p, 3)
    
    candidates = []
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                # Count potential Strassen blocks
                strassen_count = 0
                for ni in ns:
                    for mj in ms:
                        for pk in ps:
                            if ni == 2 and mj == 2 and pk == 2:
                                strassen_count += 1
                # Estimate rank
                est_rank = 0
                for ni in ns:
                    for mj in ms:
                        for pk in ps:
                            if ni == 2 and mj == 2 and pk == 2:
                                est_rank += 7
                            else:
                                est_rank += ni * mj * pk
                candidates.append((est_rank, strassen_count, ns, ms, ps))
    
    # Sort by estimated rank (best first)
    candidates.sort(key=lambda x: (x[0], -x[1]))
    
    for est_rank, strassen_count, ns, ms, ps in candidates:
        if time.time() > deadline:
            break
        if est_rank >= best_rank:
            continue  # Can't be better than current best
        
        try:
            U, V, W = _block_decompose(n, m, p, ns, ms, ps)
            rank = U.shape[1]
            
            if rank < best_rank:
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = (U, V, W)
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Check for integer/half-integer
                    valid = True
                    for mat in [U_list, V_list, W_list]:
                        for row in mat:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
