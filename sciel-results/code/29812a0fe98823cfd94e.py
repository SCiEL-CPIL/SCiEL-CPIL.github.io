import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7) -> rows (i,j)
    V shape: (4, 7) -> rows (j,k)
    W shape: (4, 7) -> rows (k,i)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1 # C00 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1 # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1 # C01 += M3
    
    # M4 = A11(B01-B00)
    U[3, 3] += 1
    V[1, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1 # C00 += M4
    W[2, 3] += 1 # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1 # C00 -= M5
    W[1, 4] += 1 # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1 # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1 # C00 += M7
    W[3, 6] += 1 # C11 += M7
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m x p schoolbook decomposition.
    Rank = n*m*p.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _build_block_decomposition(n, m, p, n_splits, m_splits, p_splits):
    """
    Builds the decomposition by partitioning into blocks.
    n_splits: list of sizes summing to n
    m_splits: list of sizes summing to m
    p_splits: list of sizes summing to p
    """
    # Calculate total rank
    total_rank = 0
    blocks = []
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if ns == 2 and ms == 2 and ps == 2:
                    blocks.append(('strassen', ns, ms, ps))
                    total_rank += 7
                else:
                    blocks.append(('schoolbook', ns, ms, ps))
                    total_rank += ns * ms * ps
                    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    u_offset = 0
    v_offset = 0
    w_offset = 0
    rank_offset = 0
    
    for block_type, ns, ms, ps in blocks:
        if block_type == 'strassen':
            U_block, V_block, W_block = _strassen_2x2x2()
            # Place into global matrices
            # Rows for U: u_offset to u_offset + ns*ms
            # Rows for V: v_offset to v_offset + ms*ps
            # Rows for W: w_offset to w_offset + ps*ns
            # Cols: rank_offset to rank_offset + 7
            
            U[u_offset : u_offset + ns*ms, rank_offset : rank_offset + 7] = U_block
            V[v_offset : v_offset + ms*ps, rank_offset : rank_offset + 7] = V_block
            W[w_offset : w_offset + ps*ns, rank_offset : rank_offset + 7] = W_block
            
            rank_offset += 7
        else:
            U_block, V_block, W_block = _schoolbook_block(ns, ms, ps)
            block_rank = ns * ms * ps
            
            U[u_offset : u_offset + ns*ms, rank_offset : rank_offset + block_rank] = U_block
            V[v_offset : v_offset + ms*ps, rank_offset : rank_offset + block_rank] = V_block
            W[w_offset : w_offset + ps*ns, rank_offset : rank_offset + block_rank] = W_block
            
            rank_offset += block_rank
            
        u_offset += ns * ms
        v_offset += ms * ps
        w_offset += ps * ns
        
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Partition 4, 5, 5 into 2,2 and 2,2,1
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    U, V, W, rank = _build_block_decomposition(n, m, p, n_splits, m_splits, p_splits)
    
    # Verify
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check all (i,j,k)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to full schoolbook if verification fails (should not happen)
        U, V, W, rank = _schoolbook_block(n, m, p)
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
