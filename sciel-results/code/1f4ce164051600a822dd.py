import time

def construct():
    """
    Return a decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a block decomposition:
    - Split indices into G0={0,1}, G1={2,3}, G2={4}
    - For any 2x2x2 sub-block (where I, J, K are all size 2), use Strassen's 7-term algorithm
    - For blocks involving the singleton group G2={4}, use rank-1 outer product terms
    - For remaining 2x2x1 or 2x1x2 blocks (which don't exist in this partition), fall back
    
    Actually with partition [2,2,1]:
    - Core blocks (all from G0,G1): 8 blocks of size 2x2x2 -> Strassen (7 each) = 56
    - Shell blocks (at least one from G2): 19 blocks
      - Each shell block has at least one dimension of size 1
      - We can handle each with rank 1 (outer product)
    
    Total rank: 56 + 19 = 75
    Score: 125/75 = 1.667
    """
    start = time.time()
    deadline = start + 110  # safety margin before 120s kill
    
    n, m, p = 5, 5, 5
    
    try:
        result = _construct_optimized_block_decomposition(n, m, p)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _construct_optimized_block_decomposition(n, m, p):
    """
    Construct a low-rank decomposition using block structure.
    
    Split indices into G0 = {0, 1}, G1 = {2, 3}, G2 = {4}.
    Use Strassen for 2x2x2 blocks, rank-1 for blocks involving G2.
    """
    G0 = [0, 1]
    G1 = [2, 3]
    G2 = [4]
    groups = [G0, G1, G2]
    
    terms = []  # list of (u, v, w) sparse entries
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx, val in u_entries:
            u[idx] = val
        for idx, val in v_entries:
            v[idx] = val
        for idx, val in w_entries:
            w[idx] = val
        
        terms.append((u, v, w))
    
    # Iterate over all block combinations
    for i_group in groups:
        for j_group in groups:
            for k_group in groups:
                # Check if this is a 2x2x2 block (all groups have size 2)
                if len(i_group) == 2 and len(j_group) == 2 and len(k_group) == 2:
                    # Apply Strassen
                    _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p)
                else:
                    # For blocks where at least one dimension is size 1,
                    # use a single rank-1 term
                    _add_rank1_block(i_group, j_group, k_group, add_term, n, m, p)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            U[idx][r] = u[idx]
        for idx in range(m * p):
            V[idx][r] = v[idx]
        for idx in range(p * n):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}


def _add_rank1_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add a rank-1 term for a block where at least one dimension is size 1.
    
    The matrix multiplication tensor entry T[i,j,k] = 1 for all i,j,k.
    For a block (I, J, K), we need:
        sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for i in I, j in J, k in K
    
    Using a single rank-1 term:
        U[i,j,r] = 1 for i in I, j in J
        V[j,k,r] = 1 for j in J, k in K  
        W[k,i,r] = 1 for k in K, i in I
    
    This gives U*V*W = 1*1*1 = 1 for all (i,j,k) in the block.
    """
    u_entries = []
    v_entries = []
    w_entries = []
    
    for i in i_group:
        for j in j_group:
            u_entries.append((i * m + j, 1))
    
    for j in j_group:
        for k in k_group:
            v_entries.append((j * p + k, 1))
    
    for k in k_group:
        for i in i_group:
            w_entries.append((k * n + i, 1))
    
    add_term(u_entries, v_entries, w_entries)


def _add_strassen_block(i_group, j_group, k_group, add_term, n, m, p):
    """
    Add Strassen terms for a specific 2x2x2 block.
    i_group, j_group, k_group are lists of 2 global indices.
    """
    i_map = i_group
    j_map = j_group
    k_map = k_group
    
    def map_idx(group, local_idx):
        return group[local_idx]
    
    # Term 1: (A00+A11)(B00+B11) -> C00+1, C11+1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 2: (A10+A11)B00 -> C10+1, C11-1
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 1), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), -1)]
    add_term(u, v, w)
    
    # Term 3: A00(B01-B11) -> C01+1, C11+1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 1), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), -1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 4: A11(B10-B00) -> C00+1, C10+1
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 0), -1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1), (map_idx(k_map, 0)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 5: (A00+A01)B11 -> C00-1, C01+1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 1), 1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), -1), (map_idx(k_map, 1)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)
    
    # Term 6: (A10-A00)(B00+B01) -> C11+1
    u = [(map_idx(i_map, 1)*m + map_idx(j_map, 0), 1), (map_idx(i_map, 0)*m + map_idx(j_map, 0), -1)]
    v = [(map_idx(j_map, 0)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 0)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 1)*n + map_idx(i_map, 1), 1)]
    add_term(u, v, w)
    
    # Term 7: (A01-A11)(B10+B11) -> C00+1
    u = [(map_idx(i_map, 0)*m + map_idx(j_map, 1), 1), (map_idx(i_map, 1)*m + map_idx(j_map, 1), -1)]
    v = [(map_idx(j_map, 1)*p + map_idx(k_map, 0), 1), (map_idx(j_map, 1)*p + map_idx(k_map, 1), 1)]
    w = [(map_idx(k_map, 0)*n + map_idx(i_map, 0), 1)]
    add_term(u, v, w)


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
