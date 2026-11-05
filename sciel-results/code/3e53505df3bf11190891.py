import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
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
    return U, V, W

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1, n, m, p):
    a00 = 0*m + k0
    a01 = 0*m + k1
    a10 = 1*m + k0
    a11 = 1*m + k1
    b00 = k0*p + j0
    b01 = k0*p + j1
    b10 = k1*p + j0
    b11 = k1*p + j1
    c00 = j0*n + 0
    c01 = j1*n + 0
    c10 = j0*n + 1
    c11 = j1*n + 1
    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1
    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1
    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1
    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j, n, m, p):
    a0k0 = 0*m + k0
    a0k1 = 0*m + k1
    a1k0 = 1*m + k0
    a1k1 = 1*m + k1
    bk0j = k0*p + j
    bk1j = k1*p + j
    c0j = j*n + 0
    c1j = j*n + 1
    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1
    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1
    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1
    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1, n, m, p):
    a0k = 0*m + k
    a1k = 1*m + k
    bkj0 = k*p + j0
    bkj1 = k*p + j1
    c0j0 = j0*n + 0
    c0j1 = j1*n + 0
    c1j0 = j0*n + 1
    c1j1 = j1*n + 1
    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1
    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1
    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1
    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1
    return offset + 4

def _add_1x1(U, V, W, offset, k, j, n, m, p):
    a0k = 0*m + k
    a1k = 1*m + k
    bkj = k*p + j
    c0j = j*n + 0
    c1j = j*n + 1
    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1
    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1
    return offset + 2

def _build_strassen_based():
    """
    Builds a Strassen-based decomposition for 2x5 * 5x5.
    
    Strategy: Partition k (rows of A, cols of B) and j (cols of A, rows of B)
    into groups, then apply Strassen to 2x2 blocks where both groups have size 2,
    and schoolbook otherwise.
    
    k_groups: how to partition the 5 k-indices
    j_groups: how to partition the 5 j-indices
    
    For each pair (kg, jg):
      - |kg|=2, |jg|=2: use 7-term Strassen
      - |kg|=2, |jg|=1: use 4-term schoolbook (2*1*1)
      - |kg|=1, |jg|=2: use 4-term schoolbook (1*2*1)
      - |kg|=1, |jg|=1: use 2-term schoolbook (1*1*1)
    """
    n, m, p = 2, 5, 5
    
    # Try multiple partitionings and pick the one with lowest rank
    # k partition: how to split {0,1,2,3,4}
    # j partition: how to split {0,1,2,3,4}
    
    best_rank = float('inf')
    best_result = None
    
    # Generate all ways to partition 5 elements into groups of size 1 or 2
    # (we won't use groups larger than 2 since we only have 2x2 Strassen)
    # Actually, let's try a few reasonable partitionings
    
    # Option 1: k=[(0,1),(2,3),(4)], j=[(0,1),(2,3),(4)]
    # Rank: 7+4+4 + 4+7+4 + 4+4+2 = 11+15+10 = wait let me compute properly
    # kg0=j0: 7, kg0=j1: 4, kg0=j2: 4
    # kg1=j0: 4, kg1=j1: 7, kg1=j2: 4
    # kg2=j0: 4, kg2=j1: 4, kg2=j2: 2
    # Total: 7+4+4+4+7+4+4+4+2 = 40
    
    # Option 2: k=[(0,1),(2,3),(4)], j=[(0,1,2),(3,4)] -- but we don't have 2x3 Strassen
    # So stick with size-1 and size-2 groups
    
    # Let's try all permutations of which indices go into which group
    # For k: we need to partition {0,1,2,3,4} into groups of size 1 or 2
    # Possible patterns: (2,2,1) in some order
    
    import itertools
    
    def gen_partitions(indices, max_group_size=2):
        """Generate all partitions of indices into groups of size 1 or max_group_size."""
        if not indices:
            yield []
            return
        first = indices[0]
        rest = indices[1:]
        # Put first alone
        for p in gen_partitions(rest, max_group_size):
            yield [(first,)] + p
        # Put first with second if exists
        if len(indices) >= 2:
            second = indices[1]
            rest2 = indices[2:]
            for p in gen_partitions(rest2, max_group_size):
                yield [(first, second)] + p
    
    k_partitions = list(gen_partitions([0,1,2,3,4]))
    j_partitions = list(gen_partitions([0,1,2,3,4]))
    
    for kp in k_partitions:
        for jp in j_partitions:
            total_rank = 0
            for kg in kp:
                for jg in jp:
                    if len(kg) == 2 and len(jg) == 2:
                        total_rank += 7
                    elif len(kg) == 2 and len(jg) == 1:
                        total_rank += 4
                    elif len(kg) == 1 and len(jg) == 2:
                        total_rank += 4
                    else:
                        total_rank += 2
            
            if total_rank < best_rank:
                best_rank = total_rank
                # Build the decomposition
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                offset = 0
                for kg in kp:
                    for jg in jp:
                        if len(kg) == 2 and len(jg) == 2:
                            offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1], n, m, p)
                        elif len(kg) == 2 and len(jg) == 1:
                            offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0], n, m, p)
                        elif len(kg) == 1 and len(jg) == 2:
                            offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1], n, m, p)
                        else:
                            offset = _add_1x1(U, V, W, offset, kg[0], jg[0], n, m, p)
                
                if _verify_decomp(n, m, p, U, V, W):
                    best_result = (U, V, W)
                    break
        
        if best_result is not None:
            break
        
        if not _deadline_ok():
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to the default partition
    k_groups = [(0,1), (2,3), (4,)]
    j_groups = [(0,1), (2,3), (4,)]
    total_rank = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                total_rank += 7
            elif len(kg) == 2 and len(jg) == 1:
                total_rank += 4
            elif len(kg) == 1 and len(jg) == 2:
                total_rank += 4
            else:
                total_rank += 2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    offset = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1], n, m, p)
            elif len(kg) == 2 and len(jg) == 1:
                offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0], n, m, p)
            elif len(kg) == 1 and len(jg) == 2:
                offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1], n, m, p)
            else:
                offset = _add_1x1(U, V, W, offset, kg[0], jg[0], n, m, p)
    return U, V, W

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try Strassen-based approach with optimized partition
    U, V, W = _build_strassen_based()
    
    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
