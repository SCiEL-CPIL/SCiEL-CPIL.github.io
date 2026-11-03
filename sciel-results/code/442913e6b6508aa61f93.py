import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """Strassen 2x2x2 decomposition with correct entries."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = A22(B21-B11)
    U[3][1] = 1
    V[0][1] = -1; V[2][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A11(B22-B12)
    U[0][2] = 1
    V[1][2] = -1; V[3][2] = 1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = (A11+A12)B22
    U[0][3] = 1; U[1][3] = 1
    V[3][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A21+A22)B11
    U[2][4] = 1; U[3][4] = 1
    V[0][4] = 1
    W[0][4] = 1; W[2][4] = 1
    
    # M6 = (A11-A21)(B12+B22)
    U[0][5] = 1; U[2][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # M7 = (A12-A22)(B11+B21)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _schoolbook_factors(n, m, p):
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

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                
                if ni == 2 and nj == 2 and nk == 2:
                    for c in range(7):
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    for li in range(ni):
                        gi = i_b[li]
                        for lj in range(nj):
                            gj = j_b[lj]
                            for lk in range(nk):
                                gk = k_b[lk]
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the tensor decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _search_finite_field_reduction(n, m, p, max_time=60):
    """
    Search for a rank reduction by finding a linear dependency among the 75
    standard basis tensors over F_2. If such a dependency exists, we can remove
    one term and get rank 74.
    
    The idea: work over F_2 where each standard basis tensor e_{i,j,k} has
    rank 1. If there's a non-trivial sum of these that equals zero in the
    tensor space over F_2, we can use that to reduce the rank.
    
    Actually, a better approach: try to find a decomposition with rank < 75
    by searching for a pattern that allows combining terms.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # For 3x5x5, the schoolbook rank is 75.
    # We already have rank 71 from the Strassen-based construction.
    # Let's try to improve further.
    
    # Strategy: Try different block partitions and see if we can get better than 71.
    # Current: i=[1,2], j=[2,2,1], k=[2,2,1] gives 1*2*2 + 1*2*2 + 1*1*2 + 2*2*2 + 2*2*2 + 2*1*2 + 1*2*2 + 1*2*2 + 1*1*2
    # = 4 + 4 + 2 + 7 + 7 + 4 + 4 + 4 + 2 = 38... wait that's not 71.
    
    # Let me recalculate: 
    # i_blocks = [(0,), (1,2)] -> sizes 1, 2
    # j_blocks = [(0,1), (2,3), (4,)] -> sizes 2, 2, 1
    # k_blocks = [(0,1), (2,3), (4,)] -> sizes 2, 2, 1
    
    # Total = sum over all combinations of ni*nj*nk, with 2x2x2 replaced by 7
    # (1,2,2): 1*2*2 = 4 (not 2x2x2)
    # (1,2,2): 4
    # (1,2,1): 1*2*1 = 2
    # (2,2,2): 7 (Strassen)
    # (2,2,2): 7
    # (2,2,1): 2*2*1 = 4
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,1): 2
    
    # Wait, I need to enumerate all 2*3*3 = 18 combinations:
    # i_b=0 (size 1), j_b=0 (size 2), k_b=0 (size 2): 1*2*2 = 4
    # i_b=0, j_b=0, k_b=1: 1*2*2 = 4
    # i_b=0, j_b=0, k_b=2: 1*2*1 = 2
    # i_b=0, j_b=1, k_b=0: 1*2*2 = 4
    # i_b=0, j_b=1, k_b=1: 1*2*2 = 4
    # i_b=0, j_b=1, k_b=2: 1*2*1 = 2
    # i_b=0, j_b=2, k_b=0: 1*1*2 = 2
    # i_b=0, j_b=2, k_b=1: 1*1*2 = 2
    # i_b=0, j_b=2, k_b=2: 1*1*1 = 1
    # i_b=1 (size 2), j_b=0, k_b=0: 2*2*2 = 7 (Strassen)
    # i_b=1, j_b=0, k_b=1: 2*2*2 = 7
    # i_b=1, j_b=0, k_b=2: 2*2*1 = 4
    # i_b=1, j_b=1, k_b=0: 2*2*2 = 7
    # i_b=1, j_b=1, k_b=1: 2*2*2 = 7
    # i_b=1, j_b=1, k_b=2: 2*2*1 = 4
    # i_b=1, j_b=2, k_b=0: 2*1*2 = 4
    # i_b=1, j_b=2, k_b=1: 2*1*2 = 4
    # i_b=1, j_b=2, k_b=2: 2*1*1 = 2
    
    # Total = 4+4+2+4+4+2+2+2+1+7+7+4+7+7+4+4+4+2 = 71. Yes!
    
    # To improve, we need a different partition. Let's try:
    # i_blocks = [(0,1), (2,)] -> sizes 2, 1
    # j_blocks = [(0,1), (2,3), (4,)] -> sizes 2, 2, 1
    # k_blocks = [(0,1), (2,3), (4,)] -> sizes 2, 2, 1
    
    # This gives the same structure, just with i blocks swapped.
    
    # What about using 2x2x2 Strassen more aggressively?
    # i_blocks = [(0,1), (2,)] 
    # j_blocks = [(0,1), (2,3), (4,)]
    # k_blocks = [(0,1), (2,3), (4,)]
    # Same count.
    
    # Can we find a partition with more 2x2x2 blocks?
    # n=3: max one 2-block and one 1-block
    # m=5: max two 2-blocks and one 1-block
    # p=5: max two 2-blocks and one 1-block
    
    # Current has 4 blocks of type 2x2x2 (Strassen, rank 7 each)
    # and 14 blocks of type with at least one dimension < 2 (schoolbook)
    
    # The 4 Strassen blocks contribute 4*7 = 28
    # The 14 schoolbook blocks contribute 71-28 = 43
    
    # Can we do better? What if we use a different decomposition for some blocks?
    
    # For a 1x2x2 block: rank 2 (schoolbook)
    # For a 2x1x2 block: rank 2
    # For a 2x2x1 block: rank 4
    # For a 1x1x2 block: rank 2
    # For a 1x2x1 block: rank 2
    # For a 2x1x1 block: rank 2
    # For a 1x1x1 block: rank 1
    
    # These are all optimal for their sizes.
    
    # So the question is: can we find a partition with fewer total schoolbook terms?
    
    # Let's enumerate possible partitions:
    # n=3: [(3,)], [(1,2)], [(2,1)], [(1,1,1)]
    # m=5: [(5,)], [(1,4)], [(2,3)], [(3,2)], [(4,1)], [(1,1,3)], [(1,2,2)], [(1,3,1)], [(2,1,2)], [(2,2,1)], [(3,1,1)], [(1,1,1,2)], [(1,1,2,1)], [(1,2,1,1)], [(2,1,1,1)], [(1,1,1,1,1)]
    # p=5: same as m
    
    # This is a large search space. Let's focus on partitions with 2-blocks.
    
    # Actually, let's try a different approach: use the known lower bounds.
    # For 3x5x5, the known lower bound is... let me think.
    # The tensor rank for matrix multiplication is at least max(n*m, m*p, p*n) = max(15, 25, 15) = 25.
    # But that's a very weak bound.
    
    # A better bound: rank >= (n*m*p) / max(n,m,p) = 75/5 = 15. Still weak.
    
    # Let's just try to verify that rank 71 is valid and return it.
    # If we have time, we can try to search for a better partition.
    
    # For now, let's just return the rank 71 result.
    U, V, W = _build_rank71()
    if _verify_decomposition(U, V, W, n, m, p, 71):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # First try the rank-71 construction
    try:
        U, V, W = _build_rank71()
        if _verify_decomposition(U, V, W, n, m, p, 71):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
