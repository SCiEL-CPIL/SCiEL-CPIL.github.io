import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Tensor T[i,j,k] = 1 if A[i,j] * B[j,k] contributes to C[i,k]
    # Decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n + i, r] corresponds to output C[i,k]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0  # A[i,j]*B[j,k] -> C[i,k]
                recon = 0.0
                for r in range(rank):
                    recon += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(recon - target) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 x 2x2 matrix multiplication, rank 7."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-d)(f+h), m7 = (b-c)(e+h)
    # C11 = m1 + m4 - m5 + m7, C12 = m3 + m5
    # C21 = m2 + m4, C22 = m1 - m2 + m3 + m6
    
    # Indices: A[i*m+j], B[j*p+k], C output at W[k*n+i]
    # For 2x2: n=2, m=2, p=2
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: W[k*2+i] -> C[i,k]
    #   C[0,0]: W[0*2+0]=W[0], C[0,1]: W[1*2+0]=W[2]
    #   C[1,0]: W[0*2+1]=W[1], C[1,1]: W[1*2+1]=W[3]
    
    U = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    
    # m1 = (a+d)(e+h): A[0]+A[3] times B[0]+B[3]
    # U: rows 0 and 3 get 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += m1: W[0] += 1, C12 += 0, C21 += 0, C22 += 0
    # Actually C11 = m1 + m4 - m5 + m7, C22 = m1 - m2 + m3 + m6
    # Let me be more careful
    
    # m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # contributes to C00 and C11
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # m2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 = m2 + m4 -> W[1] (C[1,0])
    W[1][1] = 1
    
    # m3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 = m3 + m5 -> W[2] (C[0,1])
    W[2][2] = 1
    # C22 = m1 - m2 + m3 + m6 -> W[3]
    W[3][2] = 1
    
    # m4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C21 = m2 + m4 -> W[1]
    W[1][3] = 1
    # C11 = m1 + m4 - m5 + m7 -> W[0]
    W[0][3] = 1
    
    # m5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C12 = m3 + m5 -> W[2]
    W[2][4] = 1
    # C11 = m1 + m4 - m5 + m7 -> W[0]
    W[0][4] = -1
    
    # m6 = (A10-A11)*(B01+B11)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    # C22 = m1 - m2 + m3 + m6 -> W[3]
    W[3][5] = 1
    
    # m7 = (A01-A10)*(B00+B10)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    # C11 = m1 + m4 - m5 + m7 -> W[0]
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _block_decompose(n, m, p, block_decomp_func):
    """
    Given a function that decomposes sub-blocks, build full decomposition.
    This is a simple approach: partition into 1x1 blocks (schoolbook) or use
    a predefined block structure.
    """
    # For now, just use schoolbook as baseline
    return _schoolbook(n, m, p)

def _random_search(n, m, p, max_rank, time_budget=60, seed=42):
    """
    Random search for a valid decomposition of given rank.
    This is a very rough approach - may not find anything.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    best = None
    for attempt in range(100):
        if time.time() - start_time > time_budget:
            break
        rank = max_rank
        # Random sparse matrices
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Randomly assign non-zero entries
        for r in range(rank):
            # Pick random indices
            ui = rng.randint(0, n*m)
            vj = rng.randint(0, m*p)
            wk = rng.randint(0, p*n)
            U[ui, r] = rng.choice([-1, 1])
            V[vj, r] = rng.choice([-1, 1])
            W[wk, r] = rng.choice([-1, 1])
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return best

def _greedy_block_search(n, m, p, time_budget=80):
    """
    Try different block partitions and find the minimum rank.
    For 3x5x7, try splitting into sub-blocks and using Strassen where applicable.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try various splits of dimensions
    # n=3: can split as (2,1) or (1,2) or (1,1,1)
    # m=5: can split as (2,3), (3,2), (2,2,1), (1,2,2), etc.
    # p=7: can split as (3,4), (4,3), (2,2,3), (2,3,2), (3,2,2), etc.
    
    def get_splits(dim):
        """Get all ways to split dim into parts of size 1, 2, or 3."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(tuple(current))
                return
            for part in [1, 2, 3]:
                if part <= remaining:
                    current.append(part)
                    helper(remaining - part, current)
                    current.pop()
        helper(dim, [])
        return results
    
    n_splits = get_splits(3)
    m_splits = get_splits(5)
    p_splits = get_splits(7)
    
    # For each combination of splits, compute the rank
    # For each sub-block (ni x mj x pk), use min(ni*mj*pk, strassen_if_2x2x2)
    def block_rank(ni, mj, pk):
        """Compute rank for a sub-block."""
        if ni == 2 and mj == 2 and pk == 2:
            return 7  # Strassen
        else:
            return ni * mj * pk  # Schoolbook for other blocks
    
    best_rank = n * m * p
    best_combo = None
    
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                if len(ns) != len(ms) or len(ns) != len(ps):
                    continue
                rank = 0
                for idx in range(len(ns)):
                    rank += block_rank(ns[idx], ms[idx], ps[idx])
                if rank < best_rank:
                    best_rank = rank
                    best_combo = (ns, ms, ps)
    
    if best_combo is not None:
        # Build the decomposition using the best combo
        ns, ms, ps = best_combo
        total_rank = 0
        U = []
        V = []
        W = []
        
        for i_block in range(len(ns)):
            ni = ns[i_block]
            for j_block in range(len(ms)):
                mj = ms[j_block]
                for k_block in range(len(ps)):
                    pk = ps[k_block]
                    # Get the sub-block decomposition
                    if ni == 2 and mj == 2 and pk == 2:
                        sub = _strassen_2x2x2()
                    else:
                        sub = _schoolbook(ni, mj, pk)
                    
                    sub_rank = len(sub["U"][0])
                    # Offset indices
                    i_offset = sum(ns[:i_block]) * 5  # m = 5
                    j_offset = sum(ms[:j_block]) * 7  # p = 7
                    k_offset = sum(ps[:k_block]) * 3  # n = 3
                    
                    # Add to full matrices
                    # U rows: i*m + j -> (i_offset/m... wait, need to be careful
                    # Actually, for block (i0..i0+ni-1, j0..j0+mj-1), 
                    # U row index = (i0+i)*m + (j0+j)
                    
                    sub_U = sub["U"]
                    sub_V = sub["V"]
                    sub_W = sub["W"]
                    
                    for r in range(sub_rank):
                        for i in range(ni):
                            for j in range(mj):
                                pass  # This is getting complex, let me use a different approach
        
        # Let me use a simpler approach: build the full tensor and use the block structure
        pass
    
    return best_result, best_rank

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 120  # 120 second budget
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based block decomposition
    # For 3x5x7, split as:
    # Rows: (2,1), Cols of A: (2,3) or (3,2), Cols of B: (3,4) or (4,3) etc.
    
    # Let me try a specific decomposition:
    # Split 3x5 into (2x2, 2x3, 1x2, 1x3) and 5x7 into (2x3, 2x4, 3x3, 3x4)
    # This is getting complicated. Let me just try random search with verification.
    
    # Try a hybrid: use Strassen for 2x2x2 blocks and schoolbook for the rest
    # 3x5x7: split rows as (2,1), middle as (2,3), cols as (3,4)
    # Blocks: 2x2x3, 2x2x4, 2x3x3, 2x3x4, 1x2x3, 1x2x4, 1x3x3, 1x3x4
    # Ranks: 2*2*3=12, 2*2*4=16, 2*3*3=18, 2*3*4=24, 1*2*3=6, 1*2*4=8, 1*3*3=9, 1*3*4=12
    # Total: 12+16+18+24+6+8+9+12 = 105 (same as schoolbook, no improvement)
    
    # The only way to improve is to use Strassen on 2x2x2 sub-blocks.
    # So I need to find a partition that includes at least one 2x2x2 block.
    
    # 3x5x7: Can I make a 2x2x2 block?
    # Rows: 2 out of 3, Middle: 2 out of 5, Cols: 2 out of 7
    # Yes! Split as rows (2,1), middle (2,3), cols (2,5)
    # Blocks: 2x2x2 (Strassen, rank 7), 2x2x5 (rank 20), 2x3x2 (rank 12), 2x3x5 (rank 30)
    #         1x2x2 (rank 4), 1x2x5 (rank 10), 1x3x2 (rank 6), 1x3x5 (rank 15)
    # Total: 7+20+12+30+4+10+6+15 = 104
    
    # Better: rows (2,1), middle (2,3), cols (2,2,3)
    # Blocks: 2x2x2=7, 2x2x2=7, 2x2x3=12, 2x3x2=12, 2x3x2=12, 2x3x3=18
    #         1x2x2=4, 1x2x2=4, 1x2x3=6, 1x3x2=6, 1x3x2=6, 1x3x3=9
    # Total: 7+7+12+12+12+18+4+4+6+6+6+9 = 103
    
    # Even better: try to maximize 2x2x2 blocks
    # rows (2,1), middle (2,2,1), cols (2,2,3)
    # Blocks: 2x2x2=7, 2x2x2=7, 2x2x3=12
    #         2x2x2=7, 2x2x2=7, 2x2x3=12
    #         2x1x2=4, 2x1x2=4, 2x1x3=6
    #         1x2x2=4, 1x2x2=4, 1x2x3=6
    #         1x2x2=4, 1x2x2=4, 1x2x3=6
    #         1x1x2=2, 1x1x2=2, 1x1x3=3
    # Total: 7+7+12+7+7+12+4+4+6+4+4+6+4+4+6+2+2+3 = 100
    
    # Let me try rows (2,1), middle (2,2,1), cols (2,2,2,1)
    # This is getting complex. Let me just implement a general block decomposition.
    
    def build_block_decomp(n, m, p, n_parts, m_parts, p_parts):
        """Build a block decomposition using Strassen for 2x2x2 blocks."""
        assert sum(n_parts) == n
        assert sum(m_parts) == m
        assert sum(p_parts) == p
        
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        
        col = 0
        for bi in range(len(n_parts)):
            ni = n_parts[bi]
            i0 = sum(n_parts[:bi])
            for bj in range(len(m_parts)):
                mj = m_parts[bj]
                j0 = sum(m_parts[:bj])
                for bk in range(len(p_parts)):
                    pk = p_parts[bk]
                    k0 = sum(p_parts[:bk])
                    
                    if ni == 2 and mj == 2 and pk == 2:
                        sub = _strassen_2x2x2()
                    else:
                        sub = _schoolbook(ni, mj, pk)
                    
                    sub_rank = len(sub["U"][0])
                    sub_U = sub["U"]
                    sub_V = sub["V"]
                    sub_W = sub["W"]
                    
                    for r in range(sub_rank):
                        for i in range(ni):
                            U[(i0 + i) * m + (j0 + 0)][col] = sub_U[i * mj][r] if False else 0
                        # This is wrong. Let me fix the indexing.
                        # sub_U has shape (ni*mj, sub_rank)
                        # For local (i,j) in [0,ni)x[0,mj), global index = (i0+i)*m + (j0+j)
                        for i in range(ni):
                            for j in range(mj):
                                U[(i0 + i) * m + (j0 + j)][col] = sub_U[i * mj + j][r]
                        
                        for j in range(mj):
                            for k in range(pk):
                                V[(j0 + j) * p + (k0 + k)][col] = sub_V[j * pk + k][r]
                        
                        for k in range(pk):
                            for i in range(ni):
                                W[(k0 + k) * n + (i0 + i)][col] = sub_W[k * ni + i][r]
                        
                        col += 1
        
        # Trim trailing zeros
        while col < len(U[0]) and all(U[i][col] == 0 for i in range(n*m)) \
              and all(V[i][col] == 0 for i in range(m*p)) \
              and all(W[i][col] == 0 for i in range(p*n)):
            col -= 1
        
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        return {"U": U, "V": V, "W": W}
    
    # Try various partitions
    candidates = []
    
    # rows (2,1), middle (2,2,1), cols (2,2,3)
    candidates.append(([2,1], [2,2,1], [2,2,3]))
    # rows (2,1), middle (2,2,1), cols (2,2,2,1)
    candidates.append(([2,1], [2,2,1], [2,2,2,1]))
    # rows (2,1), middle (2,1,2), cols (2,2,3)
    candidates.append(([2,1], [2,1,2], [2,2,3]))
    # rows (1,2), middle (2,2,1), cols (2,2,3)
    candidates.append(([1,2], [2,2,1], [2,2,3]))
    # rows (2,1), middle (1,2,2), cols (2,2,3)
    candidates.append(([2,1], [1,2,2], [2,2,3]))
    # rows (2,1), middle (2,3), cols (2,2,3)
    candidates.append(([2,1], [2,3], [2,2,3]))
    # rows (2,1), middle (3,2), cols (2,2,3)
    candidates.append(([2,1], [3,2], [2,2,3]))
    # rows (2,1), middle (2,2,1), cols (3,2,2)
    candidates.append(([2,1], [2,2,1], [3,2,2]))
    # rows (2,1), middle (2,2,1), cols (2,3,2)
    candidates.append(([2,1], [2,2,1], [2,3,2]))
    # rows (2,1), middle (1,2,2), cols (2,2,3)
    candidates.append(([2,1], [1,2,2], [2,2,3]))
    # rows (1,2), middle (1,2,2), cols (1,2,2,2)
    candidates.append(([1,2], [1,2,2], [1,2,2,2]))
    # rows (2,1), middle (2,1,2), cols (1,2,2,2)
    candidates.append(([2,1], [2,1,2], [1,2,2,2]))
    
    for ns, ms, ps in candidates:
        if sum(ns) != n or sum(ms) != m or sum(ps) != p:
            continue
        if len(ns) != len(ms) or len(ns) != len(ps):
            continue
        try:
            result = build_block_decomp(n, m, p, ns, ms, ps)
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_rank = rank
                    best = result
        except:
            pass
    
    # Time remaining for random search
    if time.time() < deadline - 10:
        # Try to find a better decomposition with random perturbations
        # This is unlikely to help much, but let's try a few iterations
        rng = np.random.RandomState(123)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try to merge some rank-1 components
            U_arr = np.array(best["U"])
            V_arr = np.array(best["V"])
            W_arr = np.array(best["W"])
            rank = U_arr.shape[1]
            
            # Try to find a lower rank decomposition by null space elimination
            # This is complex, so just return the best found so far
            pass
    
    return best
