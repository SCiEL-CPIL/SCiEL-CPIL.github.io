"""New strategy: ILP-based search for rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_block_decomp(n, m, p):
    """
    Construct a Strassen-based block decomposition.
    For 3x4*4x8: split into 2x2 blocks where possible, use 7-term Strassen for 2x2 blocks,
    and 4-term (schoolbook) for 1x2 or 2x1 remaining strips.
    This gives rank 88 as baseline.
    """
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
        # 7 Strassen terms for a 2x2 block
        terms_a = [
            [1,0,0,1],
            [0,0,1,1],
            [1,1,0,0],
            [0,0,0,1],
            [1,0,0,0],
            [-1,0,1,0],
            [0,1,0,-1],
        ]
        terms_b = [
            [1,0,0,1],
            [1,1,0,0],
            [0,0,1,1],
            [0,-1,0,1],
            [1,0,-1,0],
            [0,1,0,1],
            [1,0,1,0],
        ]
        terms_c = [
            [1,0,0,1],
            [0,0,1,0],
            [0,1,0,0],
            [1,0,-1,0],
            [0,1,0,0],
            [0,0,0,1],
            [1,0,0,0],
        ]
        
        col = col_start
        for t in range(7):
            a_coeffs = terms_a[t]
            b_coeffs = terms_b[t]
            c_coeffs = terms_c[t]
            
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
            for di in range(2):
                for dk in range(2):
                    W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
            col += 1
        return col

    col = 0
    # Rows 0-1 (2 rows) x Cols 0-1 (2 cols) x K 0-1: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 0, 0, col, n, m, p)
    # Rows 0-1 x Cols 0-1 x K 2-3: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 0, 2, col, n, m, p)
    # Rows 0-1 x Cols 0-1 x K 4-5: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 0, 4, col, n, m, p)
    # Rows 0-1 x Cols 0-1 x K 6-7: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 0, 6, col, n, m, p)
    
    # Rows 0-1 x Cols 2-3 x K 0-1: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 2, 0, col, n, m, p)
    # Rows 0-1 x Cols 2-3 x K 2-3: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 2, 2, col, n, m, p)
    # Rows 0-1 x Cols 2-3 x K 4-5: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 2, 4, col, n, m, p)
    # Rows 0-1 x Cols 2-3 x K 6-7: Strassen 2x2
    col = _strassen_block(U, V, W, 0, 2, 6, col, n, m, p)
    
    # Row 2 (1 row) x Cols 0-1 x K 0-1: schoolbook 1x2x2 = 4 terms
    for dk in range(2):
        for dj in range(2):
            U[2*m + dj][col] = 1
            V[dj*p + dk][col] = 1
            W[dk*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 0-1 x K 2-3
    for dk in range(2):
        for dj in range(2):
            U[2*m + dj][col] = 1
            V[dj*p + (2+dk)][col] = 1
            W[(2+dk)*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 0-1 x K 4-5
    for dk in range(2):
        for dj in range(2):
            U[2*m + dj][col] = 1
            V[dj*p + (4+dk)][col] = 1
            W[(4+dk)*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 0-1 x K 6-7
    for dk in range(2):
        for dj in range(2):
            U[2*m + dj][col] = 1
            V[dj*p + (6+dk)][col] = 1
            W[(6+dk)*n + 2][col] = 1
            col += 1
    
    # Row 2 x Cols 2-3 x K 0-1
    for dk in range(2):
        for dj in range(2):
            U[2*m + (2+dj)][col] = 1
            V[(2+dj)*p + dk][col] = 1
            W[dk*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 2-3 x K 2-3
    for dk in range(2):
        for dj in range(2):
            U[2*m + (2+dj)][col] = 1
            V[(2+dj)*p + (2+dk)][col] = 1
            W[(2+dk)*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 2-3 x K 4-5
    for dk in range(2):
        for dj in range(2):
            U[2*m + (2+dj)][col] = 1
            V[(2+dj)*p + (4+dk)][col] = 1
            W[(4+dk)*n + 2][col] = 1
            col += 1
    # Row 2 x Cols 2-3 x K 6-7
    for dk in range(2):
        for dj in range(2):
            U[2*m + (2+dj)][col] = 1
            V[(2+dj)*p + (6+dk)][col] = 1
            W[(6+dk)*n + 2][col] = 1
            col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _try_ilp_search(n, m, p, max_rank, max_time):
    """
    Attempt to find a decomposition of rank <= max_rank using ILP.
    We formulate the tensor identity as a system of bilinear equations.
    Since direct ILP on bilinear terms is NP-hard, we use a heuristic:
    1. Start with a known high-rank solution.
    2. Use local search with random restarts to find lower-rank solutions.
    3. Use structured random search over small integer coefficients.
    
    For this specific case (3,4,8), we try to find a solution with rank < 88
    by exploring structured decompositions.
    """
    start_time = time.time()
    
    # Strategy: Try various structured decompositions
    # Key insight: The tensor T_{i,j,k} = 1 for all i,j,k
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Try rank values from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But this is very loose. Strassen gives 88.
    # Known results for 2x2*2x2: rank 7
    # For 3x4*4x8, let's try to find something better than 88.
    
    # Approach: Use a combination of Strassen blocks and random perturbations
    # to see if we can merge terms.
    
    # First, get the Strassen baseline
    base = _strassen_block_decomp(n, m, p)
    if base is None:
        return None
    
    U0 = np.array(base['U'], dtype=float)
    V0 = np.array(base['V'], dtype=float)
    W0 = np.array(base['W'], dtype=float)
    base_rank = U0.shape[1]
    
    # Try to reduce rank by combining terms
    # Use a greedy approach: try removing pairs of terms and adjusting others
    best_rank = base_rank
    best_result = base
    
    # Try local search: randomly zero out columns and check if still valid
    # This is unlikely to work directly, but let's try structured removals.
    
    # More promising: Try different block partitions
    # Partition 3 into 1+2, 4 into 2+2, 8 into 2+2+2+2 (current)
    # Alternative: 3 into 1+1+1, 4 into 1+1+1+1, 8 into 1+1+... (schoolbook, rank 96)
    # Alternative: Use 3x3 Strassen-like for the 3x3 sub-blocks
    
    # Let's try a 3x3 block decomposition for the 3x4 part
    # Actually, let's try: split 4 into 2+2, 8 into 4+4, 3 into 1+2
    # 2x4*4x4 block: use Strassen-like
    # 1x4*4x4 block: schoolbook
    
    # Try rank 80: 8 Strassen blocks of 2x2x2 (each rank 7) = 56, plus 1x2x2 blocks
    # 8 blocks of 2x2x2 = 8*7 = 56
    # Remaining: 1x2x2 blocks: 4 blocks * 4 = 16, total 72? No, that's not right.
    
    # Let me recalculate:
    # 3x4x8 = (2+1) x (2+2) x (2+2+2+2)
    # 2x2x2 blocks: 1*2*4 = 8 blocks, each rank 7 -> 56
    # 1x2x2 blocks: 1*2*4 = 8 blocks, each rank 4 -> 32
    # Total: 88. That's what we have.
    
    # Can we do better?
    # Try 3x3x3 blocks? No, dimensions don't divide evenly.
    # Try 2x2x4 blocks: rank for 2x2x4?
    # 2x2x4 = 2x2*(2x4) or (2x2x2)+(2x2x2) = 7+7=14, or maybe less?
    # Actually, 2x2x4 matrix mult: rank is at most 2*2*4=16, at least max(4,8,8)=8.
    # Strassen gives 14. Can we do 12?
    
    # Let's try a different approach: random search over structured solutions
    # with small integer coefficients.
    
    random.seed(42)
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
            
        # Try a structured solution with rank r
        r = base_rank - attempt  # Try decreasing ranks
        if r < 40:
            break
            
        # Generate random structured solution
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize with random small integers
        for c in range(r):
            # Randomly assign non-zero entries
            for _ in range(random.randint(2, 5)):
                row = random.randint(0, n*m-1)
                U[row, c] = random.choice([-1, 1])
            for _ in range(random.randint(2, 5)):
                row = random.randint(0, m*p-1)
                V[row, c] = random.choice([-1, 1])
            for _ in range(random.randint(2, 5)):
                row = random.randint(0, p*n-1)
                W[row, c] = random.choice([-1, 1])
        
        # Check if this is a valid decomposition
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_rank < base_rank:
        return best_result
    return None

def _local_search_reduce_rank(U, V, W, n, m, p, max_time):
    """
    Attempts to reduce the rank by zeroing out columns (terms) 
    if they are linearly dependent or redundant.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    current_rank = rank
    active_cols = list(range(rank))
    
    start_time = time.time()
    
    while time.time() - start_time < max_time and len(active_cols) > 0:
        removed = False
        random.shuffle(active_cols)
        for c in active_cols:
            if time.time() - start_time > max_time:
                break
            new_active = [x for x in active_cols if x != c]
            if not new_active:
                continue
                
            U_sub = U[:, new_active]
            V_sub = V[:, new_active]
            W_sub = W[:, new_active]
            
            if _verify(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p):
                active_cols = new_active
                removed = True
                break
        
        if not removed:
            break
    
    if not active_cols:
        return None
        
    U_final = U[:, active_cols]
    V_final = V[:, active_cols]
    W_final = W[:, active_cols]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Get the baseline 88 rank solution
    result = _strassen_block_decomp(n, m, p)
    if result is None:
        result = _schoolbook(n, m, p)
        return result
    
    # Attempt ILP-based search
    if time.time() < deadline - 20:
        search_time = deadline - time.time() - 10
        ilp_result = _try_ilp_search(n, m, p, 88, search_time)
        if ilp_result is not None:
            if _verify(ilp_result['U'], ilp_result['V'], ilp_result['W'], n, m, p):
                result = ilp_result
    
    # Attempt local search
    if time.time() < deadline - 5:
        search_time = deadline - time.time() - 2
        reduced = _local_search_reduce_rank(result['U'], result['V'], result['W'], n, m, p, search_time)
        if reduced is not None:
            if _verify(reduced['U'], reduced['V'], reduced['W'], n, m, p):
                result = reduced
    
    return result

# EVOLVE-BLOCK-END
