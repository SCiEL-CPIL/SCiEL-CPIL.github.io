"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_factors():
    """
    Return (U_local, V_local, W_local) for 2x2x2 Strassen decomposition.
    
    U_local: shape (4, 7) for (i,a) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    V_local: shape (4, 7) for (b,k) in {0,0:0, 0,1:1, 1,0:2, 1,1:3}
    W_local: shape (4, 7) for (k,i) in {0,0:0, 0,1:1, 1,0:2, 1,1:3} i.e. k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M0 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M1 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2: M2 = (A00)(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # c=3: M3 = (A11)(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # c=4: M4 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # c=5: M5 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # c=6: M6 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the factor matrices correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    err = abs(s - expected)
                    if err > max_err:
                        max_err = err
                    if err > tol:
                        return False, max_err
    return True, max_err


def _generate_partitions(size):
    """Generate all possible partitions of a dimension into blocks of size 1 or 2."""
    if size == 0:
        return [[]]
    if size == 1:
        return [[(0, 1)]]
    if size == 2:
        return [[(0, 2)], [(0, 1), (1, 1)]]
    
    results = []
    if size >= 2:
        for rest in _generate_partitions(size - 2):
            results.append([(0, 2)] + [(start + 2, sz) for start, sz in rest])
    for rest in _generate_partitions(size - 1):
        results.append([(0, 1)] + [(start + 1, sz) for start, sz in rest])
    
    return results


def _build_decomposition(I_blocks, J_blocks, K_blocks, U_str, V_str, W_str):
    """Build decomposition from block partitions."""
    n, m, p = DIMENSIONS
    
    total_rank = 0
    block_info = []
    
    for (i_start, i_size) in I_blocks:
        for (j_start, j_size) in J_blocks:
            for (k_start, k_size) in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    rank_contrib = 7
                    method = 'strassen'
                else:
                    rank_contrib = i_size * j_size * k_size
                    method = 'schoolbook'
                total_rank += rank_contrib
                block_info.append((i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method))
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for (i_start, i_size, j_start, j_size, k_start, k_size, rank_contrib, method) in block_info:
        if method == 'strassen':
            for local_i in range(2):
                for local_a in range(2):
                    global_i = i_start + local_i
                    global_a = j_start + local_a
                    global_u_row = global_i * m + global_a
                    for c in range(7):
                        U[global_u_row, col_offset + c] = U_str[local_i * 2 + local_a, c]
            
            for local_b in range(2):
                for local_k in range(2):
                    global_b = j_start + local_b
                    global_k = k_start + local_k
                    global_v_row = global_b * p + global_k
                    for c in range(7):
                        V[global_v_row, col_offset + c] = V_str[local_b * 2 + local_k, c]
            
            for local_k in range(2):
                for local_i in range(2):
                    global_k = k_start + local_k
                    global_i = i_start + local_i
                    global_w_row = global_k * n + global_i
                    for c in range(7):
                        W[global_w_row, col_offset + c] = W_str[local_k * 2 + local_i, c]
            
            col_offset += 7
        else:
            for local_i in range(i_size):
                for local_a in range(j_size):
                    for local_k in range(k_size):
                        global_i = i_start + local_i
                        global_a = j_start + local_a
                        global_k = k_start + local_k
                        
                        global_u_row = global_i * m + global_a
                        global_v_row = global_a * p + global_k
                        global_w_row = global_k * n + global_i
                        
                        U[global_u_row, col_offset] = 1
                        V[global_v_row, col_offset] = 1
                        W[global_w_row, col_offset] = 1
                        col_offset += 1
    
    return U, V, W, total_rank


def _try_direct_strassen_blocks(n, m, p, U_str, V_str, W_str):
    """
    Try to find the best placement of disjoint 2x2x2 Strassen blocks.
    Returns (best_rank, best_U, best_V, best_W, valid)
    """
    best_rank = n * m * p  # Schoolbook fallback
    best_U, best_V, best_W = None, None, None
    valid = True
    
    # For each dimension, we can choose to cover some 2x2 blocks with Strassen
    # The key insight: if we use a Strassen block covering rows [i, i+1], cols [j, j+1], k [k, k+1],
    # then those elements in U, V, W cannot be used by other blocks.
    # So we need to find a set of non-overlapping 2x2x2 blocks.
    
    # Generate all possible Strassen block placements
    I_options = [range(0, n - 1), range(0, n)]  # Start positions for size-2 blocks or size-1
    # Actually, let's think differently.
    
    # A Strassen block uses:
    # U: rows (i,m),(i+1,m),(i,m+1),(i+1,m+1) for i in 0..n-2, m in 0..n-2
    # V: rows (m,p),(m+1,p),(m,p+1),(m+1,p+1) for m in 0..n-2, p in 0..n-2
    # W: rows (k,n),(k+1,n),(k,n+1),(k+1,n+1) for k in 0..n-2, n in 0..n-2
    
    # Let's enumerate all possible sets of non-overlapping Strassen blocks.
    # A block is identified by (i, j, k) where it covers rows i,i+1; cols j,j+1; k's k,k+1.
    # Two blocks overlap if they share any of the same (i,j), (j,k), or (k,i) indices.
    
    # For n=3, m=5, p=7:
    # Possible i: 0, 1
    # Possible j: 0, 1, 2, 3
    # Possible k: 0, 1, 2, 3, 4, 5
    
    # Total possible blocks: 2 * 4 * 6 = 48
    # But we need non-overlapping sets.
    
    # Let's use a greedy approach with backtracking, or just enumerate subsets.
    # Given the small sizes, let's try to find the maximum number of non-overlapping Strassen blocks.
    
    # Actually, let's think about this as a maximum matching problem or set packing.
    # For now, let's try a simpler approach: try all combinations of blocks.
    
    # Generate all possible Strassen blocks
    all_blocks = []
    for i in range(n - 1):
        for j in range(m - 1):
            for k in range(p - 1):
                all_blocks.append((i, j, k))
    
    # Try all subsets of these blocks, checking for non-overlap
    # This is exponential, but n-1=2, m-1=4, p-1=6, so 48 blocks.
    # Too many to enumerate all subsets.
    
    # Instead, let's use a greedy algorithm: sort blocks by some heuristic and add them if non-overlapping.
    
    def check_overlap(block1, block2):
        """Check if two Strassen blocks overlap."""
        i1, j1, k1 = block1
        i2, j2, k2 = block2
        
        # Overlap in I dimension
        if (i1 == i2 or i1 == i2 + 1 or i2 == i1 + 1 or i1 == i2) and (
            (j1 == j2 or j1 == j2 + 1 or j2 == j1 + 1 or j1 == j2) and
            (k1 == k2 or k1 == k2 + 1 or k2 == k1 + 1 or k1 == k2)
        ):
            # More carefully: blocks overlap if they share any of the same index pairs
            # Block 1 uses I indices: {i1, i1+1}, J indices: {j1, j1+1}, K indices: {k1, k1+1}
            # Block 2 uses I indices: {i2, i2+1}, J indices: {j2, j2+1}, K indices: {k2, k2+1}
            # They overlap if:
            # - They share an I-J pair: (i1, j1), (i1, j1+1), (i1+1, j1), (i1+1, j1+1)
            #   intersect with (i2, j2), (i2, j2+1), (i2+1, j2), (i2+1, j2+1)
            # - They share a J-K pair
            # - They share a K-I pair
            
            I1 = {i1, i1+1}
            I2 = {i2, i2+1}
            J1 = {j1, j1+1}
            J2 = {j2, j2+1}
            K1 = {k1, k1+1}
            K2 = {k2, k2+1}
            
            # Check I-J overlap
            if (I1 & I2) and (J1 & J2):
                return True
            # Check J-K overlap
            if (J1 & J2) and (K1 & K2):
                return True
            # Check K-I overlap
            if (K1 & K2) and (I1 & I2):
                return True
            
            return False
        
        return False
    
    # Greedy: try all blocks in some order, add if non-overlapping with already selected
    # Try multiple orderings to find a good set
    best_set = []
    best_rank = n * m * p
    
    # Try sorting by different criteria
    import random
    random.seed(42)
    
    for trial in range(100):
        # Shuffle blocks
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        
        selected = []
        for block in shuffled:
            # Check if this block overlaps with any already selected
            overlaps = False
            for sel in selected:
                if check_overlap(block, sel):
                    overlaps = True
                    break
            if not overlaps:
                selected.append(block)
        
        # Calculate rank for this set of Strassen blocks
        # Start with schoolbook rank
        rank = n * m * p
        # For each Strassen block, we save 8 - 7 = 1 rank (since schoolbook for 2x2x2 is 8, Strassen is 7)
        # But wait, the blocks are not independent if they overlap in the decomposition structure.
        # Actually, if blocks are non-overlapping in the sense above, then each saves 1.
        # So rank = n*m*p - len(selected)
        
        rank = n * m * p - len(selected)
        
        if rank < best_rank:
            best_rank = rank
            best_set = selected
    
    # Now build the decomposition for the best set
    if best_set and best_rank < n * m * p:
        U = np.zeros((n * m, best_rank))
        V = np.zeros((m * p, best_rank))
        W = np.zeros((p * n, best_rank))
        
        # We need to assign columns to Strassen blocks and schoolbook elements.
        # Let's first place all Strassen blocks, then fill in the remaining schoolbook elements.
        
        col_offset = 0
        used_U = set()  # (i, a) pairs used by Strassen
        used_V = set()  # (b, k) pairs used by Strassen
        used_W = set()  # (k, i) pairs used by Strassen
        
        for (i, j, k) in best_set:
            # Place Strassen block
            for local_i in range(2):
                for local_a in range(2):
                    global_i = i + local_i
                    global_a = j + local_a
                    global_u_row = global_i * m + global_a
                    for c in range(7):
                        U[global_u_row, col_offset + c] = U_str[local_i * 2 + local_a, c]
                    used_U.add((global_i, global_a))
            
            for local_b in range(2):
                for local_k in range(2):
                    global_b = j + local_b
                    global_k = k + local_k
                    global_v_row = global_b * p + global_k
                    for c in range(7):
                        V[global_v_row, col_offset + c] = V_str[local_b * 2 + local_k, c]
                    used_V.add((global_b, global_k))
            
            for local_k in range(2):
                for local_i in range(2):
                    global_k = k + local_k
                    global_i = i + local_i
                    global_w_row = global_k * n + global_i
                    for c in range(7):
                        W[global_w_row, col_offset + c] = W_str[local_k * 2 + local_i, c]
                    used_W.add((global_k, global_i))
            
            col_offset += 7
        
        # Now add schoolbook elements for all (i, a, b, k) not covered by Strassen
        for i in range(n):
            for a in range(m):
                for b in range(m):
                    for k in range(p):
                        # Check if this element is covered by any Strassen block
                        covered = False
                        for (si, sj, sk) in best_set:
                            if si <= i <= si + 1 and sj <= a <= sj + 1 and sk <= k <= sk + 1:
                                covered = True
                                break
                        if not covered:
                            U[i * m + a, col_offset] = 1
                            V[b * p + k, col_offset] = 1
                            W[k * n + i, col_offset] = 1
                            col_offset += 1
        
        # Verify
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return best_rank, U, V, W, True
        else:
            return best_rank, U, V, W, False
    
    return best_rank, None, None, None, False


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication
    by searching over block partition strategies and direct Strassen block placement.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Precompute Strassen factors
    U_str, V_str, W_str = _strassen_2x2x2_factors()
    
    # Method 1: Block partition search
    I_partitions = _generate_partitions(n)
    J_partitions = _generate_partitions(m)
    K_partitions = _generate_partitions(p)
    
    best_rank = n * m * p  # Schoolbook fallback
    best_U, best_V, best_W = None, None, None
    
    for I_blocks in I_partitions:
        if time.time() > deadline:
            break
        for J_blocks in J_partitions:
            if time.time() > deadline:
                break
            for K_blocks in K_partitions:
                if time.time() > deadline:
                    break
                
                U, V, W, rank = _build_decomposition(I_blocks, J_blocks, K_blocks, U_str, V_str, W_str)
                
                if rank < best_rank:
                    valid, max_err = _verify_decomposition(U, V, W, n, m, p)
                    if valid:
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
    
    # Method 2: Direct Strassen block placement
    if time.time() < deadline:
        rank2, U2, V2, W2, valid2 = _try_direct_strassen_blocks(n, m, p, U_str, V_str, W_str)
        if valid2 and rank2 < best_rank:
            best_rank = rank2
            best_U, best_V, best_W = U2, V2, W2
    
    # If no valid decomposition found, fall back to schoolbook
    if best_U is None:
        rank = n * m * p
        best_U = np.zeros((n * m, rank))
        best_V = np.zeros((m * p, rank))
        best_W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1
                    best_V[j * p + k, col] = 1
                    best_W[k * n + i, col] = 1
                    col += 1
    
    # Convert to lists with integer/half-integer entries
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                rounded = round(val)
                if abs(val - rounded) < 1e-9:
                    row_list.append(int(rounded))
                else:
                    half_rounded = round(val * 2) / 2
                    row_list.append(half_rounded)
            result.append(row_list)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
