"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Convert to numpy for faster verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k]) 
                # Wait, the tensor is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # So it's element-wise product sum
                s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrices, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a+d)(e+h) -> C00(+1), C11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (c+d)*e -> C11(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[3][1] = -1
    
    # p3 = a*(f-h) -> C11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[3][2] = 1
    
    # p4 = d*(g-e) -> C00(-1), C10(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # p5 = (a+b)*h -> C01(+1), C10(+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[2][4] = 1
    
    # p6 = (c-a)*(e+f) -> C00(+1), C11(+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[3][5] = 1
    
    # p7 = (b-d)*(g+h) -> C00(+1), C01(+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _get_block_decomp(n, m, p):
    """
    Returns the best known decomposition for a block of size n x m * m x p.
    Uses Strassen if possible, otherwise schoolbook.
    """
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W
    else:
        return _schoolbook(n, m, p)["U"], _schoolbook(n, m, p)["V"], _schoolbook(n, m, p)["W"]

def _combine_blocks(n, m, p, split_m, splits_n, splits_p):
    """
    Combines decompositions of blocks defined by splitting dimensions.
    n, m, p are total dimensions.
    split_m: list of sizes for middle dimension summing to m.
    splits_n: list of lists. splits_n[i] is list of sizes for dim n for block i.
    splits_p: list of lists. splits_p[i] is list of sizes for dim p for block i.
    """
    U_total = []
    V_total = []
    W_total = []
    
    # We need to map global indices to local indices for each block.
    # Global U index: i*m + j
    # Global V index: j*p + k
    # Global W index: k*n + i
    
    # Pre-allocate full size matrices with zeros
    # We don't know total rank yet, so we build lists of (global_idx, local_val, local_block_idx)
    # Actually, easier: create empty lists for columns, then flatten.
    
    col_blocks = []
    
    for b_idx, m_size in enumerate(split_m):
        n_sizes = splits_n[b_idx]
        p_sizes = splits_p[b_idx]
        
        if sum(n_sizes) != n or sum(p_sizes) != p:
            return None # Invalid split
        
        # For each block, we have a sub-problem n_size x m_size * m_size x p_size
        # We need to decompose this.
        # We iterate over all sub-blocks in this m-slice?
        # Actually, the split_m defines the j-range for this block.
        # The splits_n and splits_p define the i-range and k-range.
        
        # Let's assume a simple partition:
        # Block b covers j in [sum(split_m[:b]), sum(split_m[:b])+m_size)
        # And for this block, we partition i and k according to n_sizes and p_sizes.
        
        j_start = sum(split_m[:b_idx])
        
        # Iterate through the sub-blocks defined by n_sizes and p_sizes
        # This is a bit complex. Let's simplify: 
        # We just treat each (n_size, m_size, p_size) combination as a block.
        # But n_sizes and p_sizes are lists. This implies a further partition?
        # Let's stick to a simpler partition strategy:
        # Partition m into chunks. For each chunk, we have a full n x p product?
        # No, that's just schoolbook split.
        
        # Let's try a specific partition:
        # 1. Split m into 2 parts: m1, m2.
        # 2. For part 1, use Strassen if n=2, p=2? No, n and p are fixed.
        # 
        # Let's implement a generic "Block Diagonal" combination.
        # We define a set of blocks. Each block is defined by:
        # i_range, j_range, k_range.
        # The tensor is the sum over blocks of the tensor for that block.
        # This requires that the blocks partition the index space completely.
        
        pass # This function structure is getting complicated. Let's use a direct builder.

def _build_decomposition(n, m, p, strategy):
    """
    Builds a decomposition based on a strategy string.
    Strategies:
    - 'schoolbook'
    - 'strassen_2x2x2_embedded': Tries to embed Strassen blocks.
    """
    if strategy == 'schoolbook':
        return _schoolbook(n, m, p)
    
    # Strategy: Partition the middle dimension m into two halves of size 2 (since m=4).
    # A = [A1 | A2] where A1 is n x 2, A2 is n x 2.
    # B = [B1; B2] where B1 is 2 x p, B2 is 2 x p.
    # C = A1 B1 + A2 B2.
    
    # We can decompose A1 B1 and A2 B2 separately.
    # If n=3, p=7, these are 3x2 * 2x7.
    # Rank of 3x2 * 2x7 is at most 42 (schoolbook).
    # Total rank 84. No gain.
    
    # Can we do better for 3x2 * 2x7?
    # Lower bound max(6, 14, 21) = 21.
    # Is there a known algorithm for 3x2 * 2x7 with rank < 42?
    # Likely not standard.
    
    # What if we partition differently?
    # Split n=3 into 1+2?
    # Split p=7 into 1+6?
    
    # Let's try a "Greedy Block" approach.
    # We want to cover the 3x4x7 tensor with rectangular blocks.
    # Each block is a matrix multiplication tensor of size n_i x m_j * m_j x p_k.
    # The blocks must partition the index sets {i}, {j}, {k}.
    # i in [0,3), j in [0,4), k in [0,7).
    
    # Possible partitions of 3: [3], [1,2], [1,1,1], [2,1]
    # Possible partitions of 4: [4], [1,3], [2,2], [3,1], [1,1,2], etc.
    # Possible partitions of 7: [7], [1,6], [2,5], [3,4], etc.
    
    # For each combination of partitions, we calculate the sum of ranks.
    # For a block (ni, mj, pk), we use the best known rank.
    # For now, only 2x2x2 has rank 7. Others use ni*mj*pk.
    
    def get_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        return ni * mj * pk
    
    best_rank = n * m * p
    best_config = None
    
    # Generate partitions
    def get_partitions(n):
        res = []
        def helper(rem, curr):
            if rem == 0:
                res.append(curr[:])
                return
            for x in range(1, rem + 1):
                curr.append(x)
                helper(rem - x, curr)
                curr.pop()
        helper(n, [])
        return res
    
    part_n = get_partitions(n)
    part_m = get_partitions(m)
    part_p = get_partitions(p)
    
    # To keep search space small, limit partition lengths or just try a few.
    # Actually, partitions of 3, 4, 7 are small enough to enumerate all.
    # 3: 4 partitions
    # 4: 8 partitions
    # 7: 16 partitions
    # Total combos: 4 * 8 * 16 = 512. Very fast.
    
    for pn in part_n:
        for pm in part_m:
            for pp in part_p:
                # The number of blocks is determined by the finest partition?
                # No, a block is defined by one part from each partition.
                # But the partitions must be aligned?
                # If we split i into [1,2], j into [2,2], k into [3,4].
                # Block 1: i[0:1], j[0:2], k[0:3] -> 1x2x3
                # Block 2: i[0:1], j[2:4], k[0:3] -> 1x2x3
                # Block 3: i[1:3], j[0:2], k[0:3] -> 2x2x3
                # Block 4: i[1:3], j[2:4], k[0:3] -> 2x2x3
                # ... and so on for k[3:7].
                
                # This creates a grid of blocks.
                # We sum the ranks of all blocks in the grid.
                
                total_rank = 0
                # Iterate through the grid
                i_start = 0
                j_start = 0
                k_start = 0
                
                # We need to iterate through the combinations of parts.
                # The number of blocks is len(pn) * len(pm) * len(pp).
                # Wait, the blocks must tile the space.
                # If pn=[1,2], pm=[2,2], pp=[3,4].
                # Blocks:
                # (1,2,3), (1,2,4)
                # (2,2,3), (2,2,4)
                
                for ni in pn:
                    for mj in pm:
                        for pk in pp:
                            total_rank += get_rank(ni, mj, pk)
                
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_config = (pn, pm, pp)
    
    if best_config is None:
        return _schoolbook(n, m, p)
    
    pn, pm, pp = best_config
    
    # Now construct the actual matrices U, V, W for this configuration.
    # We will build column by column.
    # Each block contributes its own decomposition.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    i_offset = 0
    for ni in pn:
        j_offset = 0
        for mj in pm:
            k_offset = 0
            for pk in pp:
                # Get decomposition for this block
                U_b, V_b, W_b = _get_block_decomp(ni, mj, pk)
                rank_b = len(U_b[0])
                
                for r in range(rank_b):
                    # Create full-length vectors with zeros
                    u_vec = [0.0] * (n * m)
                    v_vec = [0.0] * (m * p)
                    w_vec = [0.0] * (p * n)
                    
                    # Fill in the block values
                    for li in range(ni):
                        for lj in range(mj):
                            # Global index in U: (i_offset + li) * m + (j_offset + lj)
                            u_vec[(i_offset + li) * m + (j_offset + lj)] = U_b[li * mj + lj][r]
                    
                    for lj in range(mj):
                        for lk in range(pk):
                            # Global index in V: (j_offset + lj) * p + (k_offset + lk)
                            v_vec[(j_offset + lj) * p + (k_offset + lk)] = V_b[lj * pk + lk][r]
                    
                    for lk in range(pk):
                        for li in range(ni):
                            # Global index in W: (k_offset + lk) * n + (i_offset + li)
                            w_vec[(k_offset + lk) * n + (i_offset + li)] = W_b[lk * ni + li][r]
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                
                k_offset += pk
            j_offset += mj
        i_offset += ni
    
    # Transpose to get matrices of shape (dim, rank)
    rank = len(U_cols)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    assert _verify(best["U"], best["V"], best["W"], n, m, p), "Schoolbook verification failed"
    
    # Try structured block decomposition
    # This explores all partitions of n, m, p and sums the ranks of the resulting blocks.
    # If any partition yields a total rank < 84, we use it.
    
    if time.time() - t0 < DEADLINE - 10:
        candidate = _build_decomposition(n, m, p, 'blocks')
        if candidate is not None:
            # Verify the candidate
            if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                cand_rank = len(candidate["U"][0])
                if cand_rank < best_rank:
                    best = candidate
                    best_rank = cand_rank
    
    return best
# EVOLVE-BLOCK-END
