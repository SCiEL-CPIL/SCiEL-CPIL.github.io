import time
import itertools

DIMENSIONS = [3, 5, 7]

def get_partitions(n, max_size=2):
    """Generate all partitions of n into parts of size 1 or max_size."""
    results = []
    def helper(remaining, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for size in range(1, min(max_size, remaining) + 1):
            helper(remaining - size, current + [size])
    helper(n, [])
    return results

def block_rank(a, b, c):
    """Rank of a x b x c multiplication tensor.
    For 2x2x2, use Strassen (rank 7). Otherwise, schoolbook (a*b*c)."""
    if a == 2 and b == 2 and c == 2:
        return 7
    return a * b * c

def compute_total_rank(part_i, part_j, part_k):
    """Compute total rank given partitions of i, j, k indices."""
    total = 0
    for a in part_i:
        for b in part_j:
            for c in part_k:
                total += block_rank(a, b, c)
    return total

def find_optimal_partition(n, m, p):
    """Find the partition that minimizes total rank."""
    parts_n = get_partitions(n)
    parts_m = get_partitions(m)
    parts_p = get_partitions(p)
    
    best_rank = None
    best_partition = None
    for pn in parts_n:
        for pm in parts_m:
            for pp in parts_p:
                rank = compute_total_rank(pn, pm, pp)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_partition = (pn, pm, pp)
    return best_rank, best_partition

def construct_strassen_decomposition(part_i, part_j, part_k):
    """Construct the U, V, W matrices given the optimal block partition.
    
    For each block triple (I, J, K), we generate rank-1 terms.
    - If the block is 2x2x2, use Strassen (7 terms).
    - Otherwise, use schoolbook (a*b*c terms).
    
    We need to track the global indices.
    """
    n, m, p = 3, 5, 7
    
    # Determine the actual index ranges for each block
    # part_i is a tuple of block sizes, e.g., (2, 1) means first 2 i's, then 1 i
    i_blocks = []
    start = 0
    for size in part_i:
        i_blocks.append(list(range(start, start + size)))
        start += size
    
    j_blocks = []
    start = 0
    for size in part_j:
        j_blocks.append(list(range(start, start + size)))
        start += size
    
    k_blocks = []
    start = 0
    for size in part_k:
        k_blocks.append(list(range(start, start + size)))
        start += size
    
    # Compute total rank
    total_rank = 0
    for I in i_blocks:
        for J in j_blocks:
            for K in k_blocks:
                total_rank += block_rank(len(I), len(J), len(K))
    
    # Initialize U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for I in i_blocks:
        for J in j_blocks:
            for K in k_blocks:
                a, b, c = len(I), len(J), len(K)
                if a == 2 and b == 2 and c == 2:
                    # Strassen for 2x2x2
                    # I = [i0, i1], J = [j0, j1], K = [k0, k1]
                    i0, i1 = I
                    j0, j1 = J
                    k0, k1 = K
                    
                    strassen_terms = [
                        {
                            'U': {(0,0): 1, (1,1): 1},
                            'V': {(0,0): 1, (1,1): 1},
                            'W': {(0,0): 1, (1,1): 1}
                        },
                        {
                            'U': {(1,0): 1, (1,1): 1},
                            'V': {(1,1): 1},
                            'W': {(1,0): 1, (1,1): -1}
                        },
                        {
                            'U': {(0,0): 1},
                            'V': {(0,1): 1, (1,0): -1},
                            'W': {(0,1): 1, (1,1): 1}
                        },
                        {
                            'U': {(1,1): 1},
                            'V': {(1,1): 1, (0,1): -1},
                            'W': {(0,0): 1, (1,0): 1}
                        },
                        {
                            'U': {(0,0): 1, (0,1): 1},
                            'V': {(1,1): 1},
                            'W': {(0,0): -1, (0,1): 1}
                        },
                        {
                            'U': {(1,0): 1, (0,0): -1},
                            'V': {(0,0): 1},
                            'W': {(1,1): 1}
                        },
                        {
                            'U': {(0,1): 1, (1,1): -1},
                            'V': {(0,0): 1, (1,0): 1},
                            'W': {(0,0): 1}
                        },
                    ]
                    
                    for term in strassen_terms:
                        # Add U contributions
                        for (ii, jj), coeff in term['U'].items():
                            global_i = I[ii]
                            global_j = J[jj]
                            U[global_i * m + global_j][col] += coeff
                        # Add V contributions
                        for (jj, kk), coeff in term['V'].items():
                            global_j = J[jj]
                            global_k = K[kk]
                            V[global_j * p + global_k][col] += coeff
                        # Add W contributions
                        for (kk, ii), coeff in term['W'].items():
                            global_k = K[kk]
                            global_i = I[ii]
                            W[global_k * n + global_i][col] += coeff
                        col += 1
                else:
                    # Schoolbook for this block
                    for ii in range(a):
                        for jj in range(b):
                            for kk in range(c):
                                global_i = I[ii]
                                global_j = J[jj]
                                global_k = K[kk]
                                U[global_i * m + global_j][col] = 1
                                V[global_j * p + global_k][col] = 1
                                W[global_k * n + global_i][col] = 1
                                col += 1
    
    # Verify col == total_rank
    assert col == total_rank, f"col={col}, total_rank={total_rank}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Use the known optimal partition for 3x5x7
    # n=3: (2, 1)
    # m=5: (2, 2, 1)
    # p=7: (2, 2, 2, 1)
    part_i = (2, 1)
    part_j = (2, 2, 1)
    part_k = (2, 2, 2, 1)
    
    # Construct the decomposition
    result = construct_strassen_decomposition(part_i, part_j, part_k)
    
    # Verify the decomposition
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    
    # Check that the decomposition is valid
    # For each (i, j, k), the coefficient in the tensor should be 1
    # The tensor coefficient at (i, j, k) is sum_{c} U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
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
    
    return result
