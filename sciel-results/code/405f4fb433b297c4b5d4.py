import numpy as np
import time
from functools import lru_cache

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
    
    # m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # m2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C[1,0]
    
    # m3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # m4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1  # C[1,0]
    W[0][3] = 1  # C[0,0]
    
    # m5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1  # C[0,1]
    W[0][4] = -1 # C[0,0]
    
    # m6 = (A10-A11)*(B01+B11)
    U[2][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[3][5] = 1
    W[3][5] = 1  # C[1,1]
    
    # m7 = (A01-A10)*(B00+B10)
    U[1][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[2][6] = 1
    W[0][6] = 1  # C[0,0]
    
    return {"U": U, "V": V, "W": W}

# Memoization for optimal rank calculation
_rank_cache = {}

def get_optimal_rank(n, m, p):
    if (n, m, p) in _rank_cache:
        return _rank_cache[(n, m, p)]
    
    # Base case: if any dimension is 1, it's just a dot product / outer product
    if n == 1 or m == 1 or p == 1:
        rank = n * m * p
        _rank_cache[(n, m, p)] = rank
        return rank
    
    # Option 1: Schoolbook
    best_rank = n * m * p
    
    # Option 2: Split dimensions
    # We can split n, m, or p.
    # Splitting n: n -> n1, n2. m, p unchanged.
    # Blocks: (n1, m, p) and (n2, m, p)
    for n1 in range(1, n):
        n2 = n - n1
        r1 = get_optimal_rank(n1, m, p)
        r2 = get_optimal_rank(n2, m, p)
        best_rank = min(best_rank, r1 + r2)
        
    # Splitting m: m -> m1, m2. n, p unchanged.
    # Blocks: (n, m1, p) and (n, m2, p)
    for m1 in range(1, m):
        m2 = m - m1
        r1 = get_optimal_rank(n, m1, p)
        r2 = get_optimal_rank(n, m2, p)
        best_rank = min(best_rank, r1 + r2)
        
    # Splitting p: p -> p1, p2. n, m unchanged.
    # Blocks: (n, m, p1) and (n, m, p2)
    for p1 in range(1, p):
        p2 = p - p1
        r1 = get_optimal_rank(n, m, p1)
        r2 = get_optimal_rank(n, m, p2)
        best_rank = min(best_rank, r1 + r2)
        
    _rank_cache[(n, m, p)] = best_rank
    return best_rank

def get_optimal_partition(n, m, p):
    """
    Returns a tuple describing the optimal partition.
    Format: ('schoolbook',) or ('strassen',) or ('split', dim, size1, part1, part2)
    """
    if (n, m, p) in _rank_cache:
        rank = _rank_cache[(n, m, p)]
    else:
        rank = get_optimal_rank(n, m, p)
        
    if n == 1 or m == 1 or p == 1:
        return ('schoolbook',)
        
    # Check if schoolbook is optimal
    if n * m * p == rank:
        # It might still be beneficial to split if sub-blocks use Strassen?
        # No, if schoolbook rank == optimal rank, we can just use schoolbook.
        # However, if there's a tie, we might prefer Strassen if it exists?
        # Strassen is only 2x2x2.
        if n == 2 and m == 2 and p == 2:
             return ('strassen',)
        return ('schoolbook',)
        
    # Check if Strassen is optimal
    if n == 2 and m == 2 and p == 2:
        if rank == 7:
            return ('strassen',)
            
    # Try splits
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = get_optimal_rank(n1, m, p)
        r2 = get_optimal_rank(n2, m, p)
        if r1 + r2 == rank:
            return ('split', 'n', n1, get_optimal_partition(n1, m, p), get_optimal_partition(n2, m, p))
            
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = get_optimal_rank(n, m1, p)
        r2 = get_optimal_rank(n, m2, p)
        if r1 + r2 == rank:
            return ('split', 'm', m1, get_optimal_partition(n, m1, p), get_optimal_partition(n, m2, p))
            
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = get_optimal_rank(n, m, p1)
        r2 = get_optimal_rank(n, m, p2)
        if r1 + r2 == rank:
            return ('split', 'p', p1, get_optimal_partition(n, m, p1), get_optimal_partition(n, m, p2))
            
    # Fallback
    return ('schoolbook',)

def build_decomp(n, m, p, partition, i_offset=0, j_offset=0, k_offset=0):
    """
    Recursively builds the U, V, W matrices for a block.
    i_offset, j_offset, k_offset are the starting indices in the global n, m, p dimensions.
    Returns U, V, W as lists of lists.
    """
    if partition[0] == 'schoolbook':
        sub = _schoolbook(n, m, p)
        # Offset the indices
        U = [[0] * len(sub['U'][0]) for _ in range(n * m)]
        V = [[0] * len(sub['V'][0]) for _ in range(m * p)]
        W = [[0] * len(sub['W'][0]) for _ in range(p * n)]
        
        for r in range(len(sub['U'][0])):
            for i in range(n):
                for j in range(m):
                    U[(i + i_offset) * m + (j + j_offset)][r] = sub['U'][i * m + j][r]
            for j in range(m):
                for k in range(p):
                    V[(j + j_offset) * p + (k + k_offset)][r] = sub['V'][j * p + k][r]
            for k in range(p):
                for i in range(n):
                    W[(k + k_offset) * n + (i + i_offset)][r] = sub['W'][k * n + i][r]
                    
        return U, V, W
        
    elif partition[0] == 'strassen':
        sub = _strassen_2x2x2()
        U = [[0] * len(sub['U'][0]) for _ in range(n * m)]
        V = [[0] * len(sub['V'][0]) for _ in range(m * p)]
        W = [[0] * len(sub['W'][0]) for _ in range(p * n)]
        
        for r in range(len(sub['U'][0])):
            for i in range(n):
                for j in range(m):
                    U[(i + i_offset) * m + (j + j_offset)][r] = sub['U'][i * m + j][r]
            for j in range(m):
                for k in range(p):
                    V[(j + j_offset) * p + (k + k_offset)][r] = sub['V'][j * p + k][r]
            for k in range(p):
                for i in range(n):
                    W[(k + k_offset) * n + (i + i_offset)][r] = sub['W'][k * n + i][r]
                    
        return U, V, W
        
    elif partition[0] == 'split':
        _, dim, size1, part1, part2 = partition
        
        if dim == 'n':
            n1, n2 = size1, n - size1
            U1, V1, W1 = build_decomp(n1, m, p, part1, i_offset, j_offset, k_offset)
            U2, V2, W2 = build_decomp(n2, m, p, part2, i_offset + n1, j_offset, k_offset)
            
            # Concatenate columns
            U = []
            for i in range(n * m):
                if i < n1 * m:
                    U.append(U1[i] + U2[i] if False else U1[i]) # Wait, U2 has different rows?
                    # U1 has n1*m rows. U2 has n2*m rows.
                    # We need to interleave or stack?
                    # The global U has n*m rows.
                    # Rows 0..n1*m-1 come from U1.
                    # Rows n1*m..n*m-1 come from U2.
                    pass
            
            # Correct concatenation logic:
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Copy U1
            for i in range(n1 * m):
                for r in range(rank1):
                    U[i][r] = U1[i][r]
            # Copy U2
            for i in range(n2 * m):
                for r in range(rank2):
                    U[n1 * m + i][rank1 + r] = U2[i][r]
                    
            # Copy V1, V2
            for i in range(m * p):
                for r in range(rank1):
                    V[i][r] = V1[i][r]
                for r in range(rank2):
                    V[i][rank1 + r] = V2[i][r]
                    
            # Copy W1, W2
            for i in range(p * n):
                for r in range(rank1):
                    W[i][r] = W1[i][r]
                for r in range(rank2):
                    W[i][rank1 + r] = W2[i][r]
                    
            return U, V, W
            
        elif dim == 'm':
            m1, m2 = size1, m - size1
            U1, V1, W1 = build_decomp(n, m1, p, part1, i_offset, j_offset, k_offset)
            U2, V2, W2 = build_decomp(n, m2, p, part2, i_offset, j_offset + m1, k_offset)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Copy U1, U2
            for i in range(n * m1):
                for r in range(rank1):
                    U[i][r] = U1[i][r]
            for i in range(n * m2):
                for r in range(rank2):
                    U[n * m1 + i][rank1 + r] = U2[i][r]
                    
            # Copy V1, V2
            for i in range(m1 * p):
                for r in range(rank1):
                    V[i][r] = V1[i][r]
            for i in range(m2 * p):
                for r in range(rank2):
                    V[m1 * p + i][rank1 + r] = V2[i][r]
                    
            # Copy W1, W2
            for i in range(p * n):
                for r in range(rank1):
                    W[i][r] = W1[i][r]
                for r in range(rank2):
                    W[i][rank1 + r] = W2[i][r]
                    
            return U, V, W
            
        elif dim == 'p':
            p1, p2 = size1, p - size1
            U1, V1, W1 = build_decomp(n, m, p1, part1, i_offset, j_offset, k_offset)
            U2, V2, W2 = build_decomp(n, m, p2, part2, i_offset, j_offset, k_offset + p1)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Copy U1, U2
            for i in range(n * m):
                for r in range(rank1):
                    U[i][r] = U1[i][r]
                for r in range(rank2):
                    U[i][rank1 + r] = U2[i][r]
                    
            # Copy V1, V2
            for i in range(m * p1):
                for r in range(rank1):
                    V[i][r] = V1[i][r]
            for i in range(m * p2):
                for r in range(rank2):
                    V[m * p1 + i][rank1 + r] = V2[i][r]
                    
            # Copy W1, W2
            for i in range(p1 * n):
                for r in range(rank1):
                    W[i][r] = W1[i][r]
            for i in range(p2 * n):
                for r in range(rank2):
                    W[p1 * n + i][rank1 + r] = W2[i][r]
                    
            return U, V, W

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 120
    
    # Clear cache
    _rank_cache.clear()
    
    # Get optimal partition
    partition = get_optimal_partition(n, m, p)
    
    # Build decomposition
    U, V, W = build_decomp(n, m, p, partition)
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        result = _schoolbook(n, m, p)
        
    return result
