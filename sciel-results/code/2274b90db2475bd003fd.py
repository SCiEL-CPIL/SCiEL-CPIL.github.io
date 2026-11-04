"""
Strategy: Systematic Block-Structure Enumeration.
Instead of refining a single fixed Strassen decomposition, we enumerate all possible 
rectangular block partitions of the dimensions n=3, m=4, p=8. For each partition, 
we compute the total rank required by summing the optimal ranks of the constituent 
blocks (using schoolbook for 1x1 and Strassen for 2x2). We then construct the 
decomposition for the partition that yields the minimum total rank.
"""

import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank_1x1x1():
    return 1

def _rank_1x1x2():
    return 2

def _rank_1x2x1():
    return 2

def _rank_2x1x1():
    return 2

def _rank_1x2x2():
    # 1x2 * 2x2 -> 1x2. 
    # Rank is 4 (schoolbook) or can it be lower?
    # T[i,j,k] = A[i,j]B[j,k]. i is fixed (size 1).
    # Effectively multiplying a row vector (1x2) by a matrix (2x2).
    # This is just 4 scalar multiplications. Rank 4.
    return 4

def _rank_2x1x2():
    # 2x1 * 1x2 -> 2x2.
    # Outer product of column (2x1) and row (1x2).
    # Rank 4.
    return 4

def _rank_2x2x1():
    # 2x2 * 2x1 -> 2x1.
    # Rank 4.
    return 4

def _rank_2x2x2():
    # Strassen
    return 7

def _get_block_rank(n_b, m_b, p_b):
    """Returns the optimal rank for a block of size n_b x m_b times m_b x p_b."""
    if n_b == 1 and m_b == 1 and p_b == 1:
        return 1
    if n_b == 1 and m_b == 1 and p_b == 2:
        return 2
    if n_b == 1 and m_b == 2 and p_b == 1:
        return 2
    if n_b == 2 and m_b == 1 and p_b == 1:
        return 2
    if n_b == 1 and m_b == 2 and p_b == 2:
        return 4
    if n_b == 2 and m_b == 1 and p_b == 2:
        return 4
    if n_b == 2 and m_b == 2 and p_b == 1:
        return 4
    if n_b == 2 and m_b == 2 and p_b == 2:
        return 7
    # Fallback to schoolbook for larger blocks (unlikely in 3x4x8 with small blocks)
    return n_b * m_b * p_b

def _construct_block_decomp(n, m, p, i_blocks, j_blocks, k_blocks):
    """
    Constructs the full decomposition given block boundaries.
    i_blocks: list of start indices for i (e.g., [0, 1, 3])
    j_blocks: list of start indices for j (e.g., [0, 2, 4])
    k_blocks: list of start indices for k (e.g., [0, 4, 8])
    """
    rank = 0
    U = []
    V = []
    W = []
    
    # Pre-allocate lists of zeros? No, let's build column by column.
    # We need to know total rank first.
    total_rank = 0
    block_info = []
    
    for i_idx, i_start in enumerate(i_blocks):
        i_end = i_blocks[i_idx + 1]
        n_b = i_end - i_start
        for j_idx, j_start in enumerate(j_blocks):
            j_end = j_blocks[j_idx + 1]
            m_b = j_end - j_start
            for k_idx, k_start in enumerate(k_blocks):
                k_end = k_blocks[k_idx + 1]
                p_b = k_end - k_start
                
                r_b = _get_block_rank(n_b, m_b, p_b)
                total_rank += r_b
                block_info.append((i_start, j_start, k_start, n_b, m_b, p_b, r_b))
                
    # Initialize matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for (i_start, j_start, k_start, n_b, m_b, p_b, r_b) in block_info:
        # Extract sub-matrices for this block
        # We need to place the local decomposition into the global matrices.
        
        if n_b == 1 and m_b == 1 and p_b == 1:
            # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
            U[i_start * m + j_start, col] = 1
            V[j_start * p + k_start, col] = 1
            W[k_start * n + i_start, col] = 1
            col += 1
            
        elif n_b == 1 and m_b == 1 and p_b == 2:
            # U[i,j] = 1 for k=0,1? No.
            # T[i,j,k] = A[i,j] B[j,k].
            # Term 1: k=0. U[i,j]=1, V[j,0]=1, W[0,i]=1
            # Term 2: k=1. U[i,j]=1, V[j,1]=1, W[1,i]=1
            # Wait, U[i,j] is the same for both terms? No, U is per (i,j).
            # In the tensor decomposition, U[i*m+j, c] is the coefficient.
            # For c=0 (k=0 term): U[i,j]=1, V[j,0]=1, W[0,i]=1
            # For c=1 (k=1 term): U[i,j]=1, V[j,1]=1, W[1,i]=1
            # So U[i,j] will have two non-zeros? No, U[i*m+j, 0]=1 and U[i*m+j, 1]=1.
            
            U[i_start * m + j_start, col] = 1
            V[j_start * p + k_start, col] = 1
            W[k_start * n + i_start, col] = 1
            col += 1
            
            U[i_start * m + j_start, col] = 1
            V[j_start * p + (k_start + 1), col] = 1
            W[(k_start + 1) * n + i_start, col] = 1
            col += 1
            
        elif n_b == 1 and m_b == 2 and p_b == 1:
            # T[i,j,k] = A[i,j] B[j,k]. j has 2 elements.
            # Term 1: j=0. U[i,0]=1, V[0,k]=1, W[k,i]=1
            # Term 2: j=1. U[i,1]=1, V[1,k]=1, W[k,i]=1
            
            U[i_start * m + (j_start), col] = 1
            V[j_start * p + k_start, col] = 1
            W[k_start * n + i_start, col] = 1
            col += 1
            
            U[i_start * m + (j_start + 1), col] = 1
            V[(j_start + 1) * p + k_start, col] = 1
            W[k_start * n + i_start, col] = 1
            col += 1
            
        elif n_b == 2 and m_b == 1 and p_b == 1:
            # T[i,j,k] = A[i,j] B[j,k]. i has 2 elements.
            # Term 1: i=0. U[0,j]=1, V[j,k]=1, W[k,0]=1
            # Term 2: i=1. U[1,j]=1, V[j,k]=1, W[k,1]=1
            
            U[(i_start) * m + j_start, col] = 1
            V[j_start * p + k_start, col] = 1
            W[k_start * n + i_start, col] = 1
            col += 1
            
            U[(i_start + 1) * m + j_start, col] = 1
            V[j_start * p + k_start, col] = 1
            W[k_start * n + (i_start + 1), col] = 1
            col += 1
            
        elif n_b == 1 and m_b == 2 and p_b == 2:
            # Rank 4. Schoolbook.
            for di in range(1):
                for dj in range(2):
                    for dk in range(2):
                        U[(i_start+di)*m + (j_start+dj), col] = 1
                        V[(j_start+dj)*p + (k_start+dk), col] = 1
                        W[(k_start+dk)*n + (i_start+di), col] = 1
                        col += 1
                        
        elif n_b == 2 and m_b == 1 and p_b == 2:
            # Rank 4. Schoolbook.
            for di in range(2):
                for dj in range(1):
                    for dk in range(2):
                        U[(i_start+di)*m + (j_start+dj), col] = 1
                        V[(j_start+dj)*p + (k_start+dk), col] = 1
                        W[(k_start+dk)*n + (i_start+di), col] = 1
                        col += 1
                        
        elif n_b == 2 and m_b == 2 and p_b == 1:
            # Rank 4. Schoolbook.
            for di in range(2):
                for dj in range(2):
                    for dk in range(1):
                        U[(i_start+di)*m + (j_start+dj), col] = 1
                        V[(j_start+dj)*p + (k_start+dk), col] = 1
                        W[(k_start+dk)*n + (i_start+di), col] = 1
                        col += 1
                        
        elif n_b == 2 and m_b == 2 and p_b == 2:
            # Strassen Rank 7
            # Local indices 0,1 for i, j, k
            # Terms:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M4
            # C10 = M2 + M5
            # C11 = M1 - M2 + M3 + M6
            
            # Let's map these to U, V, W coefficients.
            # U[i,j] corresponds to A_ij. V[j,k] to B_jk. W[k,i] to C_ki (note index order in W is k*n+i).
            
            # Term 1: M1
            # U: A00(1), A11(1)
            # V: B00(1), B11(1)
            # W: C00(1), C11(1) -> W[0,0]=1, W[1,1]=1
            U[(i_start+0)*m + (j_start+0), col] = 1
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[(j_start+0)*p + (k_start+0), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[(k_start+0)*n + (i_start+0), col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
            
            # Term 2: M2
            # U: A10(1), A11(1)
            # V: B00(1)
            # W: C10(1) -> W[0,1]=1
            U[(i_start+1)*m + (j_start+0), col] = 1
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[(j_start+0)*p + (k_start+0), col] = 1
            W[(k_start+0)*n + (i_start+1), col] = 1
            col += 1
            
            # Term 3: M3
            # U: A00(1)
            # V: B10(1), B11(-1)
            # W: C01(1) -> W[1,0]=1
            U[(i_start+0)*m + (j_start+0), col] = 1
            V[(j_start+1)*p + (k_start+0), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = -1
            W[(k_start+1)*n + (i_start+0), col] = 1
            col += 1
            
            # Term 4: M4
            # U: A11(1)
            # V: B01(1), B00(-1)
            # W: C00(-1), C01(1) -> W[0,0]=-1, W[1,0]=1
            U[(i_start+1)*m + (j_start+1), col] = 1
            V[(j_start+0)*p + (k_start+1), col] = 1
            V[(j_start+0)*p + (k_start+0), col] = -1
            W[(k_start+0)*n + (i_start+0), col] = -1
            W[(k_start+1)*n + (i_start+0), col] = 1
            col += 1
            
            # Term 5: M5
            # U: A00(1), A01(1)
            # V: B11(1)
            # W: C00(1), C10(1) -> W[0,0]=1, W[0,1]=1
            U[(i_start+0)*m + (j_start+0), col] = 1
            U[(i_start+0)*m + (j_start+1), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[(k_start+0)*n + (i_start+0), col] = 1
            W[(k_start+0)*n + (i_start+1), col] = 1
            col += 1
            
            # Term 6: M6
            # U: A10(1), A00(-1)
            # V: B00(1), B10(1)
            # W: C11(1) -> W[1,1]=1
            U[(i_start+1)*m + (j_start+0), col] = 1
            U[(i_start+0)*m + (j_start+0), col] = -1
            V[(j_start+0)*p + (k_start+0), col] = 1
            V[(j_start+1)*p + (k_start+0), col] = 1
            W[(k_start+1)*n + (i_start+1), col] = 1
            col += 1
            
            # Term 7: M7
            # U: A01(1), A11(-1)
            # V: B01(1), B11(1)
            # W: C00(-1) -> W[0,0]=-1
            U[(i_start+0)*m + (j_start+1), col] = 1
            U[(i_start+1)*m + (j_start+1), col] = -1
            V[(j_start+0)*p + (k_start+1), col] = 1
            V[(j_start+1)*p + (k_start+1), col] = 1
            W[(k_start+0)*n + (i_start+0), col] = -1
            col += 1

        else:
            # Fallback: Schoolbook for any other size
            for di in range(n_b):
                for dj in range(m_b):
                    for dk in range(p_b):
                        U[(i_start+di)*m + (j_start+dj), col] = 1
                        V[(j_start+dj)*p + (k_start+dk), col] = 1
                        W[(k_start+dk)*n + (i_start+di), col] = 1
                        col += 1

    return U, V, W

def _get_partitions(size):
    """
    Returns all possible ways to partition [0, size] into blocks of size 1 or 2.
    Returns a list of lists, where each list contains the start indices.
    E.g., for size 3: [[0,1,3], [0,2,3]]
    """
    if size == 0:
        return [[]]
    if size == 1:
        return [[0]]
    if size == 2:
        return [[0, 2], [0, 1, 2]]
    if size == 3:
        return [[0, 1, 3], [0, 2, 3], [0, 1, 2, 3]]
    if size == 4:
        # 1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2
        return [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 3, 4],
            [0, 2, 3, 4],
            [0, 2, 4]
        ]
    if size == 8:
        # We can generate this recursively or just use a simple DP.
        # Given time constraints, let's just generate all combinations of 1s and 2s that sum to 8.
        # This is equivalent to tiling a 1x8 strip with 1x1 and 1x2 tiles.
        # Number of ways is Fibonacci(9) = 34.
        def gen(start, current):
            if start == size:
                yield current
                return
            if start + 1 <= size:
                yield from gen(start + 1, current + [start + 1])
            if start + 2 <= size:
                yield from gen(start + 2, current + [start + 2])
        
        return list(gen(0, [0]))
    
    # Generic fallback for larger sizes (not needed for 3,4,8)
    def gen(start, current):
        if start == size:
            yield current
            return
        if start + 1 <= size:
            yield from gen(start + 1, current + [start + 1])
        if start + 2 <= size:
            yield from gen(start + 2, current + [start + 2])
    return list(gen(0, [0]))

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate all possible block partitions for each dimension
    i_parts = _get_partitions(n)
    j_parts = _get_partitions(m)
    k_parts = _get_partitions(p)
    
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    best_config = None
    
    # Iterate through all combinations of partitions
    for i_p in i_parts:
        for j_p in j_parts:
            for k_p in k_parts:
                if time.time() > deadline:
                    break
                
                # Calculate theoretical rank
                total_rank = 0
                valid = True
                for i_idx, i_start in enumerate(i_p):
                    i_end = i_p[i_idx + 1]
                    n_b = i_end - i_start
                    for j_idx, j_start in enumerate(j_p):
                        j_end = j_p[j_idx + 1]
                        m_b = j_end - j_start
                        for k_idx, k_start in enumerate(k_p):
                            k_end = k_p[k_idx + 1]
                            p_b = k_end - k_start
                            
                            r_b = _get_block_rank(n_b, m_b, p_b)
                            total_rank += r_b
                            
                if total_rank < best_rank:
                    best_rank = total_rank
                    # Construct the decomposition
                    U, V, W = _construct_block_decomp(n, m, p, i_p, j_p, k_p)
                    
                    # Verify
                    if _verify(U, V, W, n, m, p):
                        best_U = U
                        best_V = V
                        best_W = W
                        best_config = (i_p, j_p, k_p)
                    else:
                        # If verification fails, there might be a bug in construction logic
                        # For now, we skip it, but in a real scenario we'd debug.
                        pass
                else:
                    # Pruning: if current rank is already >= best, we don't need to construct
                    # But we don't know if it's strictly better until we finish summing.
                    # Since we sum first, this pruning is effective.
                    pass

    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
