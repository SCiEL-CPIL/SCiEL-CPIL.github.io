"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U @ diag(c) @ V^T reconstructs the tensor correctly."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check against the standard matrix multiplication tensor
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the full tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Standard: T[i,j,k] = 1 if this is a valid (i,j,k) triple for matmul
                # For matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor has T[i,j,k] = 1 for all valid i,j,k
                expected = 1.0
                actual = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > 1e-9:
                    errors.append((i, j, k, actual, expected))
    
    return len(errors) == 0, errors[:5] if errors else []

def _embed_block(U_block, V_block, W_block, n1, m1, p1, i_off, j_off, k_off, n, m, p):
    """
    Embed a block decomposition into the full n x m x p tensor.
    U_block: (n1*m1, r), V_block: (m1*p1, r), W_block: (p1*n1, r)
    Place at offset (i_off, j_off, k_off) in the full tensor.
    Returns full U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    """
    r = U_block.shape[1]
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for i in range(n1):
        for j in range(m1):
            U[(i_off + i) * m + (j_off + j), :] = U_block[i * m1 + j, :]
    
    for j in range(m1):
        for k in range(p1):
            V[(j_off + j) * p + (k_off + k), :] = V_block[j * p1 + k, :]
    
    for k in range(p1):
        for i in range(n1):
            W[(k_off + k) * n + (i_off + i), :] = W_block[k * n1 + i, :]
    
    return U, V, W

@lru_cache(maxsize=None)
def _get_decomposition(n, m, p):
    """
    Recursively find the minimal rank decomposition for n x m x p.
    Returns (U, V, W, rank).
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W, 1

    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7

    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _get_decomposition(n1, m, p)
        U2, V2, W2, r2 = _get_decomposition(n2, m, p)
        
        if r1 + r2 < best_rank:
            # Combine
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Stack rows
            U[:n1*m, :r1] = U1
            U[n1*m:, r1:] = U2
            
            # V: Concat columns
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            # W: Scatter rows based on i index
            # W1 corresponds to i in [0, n1)
            # W2 corresponds to i in [n1, n)
            # Global W index: k*n + i
            for k in range(p):
                for i1 in range(n1):
                    W[k*n + i1, :r1] = W1[k*n1 + i1, :]
                for i2 in range(n2):
                    W[k*n + (n1 + i2), r1:] = W2[k*n2 + i2, :]
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = _get_decomposition(n, m1, p)
        U2, V2, W2, r2 = _get_decomposition(n, m2, p)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Scatter columns based on j index? No, U is indexed by i*m + j.
            # U1 corresponds to j in [0, m1)
            # U2 corresponds to j in [m1, m)
            for i in range(n):
                for j1 in range(m1):
                    U[i*m + j1, :r1] = U1[i*m1 + j1, :]
                for j2 in range(m2):
                    U[i*m + (m1 + j2), r1:] = U2[i*m2 + j2, :]
            
            # V: Stack rows
            # V is indexed by j*p + k
            # V1 corresponds to j in [0, m1)
            # V2 corresponds to j in [m1, m)
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            
            # W: Concat columns
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _get_decomposition(n, m, p1)
        U2, V2, W2, r2 = _get_decomposition(n, m, p2)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # U: Concat columns
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # V: Scatter columns based on k index? No, V is indexed by j*p + k.
            # V1 corresponds to k in [0, p1)
            # V2 corresponds to k in [p1, p)
            for j in range(m):
                for k1 in range(p1):
                    V[j*p + k1, :r1] = V1[j*p1 + k1, :]
                for k2 in range(p2):
                    V[j*p + (p1 + k2), r1:] = V2[j*p2 + k2, :]
            
            # W: Stack rows
            # W is indexed by k*n + i
            # W1 corresponds to k in [0, p1)
            # W2 corresponds to k in [p1, p)
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Fallback: schoolbook
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
        return U, V, W, n * m * p

    return best_U, best_V, best_W, best_rank

def _try_block_partition(n, m, p, n_parts, m_parts, p_parts):
    """
    Try a specific block partition and compute the total rank.
    n_parts, m_parts, p_parts are lists of sizes that sum to n, m, p respectively.
    Returns (U, V, W, total_rank) or None if not better.
    """
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    total_rank = 0
    
    i_off = 0
    for n1 in n_parts:
        j_off = 0
        for m1 in m_parts:
            k_off = 0
            for p1 in p_parts:
                U_b, V_b, W_b, r = _get_decomposition(n1, m1, p1)
                U_full, V_full, W_full = _embed_block(
                    U_b, V_b, W_b, n1, m1, p1, i_off, j_off, k_off, n, m, p
                )
                if U.shape[1] == 0:
                    U = U_full
                    V = V_full
                    W = W_full
                else:
                    U = np.hstack([U, U_full])
                    V = np.hstack([V, V_full])
                    W = np.hstack([W, W_full])
                total_rank += r
                k_off += p1
            j_off += m1
        i_off += n1
    
    return U, V, W, total_rank

def _generate_partitions(total):
    """Generate all compositions of 'total' into positive integers."""
    if total == 1:
        yield [1]
        return
    for first in range(1, total):
        for rest in _generate_partitions(total - first):
            yield [first] + rest

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    # First, get the standard recursive decomposition
    U, V, W, rank = _get_decomposition(n, m, p)
    
    # Now try various block partitions to see if we can do better
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try different partitions of the dimensions
    # For n=3: [3], [1,2], [2,1], [1,1,1]
    # For m=5: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,3,1], [3,1,1], [1,1,1,2], etc.
    # For p=5: same as m
    
    # To keep it fast, only try a subset of partitions
    n_opts = [[3], [1,2], [2,1], [1,1,1]]
    m_opts = [[5], [2,3], [3,2], [1,4], [4,1], [1,1,3], [1,3,1], [3,1,1]]
    p_opts = [[5], [2,3], [3,2], [1,4], [4,1], [1,1,3], [1,3,1], [3,1,1]]
    
    for np_ in n_opts:
        for mp_ in m_opts:
            for pp_ in p_opts:
                if time.time() > deadline:
                    break
                try:
                    U_try, V_try, W_try, r_try = _try_block_partition(n, m, p, np_, mp_, pp_)
                    if r_try < best_rank:
                        # Verify
                        valid, _ = _verify_decomposition(U_try, V_try, W_try, n, m, p)
                        if valid:
                            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, r_try
                except Exception:
                    pass
    
    # Round to nearest integer to avoid floating point representation issues
    # The decompositions used (Strassen, Schoolbook) have integer coefficients.
    U_out = np.round(best_U).astype(int).tolist()
    V_out = np.round(best_V).astype(int).tolist()
    W_out = np.round(best_W).astype(int).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
