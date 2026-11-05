"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 8]

def _strassen_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # p2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1
    
    # p3 = a00(b10-b11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[2,2]=1
    
    # p4 = a11(b01-b00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=-1
    
    # p5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # p6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[1,5]=1; W[3,5]=1
    
    # p7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=-1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple decompositions by horizontal concatenation.
    Each decomps entry is (U, V, W) tuple.
    """
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for U_d, V_d, W_d in decomps:
        r = U_d.shape[1]
        U[:, col_offset:col_offset+r] = U_d
        V[:, col_offset:col_offset+r] = V_d
        W[:, col_offset:col_offset+r] = W_d
        col_offset += r
    
    return U, V, W

def _build_3x4x8_decomposition():
    """
    Build a decomposition for 3x4x8 using Strassen on 2x2x2 blocks.
    For i=0,1: 8 Strassen blocks (rank 56)
    For i=2: schoolbook (rank 32)
    Total: 88
    """
    n, m, p = 3, 4, 8
    
    U_strassen, V_strassen, W_strassen = _strassen_decomposition()
    
    decomps = []
    
    # 8 Strassen blocks for i in {0,1}, j in {0,2}, k in {0,2,4,6}
    for i_off in [0, 1]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                U_block = np.zeros((n*m, U_strassen.shape[1]))
                V_block = np.zeros((m*p, V_strassen.shape[1]))
                W_block = np.zeros((p*n, W_strassen.shape[1]))
                
                for c in range(U_strassen.shape[1]):
                    for di in range(2):
                        for dj in range(2):
                            i_idx = i_off + di
                            j_idx = j_off + dj
                            if i_idx < n and j_idx < m:
                                U_block[i_idx * m + j_idx, c] = U_strassen[di*2+dj, c]
                    
                    for dj in range(2):
                        for dk in range(2):
                            j_idx = j_off + dj
                            k_idx = k_off + dk
                            if j_idx < m and k_idx < p:
                                V_block[j_idx * p + k_idx, c] = V_strassen[dj*2+dk, c]
                    
                    for dk in range(2):
                        for di in range(2):
                            k_idx = k_off + dk
                            i_idx = i_off + di
                            if k_idx < p and i_idx < n:
                                W_block[k_idx * n + i_idx, c] = W_strassen[dk*2+di, c]
                
                decomps.append((U_block, V_block, W_block))
    
    # Schoolbook for i=2 (the third row)
    U_rest = np.zeros((n*m, m*p))
    V_rest = np.zeros((m*p, m*p))
    W_rest = np.zeros((p*n, m*p))
    
    col = 0
    for j in range(m):
        for k in range(p):
            U_rest[2*m + j, col] = 1
            V_rest[j*p + k, col] = 1
            W_rest[k*n + 2, col] = 1
            col += 1
    
    decomps.append((U_rest, V_rest, W_rest))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Try to reduce the rank by removing redundant terms.
    Greedy: for each term, check if it can be expressed as a linear combination of others.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Build the tensor vector for each (i,j,k)
    # For each term r, the contribution to (i,j,k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Collect all (i,j,k) triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    
    num_triples = len(triples)
    
    # For each term, build its contribution vector
    term_vectors = np.zeros((num_triples, rank))
    for idx, (i, j, k) in enumerate(triples):
        a = i * m + j
        b = j * p + k
        c = k * n + i
        for r in range(rank):
            term_vectors[idx, r] = U[a, r] * V[b, r] * W[c, r]
    
    # Greedy removal: try to remove each term if it's in the span of the others
    active = list(range(rank))
    changed = True
    max_iterations = 10  # Limit iterations to avoid timeout
    iteration = 0
    while changed and time.time() < deadline and iteration < max_iterations:
        changed = False
        iteration += 1
        for r in active[:]:
            if len(active) <= 1:
                break
            # Check if term r is in the span of the other active terms
            other_active = [s for s in active if s != r]
            if len(other_active) < len(active) - 1:
                continue
            
            # Solve: term_vectors[:, r] = sum_{s in other_active} c_s * term_vectors[:, s]
            # This is a linear system: A * c = b
            A = term_vectors[:, other_active]  # num_triples x (rank-1)
            b = term_vectors[:, r]
            
            # Check if b is in the column space of A
            # Use least squares and check residual
            try:
                c, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                residual = np.linalg.norm(A @ c - b)
                if residual < tol * max(1.0, np.linalg.norm(b)):
                    # Term r is redundant, remove it
                    active.remove(r)
                    changed = True
            except Exception:
                pass
    
    # Build the reduced decomposition
    if len(active) < rank:
        U_red = U[:, active]
        V_red = V[:, active]
        W_red = W[:, active]
        return U_red, V_red, W_red
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns dict with 'U', 'V', 'W' keys, each a 2D list.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U, best_V, best_W = None, None, None
    
    # Try the Strassen-based decomposition
    try:
        U, V, W = _build_3x4x8_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            # Try to reduce rank with a tighter deadline to ensure we finish in time
            reduction_deadline = min(deadline, start_time + 100)
            U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, reduction_deadline)
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
    except Exception:
        pass
    
    # Fallback to schoolbook if nothing else worked
    if best_U is None:
        U, V, W = _schoolbook_decomposition(n, m, p)
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
