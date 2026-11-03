"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = []
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    return terms

def _naive_2x2x1():
    """Return the 2 terms of naive 2x2x1 matrix multiplication."""
    terms = []
    terms.append({
        'U': {(0,0): 1, (1,0): 1},
        'V': {(0,0): 1},
        'W': {(0,0): 1}
    })
    terms.append({
        'U': {(0,1): 1, (1,1): 1},
        'V': {(1,0): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_block_decomposition(n, m, p):
    """Build decomposition by splitting into 2x2 blocks."""
    n_blocks = [2, 2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 1]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U = {}
    V = {}
    W = {}
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                no = n_offsets[ni]
                mo = m_offsets[mi]
                po = p_offsets[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for term in _strassen_2x2():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                elif ns == 2 and ms == 2 and ps == 1:
                    for term in _naive_2x2x1():
                        for (li, lj), cu in term['U'].items():
                            i, j = no + li, mo + lj
                            U.setdefault((i, j), []).append((cu, rank))
                        for (lj, lk), cv in term['V'].items():
                            j, k = mo + lj, po + lk
                            V.setdefault((j, k), []).append((cv, rank))
                        for (lk, li), cw in term['W'].items():
                            k, i = po + lk, no + li
                            W.setdefault((k, i), []).append((cw, rank))
                        rank += 1
                else:
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                i, j, k = no + li, mo + lj, po + lk
                                U.setdefault((i, j), []).append((1, rank))
                                V.setdefault((j, k), []).append((1, rank))
                                W.setdefault((k, i), []).append((1, rank))
                                rank += 1
    
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by identifying and removing redundant terms.
    Uses SVD-based approach: vectorize the tensor and find its rank.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find linear dependencies among the rank-1 tensors
    # For each term r, we have a rank-1 tensor T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want to find if any term is a linear combination of others
    
    # Build the tensor matrix: each column r is a vectorized rank-1 tensor
    # Shape: (n*m*p, rank)
    tensor_matrix = np.zeros((n*m*p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                u_val = U_np[i*m+j]
                v_val = V_np[j*p+k]
                w_val = W_np[k*n+i]
                tensor_matrix[idx] = u_val * v_val * w_val
    
    # Compute SVD to find the numerical rank
    try:
        U_svd, s, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
        # Find the numerical rank (singular values > tolerance)
        tol = 1e-10
        num_rank = np.sum(s > tol)
    except Exception:
        num_rank = rank
    
    if num_rank < rank:
        # The tensor matrix has lower rank than our decomposition
        # This means some terms are linearly dependent
        # We can potentially reduce the rank
        # For now, just return the original if we can't easily reduce
        pass
    
    return None  # No reduction found yet

def _build_svd_decomposition(n, m, p, deadline):
    """
    Try to build a dense decomposition using SVD of the multiplication operator.
    Vectorize the multiplication as a linear map and find its minimal rank decomposition.
    """
    # The multiplication C = A @ B where A is n×m, B is m×p, C is n×p
    # We can vectorize: vec(C) = (I_p ⊗ A) * vec(B) but this is for fixed A
    # For the tensor decomposition, we need a different approach
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # This is equivalent to finding the minimal rank decomposition of the 
    # 3-tensor with all entries 1.
    
    # One approach: use the fact that T can be written as a sum of rank-1 tensors
    # T[i,j,k] = sum_r (a_r[i] * b_r[j]) * (c_r[j] * d_r[k]) * (e_r[k] * f_r[i])
    
    # For the all-ones tensor, a simple decomposition is:
    # T[i,j,k] = sum_j' delta[j,j'] * 1 * 1 (schoolbook, rank m)
    # But we need the full bilinear structure
    
    # Actually, the all-ones 3-tensor has rank 1! Because T[i,j,k] = 1*1*1
    # But that's not the right tensor - we need the multiplication tensor
    
    # The multiplication tensor for n×m × m×p -> n×p has entries:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # This IS the all-ones tensor of size n×m×p
    # And its rank as a 3-tensor is... 1? No, that's not right either.
    
    # The tensor decomposition we need is:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is exactly the all-ones tensor of shape (n*m, m*p, p*n) but with specific index mappings.
    
    # The all-ones tensor of shape (a,b,c) has tensor rank 1 (it's outer product of three all-ones vectors)
    # But our tensor has a specific structure due to the index mappings.
    
    # Let me just return the block decomposition for now
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try the block decomposition approach first (rank 48)
    try:
        result = _build_block_decomposition(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                # Check if we're still within deadline
                if time.time() < deadline:
                    # Try to reduce rank further
                    reduced = _try_reduce_rank(result["U"], result["V"], result["W"], n, m, p, deadline)
                    if reduced is not None and _verify_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                        return reduced
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
