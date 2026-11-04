"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2_terms(i0, j0, k0, n, m, p):
    """
    Return the 7 Strassen terms for a 2x2 block at position (i0,j0,k0).
    Each term is (U_dict, V_dict, W_dict) where dicts map index->coeff.
    U index: i*m+j, V index: j*p+k, W index: k*n+i
    """
    terms = []
    
    # M1 = (A[i0,j0]+A[i0+1,j0]) * (B[j0,k0]+B[j0+1,k0])
    # Contributes: C[i0,k0] += M1, C[i0+1,k0+1] += M1
    terms.append((
        {i0*m+j0: 1, (i0+1)*m+j0: 1},
        {j0*p+k0: 1, (j0+1)*p+k0: 1},
        {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
    ))
    
    # M2 = (A[i0,j0+1]+A[i0+1,j0+1]) * B[j0,k0]
    # Contributes: C[i0+1,k0] += M2
    terms.append((
        {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): 1},
        {j0*p+k0: 1},
        {k0*n+(i0+1): 1}
    ))
    
    # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0,k0])
    # Contributes: C[i0,k0+1] += M3, C[i0+1,k0+1] -= M3
    terms.append((
        {i0*m+j0: 1},
        {(j0+1)*p+k0: 1, j0*p+k0: -1},
        {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): -1}
    ))
    
    # M4 = A[i0+1,j0] * (B[j0+1,k0+1]-B[j0,k0])
    # Contributes: C[i0,k0] -= M4, C[i0+1,k0] += M4
    terms.append((
        {(i0+1)*m+j0: 1},
        {(j0+1)*p+(k0+1): 1, j0*p+k0: -1},
        {k0*n+i0: -1, k0*n+(i0+1): 1}
    ))
    
    # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    # Contributes: C[i0+1,k0] += M5, C[i0+1,k0+1] += M5
    terms.append((
        {i0*m+j0: 1, i0*m+(j0+1): 1},
        {(j0+1)*p+(k0+1): 1},
        {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): 1}
    ))
    
    # M6 = (A[i0+1,j0+1]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    # Contributes: C[i0,k0] += M6, C[i0+1,k0+1] -= M6
    terms.append((
        {(i0+1)*m+(j0+1): 1, i0*m+j0: -1},
        {j0*p+k0: 1, j0*p+(k0+1): 1},
        {k0*n+i0: 1, (k0+1)*n+(i0+1): -1}
    ))
    
    # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0+1]-B[j0,k0])
    # Contributes: C[i0,k0] -= M7
    terms.append((
        {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
        {(j0+1)*p+(k0+1): 1, j0*p+k0: -1},
        {k0*n+i0: -1}
    ))
    
    return terms

def _try_strassen_decomposition(n, m, p):
    """
    Try to build a lower-rank decomposition using Strassen for 2x2 blocks.
    For 3x4 @ 4x6, we can use Strassen on the bottom-right 2x2 block of B
    (rows 2-3, cols 4-5) with the top 2 rows of A.
    """
    # Start with all zeros
    U = np.zeros((n * m, n * m * p))  # Start with max possible, will trim
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    
    # The 3x4 @ 4x6 product:
    # C[0,k] = sum_j A[0,j]*B[j,k] for k=0..5
    # C[1,k] = sum_j A[1,j]*B[j,k] for k=0..5  
    # C[2,k] = sum_j A[2,j]*B[j,k] for k=0..5
    
    # Use Strassen for the 2x2 block: A[0:2, 2:4] @ B[2:4, 4:6]
    # This covers C[0:2, 4:6]
    strassen_terms = _strassen_2x2_terms(0, 2, 4, n, m, p)
    for U_dict, V_dict, W_dict in strassen_terms:
        for u_idx, coeff in U_dict.items():
            U[u_idx, col] += coeff
        for v_idx, coeff in V_dict.items():
            V[v_idx, col] += coeff
        for w_idx, coeff in W_dict.items():
            W[w_idx, col] += coeff
        col += 1
    
    # Now fill in the remaining terms with schoolbook
    # The Strassen block covered:
    # - A entries: (0,2), (1,2), (0,3), (1,3) 
    # - B entries: (2,4), (3,4), (2,5), (3,5)
    # - C entries: (0,4), (1,4), (0,5), (1,5)
    
    # We need to cover all other (i,j,k) triples
    # For each (i,j,k) not covered by Strassen, add a schoolbook term
    covered_U = set()
    covered_V = set()
    covered_W = set()
    
    # Actually, let's just add schoolbook terms for all (i,j,k) that aren't 
    # fully handled. The Strassen terms already contribute to the covered C entries.
    # We need to be careful not to double-count.
    
    # Simpler approach: add schoolbook terms for all (i,j,k) where the C entry
    # is not in the Strassen-covered region, OR where the A/B entries are not
    # in the Strassen-covered region.
    
    # The Strassen block handles C[0:2, 4:6] using A[0:2, 2:4] and B[2:4, 4:6]
    # For other C entries, we need schoolbook terms.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the Strassen-covered region
                if i < 2 and j >= 2 and k >= 4:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140
    
    # Start with schoolbook as guaranteed fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    rank_school = n * m * p
    
    # Try Strassen-based decomposition
    try:
        U_str, V_str, W_str, rank_str = _try_strassen_decomposition(n, m, p)
        if _verify(U_str, V_str, W_str, n, m, p):
            if rank_str < rank_school:
                return {
                    "U": U_str.tolist(),
                    "V": V_str.tolist(),
                    "W": W_str.tolist()
                }
    except Exception:
        pass
    
    # Return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
