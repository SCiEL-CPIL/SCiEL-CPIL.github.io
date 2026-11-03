"""Matrix multiplication 3x5x6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_coeffs, v_coeffs, w_coeffs) where:
      u_coeffs: dict mapping (i,j) in {0,1}x{0,1} to coefficient
      v_coeffs: dict mapping (j,k) in {0,1}x{0,1} to coefficient
      w_coeffs: dict mapping (k,i) in {0,1}x{0,1} to coefficient
    The tensor identity is:
      T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    For 2x2x2, T[i,j,k]=1 for all i,j,k in {0,1}.
    """
    # Strassen's 7 products for C = A*B where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # For the tensor T[i,j,k] = A[i,j]*B[j,k], we need:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    #
    # Each Strassen product r corresponds to:
    # U[i,j,r] = coefficient of A[i,j] in the first factor
    # V[j,k,r] = coefficient of B[j,k] in the second factor
    # W[k,i,r] = coefficient that puts this product into C[k,i]... 
    # Wait, the convention is W[k*n+i, r] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So the output index is (k, i) meaning row k of B times column i of A? No.
    # C[i,k] = sum_j A[i,j] B[j,k]. The output factor W has index (k*n+i, r).
    # So W[k,i,r] contributes to C[i,k] with weight W[k,i,r].
    #
    # For a Strassen product M_r = (sum_{i,j} a_{ij} A[i,j]) * (sum_{j,k} b_{jk} B[j,k]),
    # this product contributes to C[i,k] if and only if it's one of the C entries.
    # Actually, the way it works: M_r is a scalar, and C[i,k] gets a contribution
    # c_{ik}^{(r)} * M_r where c_{ik}^{(r)} is the coefficient in the reconstruction.
    #
    # So U[i,j,r] = a_{ij}, V[j,k,r] = b_{jk}, W[k,i,r] = c_{ik}^{(r)}
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11), contributes: C00 += 1, C11 += 1
    terms.append({
        'u': {(0,0): 1, (1,1): 1},
        'v': {(0,0): 1, (1,1): 1},
        'w': {(0,0): 1, (1,1): 1}  # W[k,i]: C[0,0] gets +1, C[1,1] gets +1
    })
    
    # M2 = (A10+A11)(B00), contributes: C10 += 1
    terms.append({
        'u': {(1,0): 1, (1,1): 1},
        'v': {(0,0): 1},
        'w': {(0,1): 1}  # C[1,0] gets +1
    })
    
    # M3 = A00*(B10-B11), contributes: C01 += 1
    terms.append({
        'u': {(0,0): 1},
        'v': {(1,0): 1, (1,1): -1},
        'w': {(1,0): 1}  # C[0,1] gets +1
    })
    
    # M4 = A11*(B01-B00), contributes: C00 -= 1, C01 += 1
    terms.append({
        'u': {(1,1): 1},
        'v': {(0,1): 1, (0,0): -1},
        'w': {(0,0): -1, (1,0): 1}  # C[0,0] gets -1, C[0,1] gets +1
    })
    
    # M5 = (A00+A01)*B11, contributes: C00 += 1, C10 += 1
    terms.append({
        'u': {(0,0): 1, (0,1): 1},
        'v': {(1,1): 1},
        'w': {(0,0): 1, (0,1): 1}  # C[0,0] gets +1, C[1,0] gets +1
    })
    
    # M6 = (A10-A00)*(B00+B01), contributes: C11 += 1
    terms.append({
        'u': {(1,0): 1, (0,0): -1},
        'v': {(0,0): 1, (0,1): 1},
        'w': {(1,1): 1}  # C[1,1] gets +1
    })
    
    # M7 = (A01-A11)*(B10+B11), contributes: C00 -= 1
    terms.append({
        'u': {(0,1): 1, (1,1): -1},
        'v': {(1,0): 1, (1,1): 1},
        'w': {(0,0): -1}  # C[0,0] gets -1
    })
    
    return terms

def _build_block_decomposition():
    """
    Build a rank-84 decomposition using 6 Strassen 2x2x2 blocks + schoolbook remainder.
    
    Partition:
      i in {0,1} vs {2}
      j in {0,1} vs {2,3} vs {4}
      k in {0,1} vs {2,3} vs {4,5}
    
    Strassen blocks (6 total, each rank 7 = 42):
      (i={0,1}, j={0,1}, k={0,1})
      (i={0,1}, j={0,1}, k={2,3})
      (i={0,1}, j={0,1}, k={4,5})
      (i={0,1}, j={2,3}, k={0,1})
      (i={0,1}, j={2,3}, k={2,3})
      (i={0,1}, j={2,3}, k={4,5})
    
    Schoolbook remainder (rank 42):
      i=2, all j in {0..4}, all k in {0..5}: 1*5*6 = 30
      i in {0,1}, j=4, all k in {0..5}: 2*1*6 = 12
      Total = 42
    
    Grand total = 42 + 42 = 84
    """
    n, m, p = 3, 5, 6
    total_rank = 84
    
    U = [[0] * total_rank for _ in range(n * m)]   # 15 rows
    V = [[0] * total_rank for _ in range(m * p)]   # 30 rows
    W = [[0] * total_rank for _ in range(p * n)]   # 18 rows
    
    col = 0
    strassen_terms = _strassen_2x2x2()
    
    # Define the 6 Strassen blocks
    blocks = []
    for i_off in [0, 1]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4]:
                blocks.append((i_off, j_off, k_off))
    
    for (i0, j0, k0) in blocks:
        for term in strassen_terms:
            for (di, dj), uval in term['u'].items():
                row = (i0 + di) * m + (j0 + dj)
                U[row][col] = uval
            for (dj, dk), vval in term['v'].items():
                row = (j0 + dj) * p + (k0 + dk)
                V[row][col] = vval
            for (dk, di), wval in term['w'].items():
                row = (k0 + dk) * n + (i0 + di)
                W[row][col] = wval
            col += 1
    
    # Schoolbook: i=2, all j, all k
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Schoolbook: i in {0,1}, j=4, all k
    for i in range(2):
        for k in range(p):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return U, V, W

def _verify(U, V, W, n=3, m=5, p=6, tol=1e-10):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, target_rank, n=3, m=5, p=6, time_budget=30.0):
    """
    Try to reduce the rank below the current decomposition by finding linear dependencies
    among the rank-1 terms. If we can express some terms as linear combinations of others,
    we can eliminate them.
    """
    start = time.time()
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Check for linear dependencies in the tensor coefficients
    # Build the 90xR tensor where entry (ijk, r) = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If the columns are linearly dependent, we can reduce rank.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build coefficient matrix: 90 x current_rank
    coeff = np.zeros((n*m*p, current_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff[idx] = U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]
                idx += 1
    
    # Find rank of coeff matrix
    try:
        _, s, vh = np.linalg.svd(coeff, full_matrices=False)
        effective_rank = np.sum(s > 1e-10)
    except Exception:
        return U, V, W
    
    if effective_rank < current_rank:
        # We have dependencies; try to find a minimal basis
        # Use the top effective_rank right singular vectors
        pass  # Complex to properly reduce; skip for now
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x6 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 20s safety margin before 150s limit
    
    # Build the rank-84 block decomposition
    U, V, W = _build_block_decomposition()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    best_U, best_V, best_W = U, V, W
    best_rank = 84
    
    # Try to reduce rank by finding and eliminating dependent terms
    if time.time() < deadline - 20:
        # Try rank 83, 82, ... by checking if any term is redundant
        for target in range(best_rank - 1, 79, -1):
            if time.time() > deadline - 15:
                break
            
            # Try removing each term and checking if the remaining still spans
            U_arr = np.array(best_U, dtype=np.float64)
            V_arr = np.array(best_V, dtype=np.float64)
            W_arr = np.array(best_W, dtype=np.float64)
            
            found_removal = False
            for r_remove in range(best_rank):
                if time.time() > deadline - 15:
                    break
                    
                # Remove column r_remove
                mask = [i for i in range(best_rank) if i != r_remove]
                U_sub = U_arr[:, mask]
                V_sub = V_arr[:, mask]
                W_sub = W_arr[:, mask]
                
                # Check if the remaining terms still produce all-ones tensor
                ok = True
                for i in range(n):
                    if not ok:
                        break
                    for j in range(m):
                        if not ok:
                            break
                        for k in range(p):
                            val = np.dot(U_sub[i*m+j], V_sub[j*p+k] * W_sub[k*n+i])
                            if abs(val - 1.0) > 1e-9:
                                ok = False
                                break
                
                if ok:
                    best_U = U_sub.tolist()
                    best_V = V_sub.tolist()
                    best_W = W_sub.tolist()
                    best_rank = target
                    found_removal = True
                    break
            
            if not found_removal:
                break
    
    # Try a more aggressive approach: use SVD to find minimal rank
    if time.time() < deadline - 10 and best_rank > 80:
        U_arr = np.array(best_U, dtype=np.float64)
        V_arr = np.array(best_V, dtype=np.float64)
        W_arr = np.array(best_W, dtype=np.float64)
        
        # Build the full coefficient matrix
        coeff = np.zeros((n*m*p, best_rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff[idx] = U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i]
                    idx += 1
        
        try:
            _, s, vh = np.linalg.svd(coeff, full_matrices=False)
            # Find effective rank
            tol_svd = 1e-10
            eff_rank = np.sum(s > tol_svd)
            
            if eff_rank < best_rank:
                # Project onto the top eff_rank components
                # This is approximate; verify before accepting
                vh_top = vh[:eff_rank]
                # New U, V, W from the projection
                # Actually this doesn't directly give valid U,V,W
                # Skip this approach as it's not straightforward
                pass
        except Exception:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
