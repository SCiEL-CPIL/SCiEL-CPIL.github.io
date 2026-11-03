"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _strassen_2x2x2_tensors():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen multiplication as (U,V,W) triples."""
    # Each product: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, coeff) for A[i,j]
    # v_coeffs: list of (j, k, coeff) for B[j,k]
    # w_coeffs: list of (k, i, coeff) for C[k,i] (using trace convention W[k*n+i])
    products = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M3 = A00(B01-B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # M4 = A11(B10-B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    return products

def _build_strassen_decomposition(n, m, p):
    """Build the full Strassen-based decomposition for n x m by m x p."""
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    products = _strassen_2x2x2_tensors()
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for u_coeffs, v_coeffs, w_coeffs in products:
                for i, j, c in u_coeffs:
                    row_u = i * m + (j_off + j)
                    U[row_u][col] += c
                for j, k, c in v_coeffs:
                    row_v = (j_off + j) * p + (k_off + k)
                    V[row_v][col] += c
                for k, i, c in w_coeffs:
                    row_w = (k_off + k) * n + i
                    W[row_w][col] += c
                col += 1
                
    return U, V, W, rank

def _tensor_to_vector(U, V, W, n, m, p):
    """Convert a rank-1 tensor (U,V,W) to a flat vector in the tensor space."""
    vec = [0.0] * (n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
    return vec

def _full_tensor_vector(n, m, p):
    """The full matrix multiplication tensor as a flat vector."""
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually the multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    vec = [0.0] * (n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = 1.0
    return vec

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(W[0])
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among the 28 rank-1 tensors to reduce rank."""
    rank = len(W[0])
    dim = n * m * p
    
    # Build the 28 tensor vectors
    tensor_vecs = []
    for r in range(rank):
        U_r = [U[i][r] for i in range(n*m)]
        V_r = [V[i][r] for i in range(m*p)]
        W_r = [W[i][r] for i in range(p*n)]
        tv = _tensor_to_vector(U_r, V_r, W_r, n, m, p)
        tensor_vecs.append(tv)
    
    tensor_vecs = np.array(tensor_vecs, dtype=float)
    
    # Check linear independence
    # Use SVD to find the rank of the set of 28 vectors
    svd_U, svd_S, svd_Vt = np.linalg.svd(tensor_vecs, full_matrices=False)
    tol = 1e-10
    num_independent = np.sum(svd_S > tol)
    
    if num_independent < rank:
        # There are linear dependencies. We can eliminate (rank - num_independent) tensors.
        # Find which tensors are dependent.
        # The left singular vectors corresponding to small singular values give the dependencies.
        # Actually, we want to find which columns of tensor_vecs are linear combinations of others.
        # Use the right singular vectors (svd_Vt) - rows corresponding to small singular values span the null space.
        
        # Better approach: find a maximal linearly independent subset, then express the rest.
        # For now, if we find dependencies, we can try to eliminate them greedily.
        
        # Find the null space of tensor_vecs (as rows)
        # tensor_vecs is (28, dim). We want to find coefficients c such that sum_r c[r] * tensor_vecs[r] = 0
        # This is the null space of tensor_vecs.T, i.e., the left null space of tensor_vecs.
        
        # Actually, we want to find if any tensor_vecs[r] is in the span of the others.
        # Let's do this: for each r, check if tensor_vecs[r] is in the span of tensor_vecs without r.
        
        removable = []
        for r in range(rank):
            if time.time() > deadline:
                break
            others = [tensor_vecs[j] for j in range(rank) if j != r]
            if len(others) == 0:
                continue
            others_arr = np.array(others, dtype=float)
            # Check if tensor_vecs[r] is in the column space of others_arr
            # i.e., is tensor_vecs[r] in the span of the rows of others_arr?
            # No: we want to know if tensor_vecs[r] is a linear combination of the other vectors.
            # The other vectors are rows of others_arr (shape: (rank-1, dim)).
            # We want: exists c in R^(rank-1) such that c @ others_arr = tensor_vecs[r]
            # This is: tensor_vecs[r] is in the row space of others_arr.
            # Equivalent: the rank of [others_arr; tensor_vecs[r]] equals rank of others_arr.
            
            combined = np.vstack([others_arr, tensor_vecs[r].reshape(1, -1)])
            rank_others = np.linalg.matrix_rank(others_arr, tol=tol)
            rank_combined = np.linalg.matrix_rank(combined, tol=tol)
            if rank_combined == rank_others:
                removable.append(r)
        
        if removable:
            # Remove the first removable tensor and rebuild
            # Actually, we need to be careful: removing one might make others independent.
            # Let's just try removing all identified removable tensors simultaneously.
            # But we need to verify the result still decomposes correctly.
            # Since the removed tensors are in the span of the remaining ones, the sum is unchanged.
            
            keep = [r for r in range(rank) if r not in removable]
            U_new = [[U[i][r] for r in keep] for i in range(n*m)]
            V_new = [[V[i][r] for r in keep] for i in range(m*p)]
            W_new = [[W[i][r] for r in keep] for i in range(p*n)]
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, len(keep)
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Build the base Strassen decomposition
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should not happen, but fallback
        pass
    
    # Try to reduce rank by finding linear dependencies
    U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline)
    
    # Convert to integer/half-integer if possible
    def _round_values(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                new_row.append(rounded)
            result.append(new_row)
        return result
    
    U = _round_values(U)
    V = _round_values(V)
    W = _round_values(W)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # If rounding broke it, use the unrounded version
        U, V, W, rank = _build_strassen_decomposition(n, m, p)
        U, V, W, rank = _try_rank_reduction(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
