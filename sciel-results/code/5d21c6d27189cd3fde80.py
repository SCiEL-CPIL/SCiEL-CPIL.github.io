# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from itertools import combinations, product

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Uses a randomized search for a rank-19 decomposition, leveraging the
    known lower bound and the structure of the tensor.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    n, m, p = 2, 3, 4
    rank_schoolbook = n * m * p  # 24
    
    # 1. Construct the known rank-24 solution (Schoolbook)
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # 2. Attempt to find a rank-19 decomposition
    if time.time() < deadline - 10.0:
        result_19 = _try_rank_random_search(n, m, p, 19, deadline)
        if result_19 is not None:
            best_result = result_19
            best_rank = 19
            return best_result
            
    # 3. If rank 19 not found, try rank 20
    if best_rank == rank_schoolbook and time.time() < deadline - 5.0:
        result_20 = _try_rank_random_search(n, m, p, 20, deadline)
        if result_20 is not None:
            best_result = result_20
            best_rank = 20
            return best_result

    return best_result

def _schoolbook(n, m, p):
    """Construct the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                column += 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the target multiplication tensor T."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _try_rank_random_search(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition of rank 'target_rank' using randomized search.
    We fix U and V to have small integer entries and solve for W.
    """
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Number of non-zero entries in T
    num_nonzero = n * m * p
    
    # We want to find U (nm x r), V (mp x r), W (pn x r)
    # such that sum_r U[:,r] V[:,r] W[:,r] = T
    
    # Strategy:
    # 1. Randomly initialize U and V with sparse integer entries (-1, 0, 1).
    # 2. Form the linear system for W: A W_flat = T_flat
    #    where A is the matrix of the tensor product U[:,r] and V[:,r].
    #    Actually, the equation is:
    #    sum_r (U[:,r] \otimes V[:,r]) W[:,r] = T
    #    Let M_r = U[:,r] \otimes V[:,r] (size nm*mp x 1? No, it's a matrix in the Kronecker sense).
    #    The term U[:,r] \otimes V[:,r] \otimes W[:,r] is a 3-tensor.
    #    Flattening:
    #    Let u_r = U[:,r], v_r = V[:,r], w_r = W[:,r].
    #    T = sum_r u_r \otimes v_r \otimes w_r
    #    Flatten T to T_vec (size nm*mp*pn).
    #    Flatten u_r \otimes v_r \otimes w_r.
    #    This is equivalent to:
    #    T_vec = sum_r (u_r \otimes v_r) \otimes w_r ? No.
    #    T_vec[r, s, t] = sum_k U[r,k] V[s,k] W[t,k]
    #    This is a linear system in W if U and V are fixed.
    #    For each k, we have a vector A_k of size nm*mp*pn?
    #    No, for a fixed k, the contribution to T is U[:,k] \otimes V[:,k] \otimes W[:,k].
    #    This is a rank-1 tensor in the 3D space.
    #    If we fix U and V, the equation for T is:
    #    sum_k (U[:,k] \otimes V[:,k] \otimes W[:,k]) = T
    #    This is linear in W[:,k].
    #    Let M_k be the matrix such that M_k W[:,k] = ...?
    #    Actually, we can view this as:
    #    For each k, let Z_k = U[:,k] \otimes V[:,k]. This is a matrix of size nm x mp.
    #    The term Z_k \otimes W[:,k] is a 3-tensor.
    #    We can flatten Z_k \otimes W[:,k] into a vector.
    #    The map W[:,k] -> vec(Z_k \otimes W[:,k]) is linear.
    #    Let B_k be the matrix representing this map.
    #    Then sum_k B_k W[:,k] = T_flat.
    #    We can combine all W[:,k] into a single vector W_all.
    #    Then A W_all = T_flat, where A is the concatenation of B_k.
    
    # Let's construct A.
    # A has dimensions (nm*mp*pn) x (pn * target_rank)
    # Column corresponding to (k, t) is the vector representing the contribution of W[t,k]
    # to the flattened tensor.
    # The contribution of W[t,k] to T[a,b,c] is U[a,k] * V[b,k] * W[c,k] (if c=t).
    # So the column for (k, t) has non-zero entries at indices (a, b, t) with value U[a,k] * V[b,k].
    
    A = np.zeros((nm * mp * pn, pn * target_rank), dtype=np.float64)
    
    # Precompute U and V randomly
    # We will try multiple random initializations
    
    max_attempts = 500
    attempt = 0
    
    while attempt < max_attempts and time.time() < deadline:
        attempt += 1
        
        # Generate sparse random U and V
        # Sparsity: each column has 2-3 non-zeros
        U = np.zeros((nm, target_rank), dtype=np.float64)
        V = np.zeros((mp, target_rank), dtype=np.float64)
        
        for k in range(target_rank):
            # Choose 2 random rows for U
            rows_u = np.random.choice(nm, size=2, replace=False)
            for r in rows_u:
                U[r, k] = np.random.choice([-1, 1])
                
            # Choose 2 random rows for V
            rows_v = np.random.choice(mp, size=2, replace=False)
            for r in rows_v:
                V[r, k] = np.random.choice([-1, 1])
        
        # Construct A
        # A[:, k*pn + t] corresponds to W[t, k]
        for k in range(target_rank):
            u_k = U[:, k]
            v_k = V[:, k]
            # The outer product u_k \otimes v_k is a matrix of size nm x mp
            # We need to place this into the columns of A for all t in range(pn)
            # For a fixed t, the column of A is:
            # vec( u_k \otimes v_k \otimes e_t )
            # where e_t is the standard basis vector of size pn.
            # This vector has non-zero entries at indices (a, b, t) with value u_k[a] * v_k[b].
            
            # We can construct this efficiently.
            # The index in the flattened tensor for (a, b, t) is a*mp*pn + b*pn + t.
            # We want to fill A[:, k*pn + t] for all t.
            
            # Let's iterate over t
            for t in range(pn):
                col_idx = k * pn + t
                # The non-zero entries are at (a, b, t)
                # We can use the outer product u_k * v_k
                # Flatten u_k (nm) and v_k (mp)
                # The contribution to T[a,b,t] is u_k[a] * v_k[b]
                
                # We can create a sparse vector or use matrix operations.
                # A[:, col_idx] should be zero except at indices (a, b, t).
                # Let's build the indices.
                # This loop might be slow. Let's optimize.
                
                # Alternative:
                # A[:, k*pn + t] is the vectorization of the tensor U[:,k] \otimes V[:,k] \otimes e_t
                # This is equal to (U[:,k] \otimes V[:,k]) \otimes e_t
                # In flattened form, it's a permutation of (U[:,k] \otimes V[:,k]) repeated pn times? No.
                
                # Let's just fill it directly for speed since dimensions are small.
                # nm=6, mp=12, pn=8. Total size 576.
                # Columns: 19 * 8 = 152.
                # 152 * 576 operations per attempt. 500 attempts -> 44 million ops. Fast enough.
                
                # Create a zero vector
                vec = np.zeros(nm * mp * pn)
                # Fill in non-zeros
                # Indices: a*mp*pn + b*pn + t
                for a in range(nm):
                    if u_k[a] != 0:
                        for b in range(mp):
                            if v_k[b] != 0:
                                idx = a * mp * pn + b * pn + t
                                vec[idx] = u_k[a] * v_k[b]
                A[:, col_idx] = vec
        
        # Solve A W_all = T_flat
        try:
            W_all, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if solution is close to half-integers
        W_all_rounded = np.round(W_all * 2) / 2
        error = np.linalg.norm(A @ W_all_rounded - T_flat)
        
        if error < 1e-9:
            # Found a solution!
            W = W_all_rounded.reshape(target_rank, pn).T # W is pn x target_rank
            # Verify
            tensor_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            if np.allclose(tensor_calc, T, atol=1e-9):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If not exact, maybe we can polish?
        # For now, just continue searching.
        
    return None

# EVOLVE-BLOCK-END
