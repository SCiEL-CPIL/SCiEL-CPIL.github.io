import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize matrices with zeros
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Strassen coefficients for 2x2 * 2x2
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col_idx = 0
    for a_block in range(2):
        for b_col_block in range(4):
            for t in range(7):
                term = strassen_terms[t]
                
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2*a_block + j_prime)
                    U[u_idx][col_idx] += coeff
                
                for (j_prime, k_prime), coeff in term['y'].items():
                    b_row = 2*a_block + j_prime
                    b_col = 2*b_col_block + k_prime
                    v_idx = b_row * p + b_col
                    V[v_idx][col_idx] += coeff
                
                for (i_prime, k_prime), coeff in term['z'].items():
                    c_col = 2*b_col_block + k_prime
                    c_row = i_prime
                    w_idx = c_col * n + c_row
                    W[w_idx][col_idx] += coeff
                
                col_idx += 1
    
    # Convert to numpy arrays for efficient computation
    U_np = np.array(U, dtype=float)  # shape (8, 56)
    V_np = np.array(V, dtype=float)  # shape (32, 56)
    W_np = np.array(W, dtype=float)  # shape (16, 56)
    
    # Try to find linear dependencies among the 56 rank-1 tensors
    # A rank-1 tensor is u_k ⊗ v_k ⊗ w_k
    # If sum_k alpha_k * u_k ⊗ v_k ⊗ w_k = 0, we can reduce rank
    
    # Strategy: search for small subsets of columns that are linearly dependent
    # in the tensor space. We check if there exist non-trivial coefficients
    # alpha_1, ..., alpha_r such that sum alpha_i * U[:,i] ⊗ V[:,i] ⊗ W[:,i] = 0
    
    # For efficiency, we'll check subsets of size 2 and 3 first
    # Two tensors u1⊗v1⊗w1 and u2⊗v2⊗w2 are dependent iff one is a scalar multiple
    # of the other (in the tensor product space)
    
    best_rank = rank
    best_indices = list(range(rank))
    
    # Check for pairs: u_i ⊗ v_i ⊗ w_i = c * u_j ⊗ v_j ⊗ w_j
    # This means U[:,i] = c*U[:,j], V[:,i] = c*V[:,j], W[:,i] = c*W[:,j]
    # Or more generally, the outer products are proportional
    
    # Build a set of "normalized" direction vectors for quick lookup
    # Normalize each column of U, V, W separately
    
    def normalize_col(vec):
        """Normalize a vector, return None if zero"""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return None
        return vec / norm
    
    # For each column, compute normalized U, V, W vectors
    U_norms = []
    V_norms = []
    W_norms = []
    for k in range(rank):
        U_norms.append(normalize_col(U_np[:, k]))
        V_norms.append(normalize_col(V_np[:, k]))
        W_norms.append(normalize_col(W_np[:, k]))
    
    # Check pairs: if U_norms[i] == U_norms[j] AND V_norms[i] == V_norms[j] AND W_norms[i] == W_norms[j]
    # then the two rank-1 terms are proportional and can be combined
    redundant_pairs = []
    for i in range(rank):
        for j in range(i+1, rank):
            if U_norms[i] is None or V_norms[i] is None or W_norms[i] is None:
                continue
            if U_norms[j] is None or V_norms[j] is None or W_norms[j] is None:
                continue
            if (np.allclose(U_norms[i], U_norms[j], atol=1e-10) and
                np.allclose(V_norms[i], V_norms[j], atol=1e-10) and
                np.allclose(W_norms[i], W_norms[j], atol=1e-10)):
                redundant_pairs.append((i, j))
    
    # If we found redundant pairs, we can merge them
    if redundant_pairs:
        # Find which columns to remove
        to_remove = set()
        for i, j in redundant_pairs:
            # Keep the one with larger magnitude, remove the other
            mag_i = np.linalg.norm(U_np[:, i]) * np.linalg.norm(V_np[:, i]) * np.linalg.norm(W_np[:, i])
            mag_j = np.linalg.norm(U_np[:, j]) * np.linalg.norm(V_np[:, j]) * np.linalg.norm(W_np[:, j])
            if mag_i >= mag_j:
                to_remove.add(j)
            else:
                to_remove.add(i)
        
        if to_remove:
            keep = [k for k in range(rank) if k not in to_remove]
            new_rank = len(keep)
            if new_rank < best_rank:
                best_rank = new_rank
                best_indices = keep
    
    # Also check for triples: three rank-1 tensors that sum to zero
    # u1⊗v1⊗w1 + u2⊗v2⊗w2 + u3⊗v3⊗w3 = 0
    # This is harder to check directly. We can use the fact that if we fix u1, v1, w1,
    # then u2⊗v2⊗w2 + u3⊗v3⊗w3 = -u1⊗v1⊗w1
    # The right side is a rank-1 tensor. The left side is a sum of two rank-1 tensors.
    # A sum of two rank-1 tensors equals a rank-1 tensor iff... this is complex.
    
    # Alternative approach: use SVD-like decomposition on the tensor
    # Reshape the tensor into a matrix and find its rank
    # The tensor T has entries T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We can check if the decomposition is minimal by verifying that no column
    # can be expressed as a linear combination of others in a way that preserves
    # the tensor structure.
    
    # For now, use the pair-based reduction if found
    if best_rank < rank:
        U_reduced = U_np[:, best_indices]
        V_reduced = V_np[:, best_indices]
        W_reduced = W_np[:, best_indices]
        
        # Convert back to list format
        U_out = U_reduced.tolist()
        V_out = V_reduced.tolist()
        W_out = W_reduced.tolist()
        
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # If no reduction found, return the original
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
