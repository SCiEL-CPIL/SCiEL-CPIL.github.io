"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_absorb_term(U17, V17, W17, T, remove_idx, deadline):
    """
    Try to absorb the contribution of term `remove_idx` into the other 16 terms.
    Uses a multi-stage approach:
    1. Direct least-squares on the 16 remaining terms
    2. Single-entry perturbations (add or remove)
    3. Pairs of single-entry perturbations
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank17 = U17.shape[1]
    
    # Build the 16 rank-1 tensors (excluding remove_idx)
    keep_indices = [r for r in range(rank17) if r != remove_idx]
    
    # Flatten T
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Helper to build A matrix from vectors
    def build_A(U_vecs, V_vecs, W_vecs):
        A = np.zeros((num_entries, len(keep_indices)))
        for j in range(len(keep_indices)):
            A[:, j] = np.einsum('a,b,c->abc', U_vecs[:, j], V_vecs[:, j], W_vecs[:, j]).flatten()
        return A
    
    # Helper to check if solution is valid and return it
    def check_and_return(A_mat):
        try:
            alpha, _, _, _ = np.linalg.lstsq(A_mat, T_flat, rcond=None)
            T_recon = A_mat @ alpha
            if np.allclose(T_recon, T_flat, atol=1e-9):
                U16 = np.zeros((n_a, 16))
                V16 = np.zeros((n_b, 16))
                W16 = np.zeros((n_c, 16))
                for j in range(16):
                    U16[:, j] = alpha[j] * U_vecs[:, j]
                    V16[:, j] = alpha[j] * V_vecs[:, j]
                    W16[:, j] = alpha[j] * W_vecs[:, j]
                return U16, V16, W16
        except:
            pass
        return None
    
    # Stage 1: Direct least squares on original 16 terms
    U_vecs = np.column_stack([U17[:, r] for r in keep_indices])
    V_vecs = np.column_stack([V17[:, r] for r in keep_indices])
    W_vecs = np.column_stack([W17[:, r] for r in keep_indices])
    
    A = build_A(U_vecs, V_vecs, W_vecs)
    result = check_and_return(A)
    if result is not None:
        return result
    
    # Get current supports
    supports_U = [set(np.nonzero(U17[:, r])[0]) for r in keep_indices]
    supports_V = [set(np.nonzero(V17[:, r])[0]) for r in keep_indices]
    supports_W = [set(np.nonzero(W17[:, r])[0]) for r in keep_indices]
    
    # Precompute all possible single-entry modifications
    # For each term j, for each matrix (U/V/W), for each index: add or remove
    modifications = []
    for j in range(16):
        for mat_idx in range(3):  # 0=U, 1=V, 2=W
            n_mat = [n_a, n_b, n_c][mat_idx]
            current_support = [supports_U[j], supports_V[j], supports_W[j]][mat_idx]
            for idx in range(n_mat):
                if idx in current_support:
                    modifications.append((j, mat_idx, idx, 'remove'))
                else:
                    modifications.append((j, mat_idx, idx, 'add'))
    
    # Stage 2: Try each single modification
    for (j, mat_idx, idx, op) in modifications:
        if time.time() > deadline:
            break
        
        U_test = U_vecs.copy()
        V_test = V_vecs.copy()
        W_test = W_vecs.copy()
        
        if mat_idx == 0:
            if op == 'add':
                U_test[idx, j] = 1
            else:
                U_test[idx, j] = 0
        elif mat_idx == 1:
            if op == 'add':
                V_test[idx, j] = 1
            else:
                V_test[idx, j] = 0
        else:
            if op == 'add':
                W_test[idx, j] = 1
            else:
                W_test[idx, j] = 0
        
        A_test = build_A(U_test, V_test, W_test)
        result = check_and_return(A_test)
        if result is not None:
            return result
    
    # Stage 3: Try pairs of modifications (limited to most promising)
    # Focus on modifications that affect terms with smaller support (more flexible)
    # Limit to avoid combinatorial explosion
    max_pairs = 500
    pair_count = 0
    for i in range(len(modifications)):
        if time.time() > deadline:
            break
        for j in range(i+1, len(modifications)):
            if pair_count >= max_pairs:
                break
            pair_count += 1
            
            mod1, mod2 = modifications[i], modifications[j]
            # Skip if same term and same matrix (conflicting)
            if mod1[0] == mod2[0] and mod1[1] == mod2[1]:
                continue
            
            U_test = U_vecs.copy()
            V_test = V_vecs.copy()
            W_test = W_vecs.copy()
            
            for (jj, mat_idx, idx, op) in [mod1, mod2]:
                if mat_idx == 0:
                    if op == 'add':
                        U_test[idx, jj] = 1
                    else:
                        U_test[idx, jj] = 0
                elif mat_idx == 1:
                    if op == 'add':
                        V_test[idx, jj] = 1
                    else:
                        V_test[idx, jj] = 0
                else:
                    if op == 'add':
                        W_test[idx, jj] = 1
                    else:
                        W_test[idx, jj] = 0
            
            A_test = build_A(U_test, V_test, W_test)
            result = check_and_return(A_test)
            if result is not None:
                return result
    
    return None

def _structured_rank16_search(start_time, deadline):
    """
    Structured search for rank-16: try absorbing each of the 17 terms into the other 16.
    Prioritize terms that are more likely to be absorbable (sparser terms first).
    """
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    U17, V17, W17 = _get_rank17_solution()
    
    search_deadline = deadline - 5
    
    # Compute sparsity of each term to prioritize
    sparsities = []
    for r in range(17):
        sparsity = (np.count_nonzero(U17[:, r]) + np.count_nonzero(V17[:, r]) + np.count_nonzero(W17[:, r]))
        sparsities.append(sparsity)
    
    # Sort by sparsity (sparser first, more likely to be absorbable)
    sorted_indices = sorted(range(17), key=lambda x: sparsities[x])
    
    for remove_idx in sorted_indices:
        if time.time() > search_deadline:
            break
        
        result = _try_absorb_term(U17, V17, W17, T, remove_idx, search_deadline)
        if result is not None:
            U16, V16, W16 = result
            if _verify_decomposition(U16, V16, W16, T):
                return U16, V16, W16
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # 1. Attempt structured rank-16 search
    result_16 = _structured_rank16_search(start_time, deadline)
    if result_16 is not None:
        U, V, W = result_16
        T = _build_tensor(n, m, p)
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    T = _build_tensor(n, m, p)
    
    if not _verify_decomposition(U, V, W, T):
        # Final fallback: Schoolbook (Rank 18)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
