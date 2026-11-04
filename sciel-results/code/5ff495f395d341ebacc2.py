"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
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

def _probabilistic_rank16_search(U17, V17, W17, T, deadline, n_trials=5000):
    """
    Probabilistic algebraic search for rank-16 decomposition.
    
    Strategy: Randomly sample sparse support patterns for 16 rank-1 tensors,
    constrained by the known rank-17 Strassen structure. For each support pattern,
    solve the linear system to find coefficients. If an exact solution exists,
    return the rank-16 decomposition.
    
    The key insight: a rank-16 decomposition must have each term's support
    contained in a "superset" of some subset of the rank-17 terms' supports.
    We sample by:
    1. Starting from the 16 terms (excluding one random term from rank-17)
    2. Randomly adding/removing entries from the support pattern
    3. Solving the resulting linear system
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank17 = U17.shape[1]
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # Precompute the contribution of each rank-17 term
    term_contribs = []
    for r in range(rank17):
        term_contribs.append(
            np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
        )
    
    # Get supports for each term in rank-17
    supports_U = [set(np.nonzero(U17[:, r])[0]) for r in range(rank17)]
    supports_V = [set(np.nonzero(V17[:, r])[0]) for r in range(rank17)]
    supports_W = [set(np.nonzero(W17[:, r])[0]) for r in range(rank17)]
    
    # All possible indices
    all_U_indices = set(range(n_a))
    all_V_indices = set(range(n_b))
    all_W_indices = set(range(n_c))
    
    trial_count = 0
    
    while trial_count < n_trials and time.time() < deadline:
        trial_count += 1
        
        # Choose which term to exclude from rank-17
        remove_idx = trial_count % rank17
        keep_indices = [r for r in range(rank17) if r != remove_idx]
        
        # Start with the 16 kept terms' supports
        U_supports = [supports_U[r].copy() for r in keep_indices]
        V_supports = [supports_V[r].copy() for r in keep_indices]
        W_supports = [supports_W[r].copy() for r in keep_indices]
        
        # Randomly perturb the supports: add or remove entries
        # Number of perturbations varies
        n_perturbations = random.randint(0, 4)
        
        for _ in range(n_perturbations):
            # Choose a random term to perturb
            j = random.randint(0, 15)
            # Choose which matrix to perturb
            mat_idx = random.randint(0, 2)
            
            if mat_idx == 0:
                current_support = U_supports[j]
                all_indices = all_U_indices
                # Decide add or remove
                if current_support and random.random() < 0.4:
                    # Remove an entry
                    to_remove = random.choice(list(current_support))
                    current_support.discard(to_remove)
                else:
                    # Add an entry
                    candidates = list(all_indices - current_support)
                    if candidates:
                        to_add = random.choice(candidates)
                        current_support.add(to_add)
            elif mat_idx == 1:
                current_support = V_supports[j]
                all_indices = all_V_indices
                if current_support and random.random() < 0.4:
                    to_remove = random.choice(list(current_support))
                    current_support.discard(to_remove)
                else:
                    candidates = list(all_indices - current_support)
                    if candidates:
                        to_add = random.choice(candidates)
                        current_support.add(to_add)
            else:
                current_support = W_supports[j]
                all_indices = all_W_indices
                if current_support and random.random() < 0.4:
                    to_remove = random.choice(list(current_support))
                    current_support.discard(to_remove)
                else:
                    candidates = list(all_indices - current_support)
                    if candidates:
                        to_add = random.choice(candidates)
                        current_support.add(to_add)
        
        # Build the design matrix A where A[:, j] = tensor product of supports
        A = np.zeros((num_entries, 16))
        for j in range(16):
            # Create sparse vectors from supports
            u_vec = np.zeros(n_a)
            v_vec = np.zeros(n_b)
            w_vec = np.zeros(n_c)
            for idx in U_supports[j]:
                u_vec[idx] = 1
            for idx in V_supports[j]:
                v_vec[idx] = 1
            for idx in W_supports[j]:
                w_vec[idx] = 1
            
            # Compute outer product
            A[:, j] = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec).flatten()
        
        # Check if A has full column rank (needed for unique solution)
        try:
            rank_A = np.linalg.matrix_rank(A)
            if rank_A < 16:
                continue  # Skip if not full rank
        except:
            continue
        
        # Solve the linear system
        try:
            alpha, residuals, rank_solve, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            T_recon = A @ alpha
            if np.allclose(T_recon, T_flat, atol=1e-9):
                # Found a valid rank-16 decomposition!
                U16 = np.zeros((n_a, 16))
                V16 = np.zeros((n_b, 16))
                W16 = np.zeros((n_c, 16))
                for j in range(16):
                    u_vec = np.zeros(n_a)
                    v_vec = np.zeros(n_b)
                    w_vec = np.zeros(n_c)
                    for idx in U_supports[j]:
                        u_vec[idx] = 1
                    for idx in V_supports[j]:
                        v_vec[idx] = 1
                    for idx in W_supports[j]:
                        w_vec[idx] = 1
                    U16[:, j] = alpha[j] * u_vec
                    V16[:, j] = alpha[j] * v_vec
                    W16[:, j] = alpha[j] * w_vec
                return U16, V16, W16
        except:
            continue
    
    return None

def _structured_rank16_search(start_time, deadline):
    """
    Structured search for rank-16 using probabilistic algebraic approach.
    Samples sparse support patterns and solves linear systems.
    """
    n, m, p = 2, 3, 3
    T = _build_tensor(n, m, p)
    U17, V17, W17 = _get_rank17_solution()
    
    search_deadline = deadline - 5
    
    # Use probabilistic algebraic search
    result = _probabilistic_rank16_search(U17, V17, W17, T, search_deadline, n_trials=10000)
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
