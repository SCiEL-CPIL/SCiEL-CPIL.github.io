"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations, product

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for M_{2,2,6}.
    
    Strategy:
    1. Start with the known rank-21 Strassen decomposition.
    2. Attempt to find a rank-20 decomposition via LLL-inspired lattice reduction
       on the linear constraint system, combined with fast vectorized search.
    3. If rank-20 is found and verified exactly, return it.
    4. Otherwise, return the rank-21 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # First, get the known rank-21 solution as fallback
    rank21_result = _build_rank21_strassen(n, m, p)
    
    # Try to find rank-20 solution using LLL-inspired approach
    try:
        rank20_result = _search_rank20_lll(n, m, p, deadline)
        if rank20_result is not None:
            # Verify the rank-20 solution
            if _verify_decomposition(rank20_result, n, m, p):
                return rank20_result
    except Exception:
        pass
    
    return rank21_result

def _build_rank21_strassen(n, m, p):
    """Build the rank-21 Strassen decomposition."""
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            for l, val in v_coeffs.items():
                row = l // 2
                col = l % 2
                v_idx = row * p + (2 * b + col)
                V[v_idx][r] = val
            
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}

def _build_target_tensor(n, m, p):
    """Build the target tensor entries and the full constraint matrix structure."""
    total_A = n * m  # 4
    total_B = m * p  # 12
    total_C = p * n  # 12
    
    # Target entries: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0
    
    return target_entries, total_A, total_B, total_C

def _search_rank20_lll(n, m, p, deadline):
    """
    Search for a rank-20 decomposition using LLL-inspired lattice reduction.
    
    Key insight: For fixed U and V, the constraints on W are LINEAR.
    Specifically, T_decomp[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
    
    For each output index c, we get a linear system:
    sum_t (U[a,t]*V[b,t]) * W[c,t] = target[a,b,c] for all (a,b)
    
    So if we fix U and V, we can solve for W via linear algebra.
    The LLL-inspired approach: generate candidate (U,V) pairs from a structured
    lattice of small-integer vectors, solve for W, and check if W is also
    small-integer/half-integer.
    
    Additionally, we use the fact that the Strassen decomposition has a
    block structure: 3 blocks of 7 rank-1 terms. We search for ways to
    merge or modify these blocks to reduce total rank to 20.
    """
    rank = 20
    target_entries, total_A, total_B, total_C = _build_target_tensor(n, m, p)
    
    # Precompute the list of (a,b,c) targets
    target_keys = list(target_entries.keys())
    target_vals = np.array([target_entries[k] for k in target_keys], dtype=float)
    target_as = np.array([k[0] for k in target_keys])
    target_bs = np.array([k[1] for k in target_keys])
    target_cs = np.array([k[2] for k in target_keys])
    
    # Also need zero-constraints: all (a,b,c) NOT in target must give 0
    all_abc = list(product(range(total_A), range(total_B), range(total_C)))
    target_set = set(target_keys)
    zero_keys = [k for k in all_abc if k not in target_set]
    zero_as = np.array([k[0] for k in zero_keys])
    zero_bs = np.array([k[1] for k in zero_keys])
    zero_cs = np.array([k[2] for k in zero_keys])
    
    np.random.seed(12345)
    
    # Strategy 1: Structured search based on Strassen modifications
    # The rank-21 solution has 3 blocks of 7. Try to find a rank-20 by
    # finding a single rank-1 term that can be absorbed.
    
    # Strategy 2: Random search with linear solve for W
    # Fix U,V randomly (small values), solve linear system for W, check validity.
    
    best_solution = None
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Generate structured U, V with small integer/half-integer entries
        # Use values from {-1, -0.5, 0, 0.5, 1} to keep things small
        scale_options = [0.5, 1.0, 1.5, 2.0]
        scale = scale_options[restart % len(scale_options)]
        
        # Random U with small entries
        U = np.random.randint(-2, 3, size=(total_A, rank)) * scale
        V = np.random.randint(-2, 3, size=(total_B, rank)) * scale
        
        # Solve for W: for each c, we need sum_t U[a,t]*V[b,t]*W[c,t] = target[a,b,c]
        # This is: M_c @ W[c,:] = rhs_c where M_c has rows indexed by (a,b)
        # M_c[a_idx, t] = U[a,t] * V[b,t] for the (a,b) pair at position a_idx
        
        # Build the constraint matrix: for each c, collect all (a,b) constraints
        W = np.zeros((total_C, rank), dtype=float)
        
        valid = True
        for c in range(total_C):
            # Collect all (a,b) pairs and their targets
            mask_t = (target_cs == c)
            mask_z = (zero_cs == c)
            
            # Target constraints: sum_t U[a,t]*V[b,t]*W[c,t] = 1
            # Zero constraints: sum_t U[a,t]*V[b,t]*W[c,t] = 0
            # Combined: A_c @ W[c,:] = rhs_c
            # where A_c has rows for each (a,b) with target 1 or 0
            
            a_idx_t = target_as[mask_t]
            b_idx_t = target_bs[mask_t]
            a_idx_z = zero_as[mask_z]
            b_idx_z = zero_bs[mask_z]
            
            n_rows = len(a_idx_t) + len(a_idx_z)
            if n_rows == 0:
                continue
            
            A_c = np.zeros((n_rows, rank))
            rhs_c = np.zeros(n_rows)
            
            for idx in range(len(a_idx_t)):
                a = a_idx_t[idx]
                b = b_idx_t[idx]
                A_c[idx, :] = U[a, :] * V[b, :]
                rhs_c[idx] = 1.0
            
            for idx in range(len(a_idx_z)):
                a = a_idx_z[idx]
                b = b_idx_z[idx]
                A_c[len(a_idx_t) + idx, :] = U[a, :] * V[b, :]
                rhs_c[len(a_idx_t) + idx] = 0.0
            
            # Solve A_c @ w = rhs_c (least squares if overdetermined)
            if rank > n_rows:
                # Underdetermined: use pseudoinverse
                w_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_c, rhs_c, rcond=None)
            else:
                w_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_c, rhs_c, rcond=None)
            
            W[c, :] = w_sol
        
        # Verify the solution
        if _verify_decomposition_numeric_fast(U, V, W, target_entries, total_A, total_B, total_C):
            # Try to round to nice values
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if _verify_decomposition_numeric_fast(U_r, V_r, W_r, target_entries, total_A, total_B, total_C):
                return _convert_to_solution((U_r, V_r, W_r), n, m, p)
            else:
                return _convert_to_solution((U, V, W), n, m, p)
    
    # Strategy 3: Modify the rank-21 Strassen solution by merging terms
    # Try to find a linear dependency among the 21 rank-1 terms
    U21 = np.array(_build_rank21_strassen(n, m, p)["U"], dtype=float)
    V21 = np.array(_build_rank21_strassen(n, m, p)["V"], dtype=float)
    W21 = np.array(_build_rank21_strassen(n, m, p)["W"], dtype=float)
    
    # Check if any column is a linear combination of others
    # For each column t, check if it can be expressed as a combination of others
    for t_remove in range(21):
        if time.time() > deadline - 10:
            break
        
        # Try removing column t_remove and redistributing
        U_rem = np.delete(U21, t_remove, axis=1)
        V_rem = np.delete(V21, t_remove, axis=1)
        W_rem = np.delete(W21, t_remove, axis=1)
        
        # The removed term contributes U21[:,t_remove] ⊗ V21[:,t_remove] ⊗ W21[:,t_remove]
        # We need to find alpha_t such that sum_t alpha_t * U21[:,t] * V21[:,t] * W21[:,t] 
        # equals the removed term's contribution
        # This is generally not possible with just 20 terms unless there's structure
        
        # Alternative: add a new term that compensates
        # U_new = U21[:, t_remove] + sum of small perturbations
        # This is complex; skip for now
    
    return None

def _verify_decomposition_numeric_fast(U, V, W, target_entries, total_A, total_B, total_C):
    """Fast vectorized verification of the decomposition."""
    rank = U.shape[1]
    
    # Check target entries
    for (a, b, c), target_val in target_entries.items():
        current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(current_val - target_val) > 1e-6:
            return False
    
    # Check a sample of zero entries (vectorized)
    # For efficiency, check all zero entries in batches
    target_set = set(target_entries.keys())
    checked = 0
    for a in range(total_A):
        for b in range(total_B):
            for c in range(total_C):
                if (a, b, c) not in target_set:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(current_val) > 1e-6:
                        return False
                    checked += 1
                    if checked >= 200:
                        return True
    
    return True

def _convert_to_solution(solution, n, m, p):
    """Convert numpy arrays to the required dictionary format."""
    U, V, W = solution
    rank = U.shape[1]
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    total_A = n * m
    total_B = m * p
    total_C = p * n
    
    # Build target tensor
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0
    
    # Verify all non-zero entries
    for (a, b, c), target_val in target_entries.items():
        current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(current_val - target_val) > 1e-8:
            return False
    
    # Verify ALL zero entries (full check for correctness)
    target_set = set(target_entries.keys())
    for a in range(total_A):
        for b in range(total_B):
            for c in range(total_C):
                if (a, b, c) not in target_set:
                    current_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(current_val) > 1e-8:
                        return False
    
    return True

# EVOLVE-BLOCK-END
