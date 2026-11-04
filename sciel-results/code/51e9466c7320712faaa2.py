import time
import sys
import random
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 and 42
    for target_rank in [43, 42]:
        if time.time() > deadline - 20:
            break
            
        # Attempt search for target_rank
        found = _search_for_rank(target_rank, deadline)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
            break # If we find 43, we might still try 42 if time permits, but 42 is better.
            
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[u_map[0], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        W[w_map[2], col] += 1
        W[w_map[3], col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0], col] += 1
        V[v_map[1], col] += 1
        V[v_map[3], col] += -1
        W[w_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3], col] += 1
        V[v_map[2], col] += 1
        V[v_map[0], col] += -1
        W[w_map[0], col] += 1
        W[w_map[2], col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0], col] += 1
        U[u_map[1], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += -1
        W[w_map[1], col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2], col] += 1
        U[u_map[0], col] += -1
        V[v_map[0], col] += 1
        V[v_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1], col] += 1
        U[u_map[3], col] += -1
        V[v_map[2], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        col += 1

    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),
            1: 0 * p + (b_start + 1),
            2: 1 * p + (b_start + 0),
            3: 1 * p + (b_start + 1)
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,
            1: (b_start + 1) * n + 0,
            2: (b_start + 0) * n + 1,
            3: (b_start + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (half * 4 + k)
                w_idx = (half * 4 + k) * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _search_for_rank(target_rank, deadline):
    """
    Try to find a decomposition of the given rank by starting from the rank-44 solution
    and attempting to eliminate (target_rank - target_rank) terms.
    
    The idea: if we have 44 terms and want rank r, we need to show that 44-r of the terms
    can be expressed as linear combinations of the other r terms, while preserving the
    tensor equation.
    """
    n, m, p = 2, 3, 8
    U44, V44, W44 = _get_rank_44_solution()
    r44 = 44
    num_eliminate = r44 - target_rank
    
    if num_eliminate <= 0:
        return None
    
    # For each subset of terms to eliminate, try to find coefficients that
    # allow us to rewrite the eliminated terms in terms of the kept terms.
    
    # The tensor equation is: for all (i,j,k), sum_t U[i,t]*V[j,t]*W[k,t] = delta_{i,j,k} (where delta is 1 if (i,j,k) corresponds to a valid multiplication term)
    
    # Let's define the target tensor T[i,j,k] = 1 if the term (i,j,k) is part of the matrix multiplication
    # i in [0, n*m) = [0, 6), j in [0, m*p) = [0, 24), k in [0, p*n) = [0, 16)
    # T[i,j,k] = 1 if i//m == j%p (i.e., the middle index matches)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        for j in range(m*p):
            if i // m == j % p:
                k = (j // p) * n + (i % m)
                T[i, j, k] = 1
    
    # For a given subset of terms to eliminate, we need to find:
    # For each eliminated term t_elim, coefficients c[t_elim][t_keep] such that:
    # For all (i,j,k): U[i,t_elim]*V[j,t_elim]*W[k,t_elim] = sum_{t_keep} c[t_elim][t_keep] * U[i,t_keep]*V[j,t_keep]*W[k,t_keep]
    
    # This is a system of equations. The unknowns are c[t_elim][t_keep].
    # For each (i,j,k), we get an equation.
    
    # Let's try all subsets of size num_eliminate to eliminate.
    # For num_eliminate = 1 or 2, this is feasible.
    
    from itertools import combinations
    
    # Precompute U, V, W as numpy arrays
    U_arr = np.array(U44, dtype=float)
    V_arr = np.array(V44, dtype=float)
    W_arr = np.array(W44, dtype=float)
    
    # For each choice of terms to eliminate
    for elim_set in combinations(range(r44), num_eliminate):
        if time.time() > deadline - 10:
            return None
        
        keep_set = [t for t in range(r44) if t not in elim_set]
        
        # Build the system: for each (i,j,k) where T[i,j,k] != 0, we need:
        # sum_{t in elim_set} U[i,t]*V[j,t]*W[k,t] = sum_{t in keep_set} c[t_elim][t_keep] * U[i,t_keep]*V[j,t_keep]*W[k,t_keep]
        
        # Actually, the full equation is:
        # sum_{t in all} U[i,t]*V[j,t]*W[k,t] = T[i,j,k]
        # If we want to eliminate term t_e, we need:
        # U[i,t_e]*V[j,t_e]*W[k,t_e] = sum_{t_k in keep} c[t_e,t_k] * U[i,t_k]*V[j,t_k]*W[k,t_k]
        
        # Let's set up the linear system for the coefficients.
        # Unknowns: for each t_e in elim_set, for each t_k in keep_set, a coefficient c[t_e][t_k]
        
        num_unknowns = len(elim_set) * len(keep_set)
        
        # For each (i,j,k) where T[i,j,k] = 1, we get an equation.
        # But we only need to satisfy the equation for the eliminated terms.
        
        # Let's build the matrix A and vector b.
        # A has rows for each (i,j,k) and columns for each (t_e, t_k) pair.
        # b has entries for each (i,j,k) which is U[i,t_e]*V[j,t_e]*W[k,t_e] for the eliminated terms.
        
        # Actually, for each eliminated term t_e, we need a separate system.
        # Let's handle one eliminated term at a time.
        
        valid = True
        new_U = U_arr.copy()
        new_V = V_arr.copy()
        new_W = W_arr.copy()
        
        for t_e in elim_set:
            # For this eliminated term, find coefficients c[t_k] for t_k in keep_set
            # such that: U[:,t_e] * V[:,t_e] * W[:,t_e] (as a tensor) = sum_{t_k} c[t_k] * U[:,t_k] * V[:,t_k] * W[:,t_k]
            
            # This is a bilinear equation. Let's set it up as a linear system.
            # For each (i,j,k): U[i,t_e]*V[j,t_e]*W[k,t_e] = sum_{t_k} c[t_k] * U[i,t_k]*V[j,t_k]*W[k,t_k]
            
            # Build the system.
            # Unknowns: c[t_k] for t_k in keep_set
            num_keep = len(keep_set)
            
            # Build A matrix: rows are (i,j,k), columns are t_k
            A = []
            b = []
            
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        lhs = U_arr[i, t_e] * V_arr[j, t_e] * W_arr[k, t_e]
                        if abs(lhs) < 1e-10:
                            continue
                        row = []
                        for t_k in keep_set:
                            row.append(U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k])
                        A.append(row)
                        b.append(lhs)
            
            if not A:
                valid = False
                break
            
            A = np.array(A)
            b = np.array(b)
            
            # Solve the least squares problem
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                valid = False
                break
            
            # Check if the solution is good
            if residuals.size > 0:
                if residuals[0] > 1e-8 * (np.sum(b**2) + 1e-10):
                    valid = False
                    break
            
            # If valid, we can eliminate this term by adding the contributions
            # to the kept terms.
            for idx, t_k in enumerate(keep_set):
                if abs(c[idx]) > 1e-10:
                    new_U[:, t_k] += c[idx] * new_U[:, t_e]
                    new_V[:, t_k] += c[idx] * new_V[:, t_e]
                    new_W[:, t_k] += c[idx] * new_W[:, t_e]
            
            # Zero out the eliminated term
            new_U[:, t_e] = 0
            new_V[:, t_e] = 0
            new_W[:, t_e] = 0
        
        if not valid:
            continue
        
        # Verify the solution
        # Check that the tensor equation is satisfied with the new matrices
        # (ignoring the eliminated terms which are zero)
        
        error = 0
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    computed = 0
                    for t in keep_set:
                        computed += new_U[i, t] * new_V[j, t] * new_W[k, t]
                    expected = T[i, j, k]
                    error += (computed - expected) ** 2
        
        if error < 1e-6:
            # Found a valid solution!
            # Convert back to lists
            result_U = new_U.tolist()
            result_V = new_V.tolist()
            result_W = new_W.tolist()
            
            # Remove the eliminated terms to get the final rank
            final_U = []
            final_V = []
            final_W = []
            for t in keep_set:
                final_U.append([result_U[i][t] for i in range(n*m)])
                final_V.append([result_V[i][t] for i in range(m*p)])
                final_W.append([result_W[i][t] for i in range(p*n)])
            
            # Transpose to get the correct shape
            final_U = np.array(final_U).T.tolist()
            final_V = np.array(final_V).T.tolist()
            final_W = np.array(final_W).T.tolist()
            
            return final_U, final_V, final_W
    
    return None
