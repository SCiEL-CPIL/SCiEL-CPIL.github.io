"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication."""
    terms = []
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _strassen_block_decomposition():
    """Construct a rank-42 decomposition using Strassen's algorithm."""
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=4, p=6):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_target[a_idx, b_idx, c_idx] = 1.0
    
    diff = np.max(np.abs(T_computed - T_target))
    return diff < 1e-10, diff

def _standard_decomposition():
    """Build the standard rank-48 decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_column_removal(U_init, V_init, W_init, max_time=60.0):
    """Try to reduce rank by removing columns and verifying."""
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    current_rank = U_np.shape[1]
    best_rank = current_rank
    
    removed_any = True
    while removed_any:
        removed_any = False
        for r_remove in range(current_rank):
            if time.time() - start_time > max_time * 0.85:
                break
            
            U_rem = np.delete(U_np, r_remove, axis=1)
            V_rem = np.delete(V_np, r_remove, axis=1)
            W_rem = np.delete(W_np, r_remove, axis=1)
            
            T_rem = np.einsum('ir,jr,kr->ijk', U_rem, V_rem, W_rem)
            residual = T_target - T_rem
            
            if np.abs(residual).max() < 1e-10:
                best_rank = min(best_rank, current_rank - 1)
                U_np = U_rem
                V_np = V_rem
                W_np = W_rem
                current_rank -= 1
                removed_any = True
                break
        
        if time.time() - start_time > max_time * 0.85:
            break
    
    if best_rank < 42:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }
    return None

def _try_pair_removal(U_init, V_init, W_init, max_time=60.0):
    """Try to reduce rank by removing pairs of columns."""
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    current_rank = U_np.shape[1]
    best_rank = current_rank
    
    for r1 in range(current_rank):
        if time.time() - start_time > max_time * 0.95:
            break
        for r2 in range(r1 + 1, current_rank):
            if time.time() - start_time > max_time * 0.95:
                break
            
            idx_keep = [x for x in range(current_rank) if x != r1 and x != r2]
            U_rem = U_np[:, idx_keep]
            V_rem = V_np[:, idx_keep]
            W_rem = W_np[:, idx_keep]
            
            T_rem = np.einsum('ir,jr,kr->ijk', U_rem, V_rem, W_rem)
            residual = T_target - T_rem
            
            if np.abs(residual).max() < 1e-10:
                best_rank = min(best_rank, current_rank - 2)
                U_np = U_rem
                V_np = V_rem
                W_np = W_rem
                current_rank -= 2
                break
        if current_rank < best_rank:
            break
    
    if best_rank < 42:
        return {
            "U": U_np.tolist(),
            "V": V_np.tolist(),
            "W": W_np.tolist()
        }
    return None

def _try_linear_combination_merge(U_init, V_init, W_init, max_time=60.0):
    """
    Structural change: try to merge two columns into one via linear combination.
    
    If we have two rank-1 tensors u1⊗v1⊗w1 and u2⊗v2⊗w2, we can try to
    replace them with a single rank-1 tensor u⊗v⊗w if there exist scalars
    α, β such that:
      u = α*u1 + β*u2
      v = α*v1 + β*v2  
      w = α*w1 + β*w2
    and the sum of the two original rank-1 tensors equals the new one plus
    a correction that can be absorbed into other terms.
    
    More practically: try to find a rank-1 tensor that is a linear combination
    of two existing columns, and check if removing both and adding the new one
    still gives a valid decomposition.
    """
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    rank = U_np.shape[1]
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try to merge pairs of columns
    for r1 in range(rank):
        if time.time() - start_time > max_time * 0.9:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() - start_time > max_time * 0.9:
                break
            
            # Try to find α, β such that the merged column works
            # This is a non-linear problem, so try a few simple ratios
            for alpha, beta in [(1, 1), (1, -1), (1, 0.5), (0.5, 1), (1, 2), (2, 1)]:
                # Create merged column
                u_merged = alpha * U_np[:, r1] + beta * U_np[:, r2]
                v_merged = alpha * V_np[:, r1] + beta * V_np[:, r2]
                w_merged = alpha * W_np[:, r1] + beta * W_np[:, r2]
                
                # Check if the merged column is non-trivial
                if np.linalg.norm(u_merged) < 1e-10 or np.linalg.norm(v_merged) < 1e-10:
                    continue
                
                # Build new matrices with merged column replacing both
                idx_keep = [x for x in range(rank) if x != r1 and x != r2]
                U_new = np.column_stack([U_np[:, idx_keep], u_merged])
                V_new = np.column_stack([V_np[:, idx_keep], v_merged])
                W_new = np.column_stack([W_np[:, idx_keep], w_merged])
                
                T_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                residual = T_target - T_new
                
                if np.abs(residual).max() < 1e-10:
                    new_rank = rank - 1
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
    
    if best_rank < 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    return None

def _try_random_perturbation_reduce(U_init, V_init, W_init, max_time=30.0):
    """
    Randomized local search: perturb the decomposition slightly and try to
    reduce rank by finding near-linearly-dependent columns.
    """
    start_time = time.time()
    U_np = np.array(U_init, dtype=float).copy()
    V_np = np.array(V_init, dtype=float).copy()
    W_np = np.array(W_init, dtype=float).copy()
    n, m, p = 2, 4, 6
    rank = U_np.shape[1]
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    np.random.seed(42)
    
    # Try to find columns that are approximately linearly dependent
    for trial in range(50):
        if time.time() - start_time > max_time * 0.95:
            break
        
        # Pick two random columns to try to merge
        r1, r2 = np.random.choice(rank, 2, replace=False)
        
        # Try to find a linear combination that makes the merged column work
        # Use least squares to find the best merge
        u_merged = U_np[:, r1] + U_np[:, r2]
        v_merged = V_np[:, r1] + V_np[:, r2]
        w_merged = W_np[:, r1] + W_np[:, r2]
        
        if np.linalg.norm(u_merged) < 1e-10 or np.linalg.norm(v_merged) < 1e-10:
            continue
        
        idx_keep = [x for x in range(rank) if x != r1 and x != r2]
        U_new = np.column_stack([U_np[:, idx_keep], u_merged])
        V_new = np.column_stack([V_np[:, idx_keep], v_merged])
        W_new = np.column_stack([W_np[:, idx_keep], w_merged])
        
        T_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
        residual = T_target - T_new
        
        if np.abs(residual).max() < 1e-10:
            new_rank = rank - 1
            if new_rank < best_rank:
                best_rank = new_rank
                best_U = U_new
                best_V = V_new
                best_W = W_new
    
    if best_rank < 42:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    return None

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Start with the known rank-42 decomposition
    result_42 = _strassen_block_decomposition()
    
    # Verify the initial decomposition
    valid, diff = _verify_decomposition(result_42["U"], result_42["V"], result_42["W"])
    
    if not valid:
        # Fallback to standard rank-48
        result_std = _standard_decomposition()
        valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
        if valid_std:
            return result_std
        return result_42
    
    best_result = result_42
    best_rank = 42
    
    # Try column removal with generous time budget
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        reduced = _try_column_removal(
            result_42["U"], result_42["V"], result_42["W"],
            max_time=min(20.0, deadline - elapsed - 20.0)
        )
        if reduced is not None:
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Try pair removal
    elapsed = time.time() - start_time
    if best_rank == 42 and elapsed < deadline - 20.0:
        reduced = _try_pair_removal(
            result_42["U"], result_42["V"], result_42["W"],
            max_time=min(15.0, deadline - elapsed - 15.0)
        )
        if reduced is not None:
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Try linear combination merge (structural change)
    elapsed = time.time() - start_time
    if best_rank == 42 and elapsed < deadline - 15.0:
        reduced = _try_linear_combination_merge(
            result_42["U"], result_42["V"], result_42["W"],
            max_time=min(15.0, deadline - elapsed - 10.0)
        )
        if reduced is not None:
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Try random perturbation reduce
    elapsed = time.time() - start_time
    if best_rank == 42 and elapsed < deadline - 10.0:
        reduced = _try_random_perturbation_reduce(
            result_42["U"], result_42["V"], result_42["W"],
            max_time=min(10.0, deadline - elapsed - 5.0)
        )
        if reduced is not None:
            valid_r, diff_r = _verify_decomposition(reduced["U"], reduced["V"], reduced["W"])
            if valid_r:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Final verification
    valid_final, _ = _verify_decomposition(best_result["U"], best_result["V"], best_result["W"])
    if not valid_final:
        result_std = _standard_decomposition()
        valid_std, _ = _verify_decomposition(result_std["U"], result_std["V"], result_std["W"])
        if valid_std:
            return result_std
    
    return best_result

# EVOLVE-BLOCK-END
