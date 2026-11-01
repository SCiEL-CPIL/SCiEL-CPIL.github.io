import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # Get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 20:
            break
        result = _search_low_rank(target_rank, remaining - 5)
        if result is not None:
            return result
    
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, nonzero_indices, num_a, num_b, num_c, tol=1e-6):
    """Verify that U, V, W form a valid decomposition using efficient einsum."""
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check non-zero entries (32 entries)
    for a, b, c in nonzero_indices:
        if abs(T_hat[a, b, c] - 1) > tol:
            return False
    
    # Check zero entries using vectorized approach
    # Build target tensor
    T_target = np.zeros((num_a, num_b, num_c), dtype=float)
    for a, b, c in nonzero_indices:
        T_target[a, b, c] = 1.0
    
    # Check all entries
    if np.max(np.abs(T_hat - T_target)) > tol:
        return False
    
    return True

def _search_low_rank(target_rank, time_budget):
    """Search for a valid decomposition of given rank using structured approach."""
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the list of (a,b,c) indices where T=1
    nonzero_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                nonzero_indices.append((a, b, c))
    
    nonzero_set = set(nonzero_indices)
    
    # Build target tensor for fast verification
    T_target = np.zeros((num_a, num_b, num_c), dtype=float)
    for a, b, c in nonzero_indices:
        T_target[a, b, c] = 1.0
    
    # Strategy: Exploit the structure where A is 2x2 (shared across all 8 columns)
    # Use structured random search with the constraint that U factors for
    # different k-blocks should be related
    
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        rng = np.random.default_rng(seed=restart * 7919 + 12345)
        
        # Structured initialization: use Strassen-like patterns
        # For rank R, try to build from 7 Strassen terms per group
        # with shared U structure
        
        if target_rank >= 24:
            # Use 4 groups of 6-7 terms each
            num_groups = 4
            terms_per_group = target_rank // num_groups
            remainder = target_rank % num_groups
            
            U = np.zeros((num_a, target_rank), dtype=float)
            V = np.zeros((num_b, target_rank), dtype=float)
            W = np.zeros((num_c, target_rank), dtype=float)
            
            col = 0
            for g in range(num_groups):
                n_terms = terms_per_group + (1 if g < remainder else 0)
                for t in range(n_terms):
                    # Random Strassen-like pattern
                    u_vals = rng.choice([-1, 0, 1], size=4)
                    v_vals = rng.choice([-1, 0, 1], size=4)
                    w_vals = rng.choice([-1, 0, 1], size=4)
                    
                    U[0, col] = u_vals[0]
                    U[1, col] = u_vals[1]
                    U[2, col] = u_vals[2]
                    U[3, col] = u_vals[3]
                    
                    # V: 16 entries, structured as 2x8
                    for k in range(8):
                        V[0*8 + k, col] = v_vals[0] if (k % 2 == g % 2) else 0
                        V[1*8 + k, col] = v_vals[1] if (k % 2 == g % 2) else 0
                        # Add some randomness
                        if rng.random() < 0.3:
                            V[0*8 + k, col] += rng.choice([-1, 1])
                            V[1*8 + k, col] += rng.choice([-1, 1])
                    
                    # W: 16 entries, structured as 8x2
                    for k in range(8):
                        W[k*2 + 0, col] = w_vals[0] if (k % 2 == g % 2) else 0
                        W[k*2 + 1, col] = w_vals[1] if (k % 2 == g % 2) else 0
                        if rng.random() < 0.3:
                            W[k*2 + 0, col] += rng.choice([-1, 1])
                            W[k*2 + 1, col] += rng.choice([-1, 1])
                    
                    col += 1
            
            # Quick error check
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_hat - T_target))
            
            if err < 2.0:
                # Try refinement
                U_ref, V_ref, W_ref = _refine(U, V, W, nonzero_indices, nonzero_set,
                                               num_a, num_b, num_c, time_budget, start, T_target)
                if U_ref is not None:
                    if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices,
                                             num_a, num_b, num_c, tol=1e-6):
                        # Round to half-integers
                        U_ref = np.round(U_ref * 2) / 2
                        V_ref = np.round(V_ref * 2) / 2
                        W_ref = np.round(W_ref * 2) / 2
                        if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices,
                                                 num_a, num_b, num_c, tol=1e-6):
                            return {"U": U_ref.tolist(), "V": V_ref.tolist(),
                                    "W": W_ref.tolist()}
    
    # Try structured approach: build from 2x2 Strassen with shared factors
    result = _structured_search(target_rank, time_budget, start, T_target, nonzero_indices)
    if result is not None:
        return result
    
    return None

def _refine(U, V, W, nonzero_indices, nonzero_set, num_a, num_b, num_c, 
            time_budget, start, T_target=None):
    """Refine a near-solution using gradient descent with vectorized operations."""
    if T_target is None:
        T_target = np.zeros((num_a, num_b, num_c), dtype=float)
        for a, b, c in nonzero_indices:
            T_target[a, b, c] = 1.0
    
    max_iters = 200
    lr = 0.05
    
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    
    for it in range(max_iters):
        if time.time() - start > time_budget - 5:
            break
        
        # Compute T_hat
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_hat - T_target
        
        # Vectorized gradient computation
        # grad_U[a,r] = sum_{b,c} V[b,r]*W[c,r]*diff[a,b,c]
        # = einsum('br,cr,abc->ar', V, W, diff)
        grad_U = np.einsum('br,cr,abc->ar', V, W, diff)
        grad_V = np.einsum('ar,cr,abc->br', U, W, diff)
        grad_W = np.einsum('ar,br,abc->cr', U, V, diff)
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        T_hat_new = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_hat_new - T_target))
        
        if max_err < 1e-6:
            return U, V, W
        
        # Decay learning rate
        lr *= 0.95
    
    return None, None, None

def _structured_search(target_rank, time_budget, start, T_target, nonzero_indices):
    """Structured search exploiting the 2x2 matrix multiplication structure.
    
    Key insight: The tensor T[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
    can be decomposed by noting that for each k, we're doing 2x2 * 2x1, but A is shared.
    
    We use a basis approach: find R rank-1 tensors that span the multiplication tensor.
    """
    num_a, num_b, num_c = 4, 16, 16
    nonzero_set = set(nonzero_indices)
    
    # Approach 1: Use the fact that Strassen for 2x2x2 has rank 7.
    # For 2x2x8, we can potentially do better than 4*7=28 by sharing.
    
    # Let's try a systematic construction:
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # Index: a = i*2+j, b = j*8+k, c = k*2+i
    
    # For a rank-R decomposition, we need:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: Use random matrices with the right sparsity pattern
    # and solve for the remaining factors.
    
    rng = np.random.default_rng(seed=999)
    
    for attempt in range(500):
        if time.time() - start > time_budget:
            break
        
        # Initialize U with random sparse patterns
        U = np.zeros((num_a, target_rank), dtype=float)
        for r in range(target_rank):
            # Each column of U has 2-3 nonzeros
            nnz = rng.integers(2, 4)
            rows = rng.choice(num_a, size=nnz, replace=False)
            U[rows, r] = rng.choice([-1, 0.5, 1, 2], size=nnz)
        
        # For each (a,c) pair where T[a,?,c] is nonzero, we need to find b such that
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # This is a system of equations in V and W.
        
        # Try to solve: for fixed U, find V and W such that the tensor matches.
        # This is a bilinear system, so we iterate.
        
        V = rng.choice([-1, 0, 1], size=(num_b, target_rank))
        W = rng.choice([-1, 0, 1], size=(num_c, target_rank))
        
        # Iterate: fix V, solve for W; fix W, solve for V
        for inner in range(50):
            if time.time() - start > time_budget:
                break
            
            # Fix V, solve for W:
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For each (a,b,c) in nonzero: sum_r (U[a,r]*V[b,r])*W[c,r] = 1
            # This is linear in W for fixed U,V.
            
            # Build linear system for W:
            # For each c, we have equations from all (a,b) pairs.
            # Let M_c be a matrix where M_c[(a,b), r] = U[a,r]*V[b,r]
            # Then sum_r M_c[(a,b),r] * W[c,r] = T[a,b,c]
            
            # Actually, let's use least squares.
            # For each c, collect all (a,b) such that T[a,b,c] is defined.
            
            W_new = np.zeros_like(W)
            for c in range(num_c):
                # Find all (a,b) where T[a,b,c] is nonzero
                eqs_A = []
                eqs_b = []
                for a in range(num_a):
                    for b in range(num_b):
                        if (a, b, c) in nonzero_set:
                            eqs_A.append(U[a, :] * V[b, :])
                            eqs_b.append(1.0)
                        else:
                            eqs_A.append(U[a, :] * V[b, :])
                            eqs_b.append(0.0)
                
                if len(eqs_A) > 0:
                    A_mat = np.array(eqs_A)  # (num_eqs, target_rank)
                    b_vec = np.array(eqs_b)
                    # Solve A_mat @ W[c,:] = b_vec
                    try:
                        W_new[c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        W_new[c, :] = W[c, :]
            
            # Fix W, solve for V:
            V_new = np.zeros_like(V)
            for b in range(num_b):
                eqs_A = []
                eqs_b = []
                for a in range(num_a):
                    for c in range(num_c):
                        if (a, b, c) in nonzero_set:
                            eqs_A.append(U[a, :] * W[c, :])
                            eqs_b.append(1.0)
                        else:
                            eqs_A.append(U[a, :] * W[c, :])
                            eqs_b.append(0.0)
                
                if len(eqs_A) > 0:
                    A_mat = np.array(eqs_A)
                    b_vec = np.array(eqs_b)
                    try:
                        V_new[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        V_new[b, :] = V[b, :]
            
            V, W = V_new, W_new
            
            # Check convergence
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = np.max(np.abs(T_hat - T_target))
            
            if max_err < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_decomposition(U_r, V_r, W_r, nonzero_indices,
                                         num_a, num_b, num_c, tol=1e-6):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try refinement
        U_ref, V_ref, W_ref = _refine(U, V, W, nonzero_indices, nonzero_set,
                                       num_a, num_b, num_c, time_budget, start, T_target)
        if U_ref is not None:
            if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices,
                                     num_a, num_b, num_c, tol=1e-6):
                U_ref = np.round(U_ref * 2) / 2
                V_ref = np.round(V_ref * 2) / 2
                W_ref = np.round(W_ref * 2) / 2
                if _verify_decomposition(U_ref, V_ref, W_ref, nonzero_indices,
                                         num_a, num_b, num_c, tol=1e-6):
                    return {"U": U_ref.tolist(), "V": V_ref.tolist(), "W": W_ref.tolist()}
    
    return None
