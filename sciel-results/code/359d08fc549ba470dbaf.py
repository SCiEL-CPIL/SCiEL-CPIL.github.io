"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the nm x mp x pn tensor T where T[a,b,c]=1 iff indices are consistent."""
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

def _get_rank17_terms():
    """Get the rank-17 decomposition terms as dictionaries."""
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    terms.append(({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1}))
    terms.append(({3:1, 4:1}, {0:1}, {1:1, 3:-1}))
    terms.append(({0:1}, {1:1, 4:-1}, {2:1, 3:1}))
    terms.append(({4:1}, {3:1, 0:-1}, {0:1, 1:1}))
    terms.append(({0:1, 1:1}, {4:1}, {0:-1, 2:1}))
    terms.append(({3:1, 0:-1}, {0:1, 1:1}, {3:1}))
    terms.append(({1:1, 4:-1}, {3:1, 4:1}, {0:1}))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    terms.append(({0:1}, {2:1}, {4:1}))
    terms.append(({1:1}, {5:1}, {4:1}))
    terms.append(({3:1}, {2:1}, {5:1}))
    terms.append(({4:1}, {5:1}, {5:1}))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    terms.append(({2:1}, {6:1}, {0:1}))
    terms.append(({2:1}, {7:1}, {2:1}))
    terms.append(({2:1}, {8:1}, {4:1}))
    terms.append(({5:1}, {6:1}, {1:1}))
    terms.append(({5:1}, {7:1}, {3:1}))
    terms.append(({5:1}, {8:1}, {5:1}))
    
    return terms

def _terms_to_matrices(terms, n_a, n_b, n_c):
    """Convert terms to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
    
    return U, V, W

def _try_rank16_polish(U, V, W, T, deadline, tol=1e-9):
    """Try to find a rank-16 decomposition by local optimization from rank-17."""
    start_time = time.time()
    n_a, n_b, n_c = U.shape[0], V.shape[0], W.shape[0]
    rank17 = U.shape[1]
    
    # Strategy: Try to express one term as a linear combination of others
    # This reduces rank by 1 if successful
    
    # For each pair of terms (i, j), try to find if term j can be written
    # as a linear combination of the other terms
    for j in range(rank17):
        if time.time() > deadline:
            break
        
        # Build the matrix of other terms
        other_indices = [i for i in range(rank17) if i != j]
        n_others = len(other_indices)
        
        # We want to find coefficients c_i such that:
        # sum_{i in others} c_i * (u_i, v_i, w_i) = (u_j, v_j, w_j)
        # This is a system of equations: for each (a,b,c) in T,
        # sum_i c_i * U[a,i] * V[b,i] * W[c,i] = U[a,j] * V[b,j] * W[c,j]
        
        # Build the linear system
        # Number of equations: n_a * n_b * n_c (but many are redundant)
        # Let's use a subset of equations
        
        # Collect equations from the tensor entries
        equations_A = []
        equations_b = []
        
        # Sample equations from the tensor
        sampled = 0
        max_samples = 100
        
        for a in range(n_a):
            for b in range(n_b):
                for c in range(n_c):
                    if T[a,b,c] != 0 or (U[a,j] * V[b,j] * W[c,j] != 0):
                        # This entry contributes
                        row = np.array([U[a,i] * V[b,i] * W[c,i] for i in other_indices])
                        rhs = U[a,j] * V[b,j] * W[c,j]
                        equations_A.append(row)
                        equations_b.append(rhs)
                        sampled += 1
                        if sampled >= max_samples:
                            break
                if sampled >= max_samples:
                    break
            if sampled >= max_samples:
                break
        
        if len(equations_A) < n_others:
            continue
        
        A_mat = np.array(equations_A[:n_others])
        b_vec = np.array(equations_b[:n_others])
        
        try:
            # Solve the linear system
            if np.linalg.matrix_rank(A_mat) < n_others:
                continue
            
            coeffs = np.linalg.solve(A_mat, b_vec)
            
            # Verify the solution
            full_A = np.array(equations_A)
            full_b = np.array(equations_b)
            residuals = full_A @ coeffs - full_b
            
            if np.allclose(residuals, 0, atol=tol):
                # Great! We can remove term j
                new_U = np.zeros((n_a, n_others))
                new_V = np.zeros((n_b, n_others))
                new_W = np.zeros((n_c, n_others))
                
                for idx, i in enumerate(other_indices):
                    new_U[:, idx] = U[:, i]
                    new_V[:, idx] = V[:, i]
                    new_W[:, idx] = W[:, i]
                
                # Add the contribution of term j distributed among others
                for a in range(n_a):
                    for idx, i in enumerate(other_indices):
                        # The term j contribution at (a, b, c) is U[a,j]*V[b,j]*W[c,j]
                        # We've already accounted for this via the coefficients
                        pass
                
                # Actually, the new decomposition is:
                # sum_{i in others} (U[i] + c_i * U[j]) * V[i] * W[i]  -- NO, this is wrong
                # The correct form: we found c_i such that term_j = sum c_i * term_i
                # So the total tensor = sum_{all} term_k = sum_{others} (1 + c_i) * term_i + (1-c_j)*term_j
                # Wait, that's not right either.
                
                # If term_j = sum_{i in others} c_i * term_i, then:
                # Total = sum_{others} term_i + term_j = sum_{others} (1 + c_i) * term_i
                # But this changes the coefficients, not the rank.
                
                # To reduce rank, we need term_j to be a linear combination with
                # coefficients that allow us to absorb it.
                
                # Actually, if term_j = sum c_i * term_i, we can just remove term_j
                # and adjust... but rank-1 terms don't combine that way.
                
                # Let's verify the new decomposition
                if _verify_decomposition(new_U, new_V, new_W, T, tol=tol):
                    return new_U, new_V, new_W
        except Exception:
            continue
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Start with rank-17
    terms = _get_rank17_terms()
    U, V, W = _terms_to_matrices(terms, n_a, n_b, n_c)
    
    # Try to polish to rank-16
    result = _try_rank16_polish(U, V, W, T, deadline)
    if result is not None:
        U, V, W = result
    
    # Final verification
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
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
