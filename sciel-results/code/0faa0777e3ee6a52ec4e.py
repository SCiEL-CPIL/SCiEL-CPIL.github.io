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
    """Return the 17 terms as (u_dict, v_dict, w_dict) tuples."""
    terms = [
        ({0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1}),           # 0: Strassen M1
        ({3:1, 4:1}, {0:1}, {1:1, 3:-1}),              # 1: Strassen M2
        ({0:1}, {1:1, 4:-1}, {2:1, 3:1}),              # 2: Strassen M3
        ({4:1}, {3:1, 0:-1}, {0:1, 1:1}),              # 3: Strassen M4
        ({0:1, 1:1}, {4:1}, {0:-1, 2:1}),              # 4: Strassen M5
        ({3:1, 0:-1}, {0:1, 1:1}, {3:1}),              # 5: Strassen M6
        ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),              # 6: Strassen M7
        ({0:1}, {2:1}, {4:1}),                         # 7: a0*b2
        ({1:1}, {5:1}, {4:1}),                         # 8: a1*b5
        ({3:1}, {2:1}, {5:1}),                         # 9: a3*b2
        ({4:1}, {5:1}, {5:1}),                         # 10: a4*b5
        ({2:1}, {6:1}, {0:1}),                         # 11: a2*b6
        ({2:1}, {7:1}, {2:1}),                         # 12: a2*b7
        ({2:1}, {8:1}, {4:1}),                         # 13: a2*b8
        ({5:1}, {6:1}, {1:1}),                         # 14: a5*b6
        ({5:1}, {7:1}, {3:1}),                         # 15: a5*b7
        ({5:1}, {8:1}, {5:1}),                         # 16: a5*b8
    ]
    return terms

def _terms_to_matrices(terms, n_a=6, n_b=9, n_c=6):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    for r, (u_d, v_d, w_d) in enumerate(terms):
        for a, val in u_d.items():
            U[a, r] = val
        for b, val in v_d.items():
            V[b, r] = val
        for c, val in w_d.items():
            W[c, r] = val
    return U, V, W

def _try_merge_two_terms(terms, i, j):
    """
    Try to merge terms i and j into a single term.
    Returns (merged_term, other_terms) if successful, else None.
    
    Two rank-1 terms u1⊗v1⊗w1 and u2⊗v2⊗w2 can be merged if
    there exist scalars α, β such that:
      u1 + α*u2 = λ * u_merged
      v1 + α*v2 = μ * v_merged  
      w1 + β*w2 = ν * w_merged
    where the merged term is u_merged ⊗ v_merged ⊗ w_merged scaled appropriately.
    
    Actually, the correct condition: we want to find u', v', w' such that
      u1⊗v1⊗w1 + u2⊗v2⊗w2 = u'⊗v'⊗w'
    
    This is possible iff the 2x2x2 tensor has rank 1.
    A 2x2x2 tensor T has rank 1 iff all 2x2 minors of its matricizations are zero,
    equivalently, the tensor is "decomposable."
    
    For two terms with support patterns, we can check:
    - Collect all unique indices in each dimension
    - Build the 2x2x2 sub-tensor
    - Check if it has rank 1
    """
    u1, v1, w1 = terms[i]
    u2, v2, w2 = terms[j]
    
    # Get unique indices
    u_idx = sorted(set(u1.keys()) | set(u2.keys()))
    v_idx = sorted(set(v1.keys()) | set(v2.keys()))
    w_idx = sorted(set(w1.keys()) | set(w2.keys()))
    
    if len(u_idx) > 2 or len(v_idx) > 2 or len(w_idx) > 2:
        return None  # Can't merge if any dimension has >2 distinct indices
    
    # Build 2x2x2 tensor
    T = np.zeros((len(u_idx), len(v_idx), len(w_idx)))
    u_map = {idx: k for k, idx in enumerate(u_idx)}
    v_map = {idx: k for k, idx in enumerate(v_idx)}
    w_map = {idx: k for k, idx in enumerate(w_idx)}
    
    for a, val in u1.items():
        for b, val2 in v1.items():
            for c, val3 in w1.items():
                T[u_map[a], v_map[b], w_map[c]] += val * val2 * val3
    
    for a, val in u2.items():
        for b, val2 in v2.items():
            for c, val3 in w2.items():
                T[u_map[a], v_map[b], w_map[c]] += val * val2 * val3
    
    # Check if T has rank 1
    # A 2x2x2 tensor has rank 1 iff all its 2x2 slices have rank 1
    # Equivalently, the flattening along any mode has rank 1
    # Check: T.reshape(2, 4) should have rank 1
    T_flat = T.reshape(2, -1)
    if T_flat.shape[0] >= 2:
        # Compute rank
        if np.linalg.matrix_rank(T_flat, tol=1e-10) > 1:
            return None
    
    # Also check other flattenings
    T_flat2 = T.reshape(-1, 2)
    if T_flat2.shape[0] >= 2:
        if np.linalg.matrix_rank(T_flat2, tol=1e-10) > 1:
            return None
    
    T_flat3 = T.reshape(2, 2, -1).reshape(2, -1)
    if np.linalg.matrix_rank(T_flat3, tol=1e-10) > 1:
        return None
    
    # If rank 1, extract the merged term
    # T = u' ⊗ v' ⊗ w'
    # Find non-zero entries
    nonzero = np.argwhere(T != 0)
    if len(nonzero) == 0:
        return None
    
    # For rank-1 tensor, T[a,b,c] = u'[a] * v'[b] * w'[c]
    # We can extract by:
    # u'[a] = T[a, b0, c0] / (v'[b0] * w'[c0]) for some fixed b0, c0 where v'[b0]*w'[c0] != 0
    # Simpler: just find a canonical form
    
    # Pick first non-zero entry
    a0, b0, c0 = nonzero[0]
    val0 = T[a0, b0, c0]
    
    # Extract u'
    u_prime = np.zeros(len(u_idx))
    u_prime[a0] = 1
    for a in range(len(u_idx)):
        if T[a, b0, c0] != 0:
            u_prime[a] = T[a, b0, c0] / val0
    
    # Extract v'
    v_prime = np.zeros(len(v_idx))
    v_prime[b0] = 1
    for b in range(len(v_idx)):
        if T[a0, b, c0] != 0:
            v_prime[b] = T[a0, b, c0] / val0
    
    # Extract w'
    w_prime = np.zeros(len(w_idx))
    w_prime[c0] = 1
    for c in range(len(w_idx)):
        if T[a0, b0, c] != 0:
            w_prime[c] = T[a0, b0, c] / val0
    
    # Scale: we need T = u' ⊗ v' ⊗ w' * scale
    # Check: T[a0,b0,c0] should equal u'[a0]*v'[b0]*w'[c0]*scale
    # u'[a0]=1, v'[b0]=1, w'[c0]=1, so scale = val0
    scale = val0
    
    # Build merged term dict
    merged_u = {}
    merged_v = {}
    merged_w = {}
    
    for k, idx in enumerate(u_idx):
        if abs(u_prime[k]) > 1e-10:
            merged_u[idx] = u_prime[k]
    for k, idx in enumerate(v_idx):
        if abs(v_prime[k]) > 1e-10:
            merged_v[idx] = v_prime[k]
    for k, idx in enumerate(w_idx):
        if abs(w_prime[k]) > 1e-10:
            merged_w[idx] = w_prime[k]
    
    # Apply scale
    for key in merged_u:
        merged_u[key] *= scale
    for key in merged_v:
        merged_v[key] *= scale
    for key in merged_w:
        merged_w[key] *= scale
    
    # Round to nearest integer or half-integer
    for key in merged_u:
        merged_u[key] = round(merged_u[key] * 2) / 2
    for key in merged_v:
        merged_v[key] = round(merged_v[key] * 2) / 2
    for key in merged_w:
        merged_w[key] = round(merged_w[key] * 2) / 2
    
    merged_term = (merged_u, merged_v, merged_w)
    
    # Remove terms i and j, add merged term
    other_terms = [terms[k] for k in range(len(terms)) if k != i and k != j]
    other_terms.append(merged_term)
    
    return (merged_term, other_terms)

def _verify_terms(terms, n=2, m=3, p=3, tol=1e-9):
    """Verify that the list of terms correctly decomposes the tensor."""
    n_a = n * m
    n_b = m * p
    n_c = p * n
    T = _build_tensor(n, m, p)
    U, V, W = _terms_to_matrices(terms, n_a, n_b, n_c)
    return _verify_decomposition(U, V, W, T, tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Start with rank-17 terms
    terms = _get_rank17_terms()
    
    # Verify initial decomposition
    if not _verify_terms(terms, n, m, p):
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
    
    best_terms = terms
    best_rank = len(terms)
    
    # Try to merge pairs of terms to reduce rank
    # Search strategy: try all pairs, verify result
    while time.time() < deadline - 5:
        found_merge = False
        current_terms = best_terms[:]
        current_rank = len(current_terms)
        
        for i in range(current_rank):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, current_rank):
                result = _try_merge_two_terms(current_terms, i, j)
                if result is not None:
                    merged_term, new_terms = result
                    # Verify the new decomposition
                    if _verify_terms(new_terms, n, m, p):
                        best_terms = new_terms
                        best_rank = len(new_terms)
                        found_merge = True
                        break
            if found_merge or time.time() > deadline - 5:
                break
        
        if not found_merge:
            break
    
    # Convert best terms to matrices
    U, V, W = _terms_to_matrices(best_terms, n_a, n_b, n_c)
    
    # Final verification
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to original rank-17
        U, V, W = _terms_to_matrices(_get_rank17_terms(), n_a, n_b, n_c)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
