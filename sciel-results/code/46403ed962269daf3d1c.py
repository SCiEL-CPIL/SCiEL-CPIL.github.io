"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

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

def _get_rank17_terms(n_a, n_b, n_c):
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

def _try_merge_two_terms(terms, i, j, n_a, n_b, n_c, T, tol=1e-9):
    """Try to merge terms i and j into a single rank-1 term.
    
    We look for coefficients alpha, beta such that:
    alpha * (u_i, v_i, w_i) + beta * (u_j, v_j, w_j) = (u_new, v_new, w_new)
    where (u_new, v_new, w_new) is rank-1 (i.e., u_new = u1*u2, v_new = v1*v2, w_new = w1*w2
    for some vectors u1, u2, v1, v2, w1, w2).
    
    Actually, a simpler approach: try to see if the sum of two rank-1 tensors is itself
    rank-1. This happens when the two tensors are "compatible" in a specific way.
    
    For tensor T = u⊗v⊗w + u'⊗v'⊗w' to be rank-1, we need specific conditions.
    
    Alternative: Use the fact that if we have a rank-r decomposition, we can try to find
    a linear dependence among the terms. Specifically, if sum_{k in S} c_k * T_k = 0 for
    some subset S, then we can reduce the rank.
    
    Let's try a different approach: use numerical optimization to find if we can express
    the tensor as rank 16.
    """
    # For now, return None to indicate no merge found
    return None

def _try_rank16_search(n_a, n_b, n_c, T, deadline, tol=1e-9):
    """Try to find a rank-16 decomposition using a heuristic search."""
    start_time = time.time()
    
    # Strategy: Start from rank-17 and try to merge pairs of terms
    terms = _get_rank17_terms(n_a, n_b, n_c)
    
    # Try all pairs of terms to see if they can be merged
    for i, j in combinations(range(17), 2):
        if time.time() > deadline:
            break
        
        # Try to find if terms i and j can be combined
        # This is a complex problem, so we use a heuristic:
        # Check if the two rank-1 tensors have overlapping support in a way
        # that allows them to be written as a single rank-1 tensor plus a correction
        
        # For simplicity, we'll try a few specific merge patterns
        u_i, v_i, w_i = terms[i]
        u_j, v_j, w_j = terms[j]
        
        # Check if the supports are compatible for a simple merge
        # This is a placeholder for a more sophisticated search
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Try to find rank-16 decomposition
    rank16_result = _try_rank16_search(n_a, n_b, n_c, T, deadline)
    if rank16_result is not None:
        U, V, W = rank16_result
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-17
    terms = _get_rank17_terms(n_a, n_b, n_c)
    U, V, W = _terms_to_matrices(terms, n_a, n_b, n_c)
    
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
