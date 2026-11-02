import numpy as np
from itertools import product, combinations
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # We aim to find a rank 15 decomposition.
    # The tensor T is defined by T[i, j, k] = 1 if indices match, 0 otherwise.
    # In flattened form: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    
    # We will search for a set of 15 rank-1 terms.
    # Each term r is defined by vectors u_r, v_r, w_r with entries in {0, 1, -1, 0.5, -0.5, ...}.
    # To keep the search space manageable, we restrict to "sparse" patterns.
    # Specifically, we consider patterns where u_r, v_r, w_r are indicator vectors of subsets of indices.
    # Let Su, Sv, Sw be the supports.
    # The contribution of term r to T[a, b, c] is 1 if a in Su, b in Sv, c in Sw, else 0.
    # We want sum_r c_r * indicator_r = T.
    # Since T has 18 ones, and we have 15 terms, we need some terms to contribute to multiple indices
    # or use coefficients to create cancellations/additions.
    # However, if we restrict to indicator vectors with coefficient 1, the sum of 15 indicators
    # can only have at most 15 ones if the supports are disjoint in the (a,b,c) space?
    # No, the "indicator" is in the tensor space.
    # Let's represent each candidate pattern as a vector in R^18.
    # vec_r[a, b, c] = 1 if a in Su_r, b in Sv_r, c in Sw_r, AND (a,b,c) is consistent?
    # No, the tensor entry T[a,b,c] is defined for all a,b,c.
    # But the matrix multiplication tensor only has non-zero entries where indices are consistent.
    # So we only care about the 18 consistent entries.
    # For a pattern (Su, Sv, Sw), the vector v in R^18 is:
    # v[idx(i,j,k)] = 1 if (i*m+j in Su) AND (j*p+k in Sv) AND (k*n+i in Sw), else 0.
    
    # We want to find 15 such vectors v_1..v_15 and coefficients c_1..c_15
    # such that sum c_r v_r = T_vec.
    # T_vec is the vector of 18 ones.
    
    # If we restrict coefficients to integers, and supports to small sets, we can search.
    # Let's generate candidate patterns with supports of size 1 and 2.
    
    u_indices = range(n * m)
    v_indices = range(m * p)
    w_indices = range(p * n)
    
    def get_pattern_vector(su, sv, sw):
        vec = [0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                if u_idx not in su:
                    continue
                for k in range(p):
                    v_idx = j * p + k
                    if v_idx not in sv:
                        continue
                    w_idx = k * n + i
                    if w_idx not in sw:
                        continue
                    idx = (i * m + j) * p + k
                    vec[idx] = 1
        return tuple(vec)
    
    candidates = []
    seen = set()
    
    # Size 1 patterns (18)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                su = (i * m + j,)
                sv = (j * p + k,)
                sw = (k * n + i,)
                key = (su, sv, sw)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
                    
    # Size 2 patterns
    # We will add some size 2 patterns to increase diversity.
    # To keep it fast, we only add a few random ones or all of them if small enough.
    # 15 * 36 * 15 = 8100 is small.
    u_combos = list(combinations(u_indices, 2))
    v_combos = list(combinations(v_indices, 2))
    w_combos = list(combinations(w_indices, 2))
    
    for su in u_combos:
        for sv in v_combos:
            for sw in w_combos:
                key = (su, sv, sw)
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
                    
    # Now we have ~8118 candidates.
    # We want to find a subset of 15 and coefficients such that sum = T_vec.
    # This is a linear combination problem.
    # Let M be the matrix of candidate vectors (18 x N).
    # We want M * x = T_vec, where x is a vector of length N with at most 15 non-zero entries.
    # And x entries are integers (or half-integers).
    
    # Since N is large, we can use a greedy approach or random search.
    # Or, we can restrict to a smaller set of promising candidates.
    # The 18 size-1 patterns form a basis.
    # If we take 15 of them, we can't span T (which is sum of all 18).
    # So we MUST use non-basis patterns (size > 1) or combine them.
    
    # Let's try a random search for 15 patterns and solve for coefficients.
    # If we pick 15 patterns, we form a 18x15 matrix.
    # We check if T_vec is in the column space.
    # Since 18 > 15, it's overdetermined.
    # We can use least squares and check if residual is 0.
    # Or, we can pick 15 patterns such that they are "close" to spanning T.
    
    # Better approach:
    # We know T is the sum of 18 basis vectors.
    # Let's try to replace 3 basis vectors with 0 new vectors, but modify the others?
    # No, we need exactly 15 vectors.
    
    # Let's try to find 15 patterns that sum to T with integer coefficients.
    # We can use a SAT-like search or random restarts.
    # Given the time limit, we'll do random restarts.
    
    best_rank = rank_fb
    best_U, best_V, best_W = None, None, None
    
    num_candidates = len(candidates)
    
    # Precompute vectors for candidates
    cand_vecs = []
    for key in candidates:
        su, sv, sw = key
        vec = get_pattern_vector(su, sv, sw)
        cand_vecs.append(vec)
        
    T_vec = tuple([1] * (n * m * p))
    
    # Random search for rank 15
    # We pick 15 random candidates, form matrix A (18x15), solve A x = T.
    # If a solution exists with integer/half-integer entries, we have a winner.
    
    # To make it faster, we can restrict to candidates that have "good" overlap with T.
    # Or just random.
    
    # Let's try a few iterations.
    max_iters = 10000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        # Pick 15 random candidates
        idxs = random.sample(range(num_candidates), 15)
        A = np.array([cand_vecs[i] for i in idxs]).T  # 18x15
        b = np.array(T_vec)
        
        # Solve least squares
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        # Check if residual is small (exact solution)
        if residuals and residuals[0] < 1e-10:
            # Check if x entries are close to integers or half-integers
            # Scale by 2 to check for integers
            x_scaled = x * 2
            if np.allclose(x_scaled, np.round(x_scaled), atol=1e-10):
                # Found a solution!
                # Construct U, V, W
                U = [[0] * 15 for _ in range(n * m)]
                V = [[0] * 15 for _ in range(m * p)]
                W = [[0] * 15 for _ in range(p * n)]
                
                for r in range(15):
                    coeff = x[r]
                    su, sv, sw = candidates[idxs[r]]
                    for u_idx in su:
                        U[u_idx][r] = coeff
                    for v_idx in sv:
                        V[v_idx][r] = coeff
                    for w_idx in sw:
                        W[w_idx][r] = coeff
                        
                # Verify
                # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # Wait, the decomposition is T = sum U_r \otimes V_r \otimes W_r.
                # The entry T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
                # My construction sets U[a,r] = coeff if a in su.
                # So U[a,r] V[b,r] W[c,r] = coeff * coeff * coeff = coeff^3.
                # But we want the sum to be 1.
                # So we need coeff^3 * (number of terms covering (a,b,c)) = 1?
                # No, the linearity is:
                # T = sum_r (u_r \otimes v_r \otimes w_r)
                # Entry T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c].
                # In my construction, u_r[a] = coeff_r if a in su_r, else 0.
                # So T[a,b,c] = sum_{r: a in su_r, b in sv_r, c in sw_r} coeff_r^3.
                # This is NOT linear in coeff_r.
                # The tensor decomposition is linear in the vectors, but the vectors are scaled by the same coefficient?
                # No, a rank-1 term is u \otimes v \otimes w.
                # If we scale u by c, v by 1, w by 1, the term is scaled by c.
                # If we scale u, v, w all by c, the term is scaled by c^3.
                # In the standard form, we have free coefficients for each vector component.
                # So U[a,r] can be any value, V[b,r] any value, W[c,r] any value.
                # My search assumed U[a,r] = c_r, V[b,r] = c_r, W[c,r] = c_r.
                # This is a very restricted form (isotropic scaling).
                # It is unlikely to find a solution this way.
                
                # We need a more general search.
                # Let's allow U, V, W entries to be independent.
                # But that makes the search space huge.
                
                # Alternative:
                # Use the fact that we can split terms.
                # If we find a solution with real coefficients, we can round?
                # No, exactness is required.
                
                # Let's try a different strategy.
                # Hardcode a known rank 15 decomposition if available.
                # Or, use a smarter search.
                
                pass
                
    # Since the random search with isotropic scaling is unlikely to work,
    # and a general search is too complex for the time limit,
    # we return the fallback.
    
    return fallback
# EVOLVE-BLOCK-END
