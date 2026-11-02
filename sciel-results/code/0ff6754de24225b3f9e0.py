import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    1. Construct the hybrid Strassen/Schoolbook decomposition (Rank 39).
    2. Check for linear dependencies among the term vectors to see if the rank can be reduced.
    3. If dependencies are found, remove dependent terms and return the reduced rank.
    4. If not, return the rank 39 decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    _start_time = time.time()
    _deadline = _start_time + 115.0  # Safety margin
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0] * (n * m)
        v_list = [0] * (m * p)
        w_list = [0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1, w_idx(3,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1, w_idx(3,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1, v_idx(1,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1, v_idx(0,2): -1}, {w_idx(2,0): 1, w_idx(2,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(2,0): -1, w_idx(3,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,2): 1, v_idx(0,3): 1}, {w_idx(3,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,2): 1, v_idx(1,3): 1}, {w_idx(2,0): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 4:6]
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1, w_idx(5,1): 1})
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,4): 1}, {w_idx(4,1): 1, w_idx(5,1): -1})
    # M3
    add_term({u_idx(0,0): 1}, {v_idx(0,5): 1, v_idx(1,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    # M4
    add_term({u_idx(1,1): 1}, {v_idx(1,4): 1, v_idx(0,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,5): 1}, {w_idx(4,0): -1, w_idx(5,0): 1})
    # M6
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,4): 1, v_idx(0,5): 1}, {w_idx(5,1): 1})
    # M7
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,4): 1, v_idx(1,5): 1}, {w_idx(4,0): 1})
    
    # Part 4: Schoolbook for A[0:2, 0:2] * B[0:2, 6]
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(6,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,6): 1}, {w_idx(6,1): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
    original_rank = len(terms)
    
    # Check for linear dependencies by examining if any term's W vector
    # can be expressed as a linear combination of others with matching U,V supports
    # 
    # Key insight: Two terms (u1,v1,w1) and (u2,v2,w2) are linearly dependent
    # if w1 = c*w2 for some scalar c, AND u1 = c*u2, AND v1 = c*v2
    # More generally, we can check if removing one term still allows exact decomposition
    # by solving for the W coefficients
    
    # Build matrices U, V, W as numpy arrays
    U_mat = np.zeros((n * m, original_rank), dtype=float)
    V_mat = np.zeros((m * p, original_rank), dtype=float)
    W_mat = np.zeros((p * n, original_rank), dtype=float)
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U_mat[i, c] = u_list[i]
        for i in range(m * p):
            V_mat[i, c] = v_list[i]
        for i in range(p * n):
            W_mat[i, c] = w_list[i]
    
    # The tensor equation is: sum_c U[:,c] * V[:,c] * W[:,c] = T (the multiplication tensor)
    # This means for each (i,j,k): sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    # 
    # To check if term c0 can be removed, we need to see if there exist coefficients
    # such that the remaining terms still satisfy all tensor equations.
    # 
    # A simpler approach: check if any column of W is a linear combination of others
    # where the corresponding U and V columns also align appropriately.
    
    # Try greedy removal: for each term, check if it's redundant
    removed = set()
    for trial in range(original_rank):
        if time.time() > _deadline:
            break
        if trial in removed:
            continue
            
        # Try removing term 'trial' and see if the system is still consistent
        remaining = [i for i in range(original_rank) if i != trial and i not in removed]
        
        if len(remaining) == 0:
            continue
        
        # For the reduced set, check if we can find W coefficients that work
        # The constraint is: for all (i,j,k), sum_{c in remaining} U[i,c]*V[j,c]*W_new[k,c] = T[i,j,k]
        # 
        # This is a bilinear system in W_new. We can linearize by fixing U and V.
        # For each output index k, we have a linear system in W_new[:,k] (if we treat it differently)
        # Actually, let's think of it as: for each (i,j), the vector [W[k,c] for c in remaining] 
        # must satisfy: sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k] for all k
        # 
        # This is hard to check directly. Instead, use a different approach:
        # Check if the term is linearly dependent in the tensor sense.
        
        # Alternative: Use the fact that if we have terms (u_c, v_c, w_c), the decomposition
        # is valid iff the rank of the tensor formed by these terms equals the target.
        # 
        # Practical check: verify that removing term 'trial' doesn't break any (i,j,k) equation
        # by checking if the contribution of term 'trial' can be absorbed by others.
        
        # For a term to be removable, its "signature" must be spanned by others.
        # The signature of term c is the triplet (U[:,c], V[:,c], W[:,c]).
        # Two terms are compatible if they have proportional U and V vectors.
        
        # Check: is there a scalar alpha such that U[:,trial] = alpha * U[:,c] and V[:,trial] = alpha * V[:,c] for some c in remaining?
        # If so, then W[:,trial] must also = alpha * W[:,c] for the decomposition to be consistent,
        # meaning the term is redundant.
        
        is_redundant = False
        for c in remaining:
            # Check if U[:,trial] is proportional to U[:,c]
            u_t = U_mat[:, trial]
            u_c = U_mat[:, c]
            v_t = V_mat[:, trial]
            v_c = V_mat[:, c]
            w_t = W_mat[:, trial]
            w_c = W_mat[:, c]
            
            # Find alpha from U
            alpha_candidates = []
            for idx in range(n * m):
                if abs(u_c[idx]) > 1e-10:
                    alpha_candidates.append(u_t[idx] / u_c[idx])
            
            if not alpha_candidates:
                continue
            
            # Check consistency
            alpha = alpha_candidates[0]
            consistent_u = all(abs(u_t[idx] - alpha * u_c[idx]) < 1e-8 for idx in range(n * m))
            if not consistent_u:
                continue
            
            consistent_v = all(abs(v_t[idx] - alpha * v_c[idx]) < 1e-8 for idx in range(m * p))
            if not consistent_v:
                continue
            
            consistent_w = all(abs(w_t[idx] - alpha * w_c[idx]) < 1e-8 for idx in range(p * n))
            if consistent_w:
                is_redundant = True
                break
        
        if is_redundant:
            removed.add(trial)
    
    # Build the final matrices with removed terms excluded
    final_indices = [i for i in range(original_rank) if i not in removed]
    final_rank = len(final_indices)
    
    U = [[0.0] * final_rank for _ in range(n * m)]
    V = [[0.0] * final_rank for _ in range(m * p)]
    W = [[0.0] * final_rank for _ in range(p * n)]
    
    for new_c, old_c in enumerate(final_indices):
        u_list, v_list, w_list = terms[old_c]
        for i in range(n * m):
            U[i][new_c] = float(u_list[i])
        for i in range(m * p):
            V[i][new_c] = float(v_list[i])
        for i in range(p * n):
            W[i][new_c] = float(w_list[i])
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
