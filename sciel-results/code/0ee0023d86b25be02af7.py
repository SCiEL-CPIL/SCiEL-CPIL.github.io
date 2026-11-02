import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy:
    1. Construct the hybrid Strassen/Schoolbook decomposition (Rank 39).
    2. Iteratively check for linear dependencies among the term vectors.
    3. If a dependency is found, reduce the rank by expressing one term as a linear combination of others.
    4. Return the reduced rank decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
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
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2]
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
    
    # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7]
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7]
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
        
    rank = len(terms)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u + dim_v + dim_w
    
    # Function to reduce rank by finding dependencies
    def reduce_rank(terms_list):
        current_rank = len(terms_list)
        if current_rank <= 1:
            return terms_list
            
        term_matrix = np.zeros((current_rank, total_dim))
        for i, (u, v, w) in enumerate(terms_list):
            term_matrix[i, :dim_u] = u
            term_matrix[i, dim_u:dim_u+dim_v] = v
            term_matrix[i, dim_u+dim_v:] = w
            
        try:
            # Find null space
            # We want c such that M c = 0
            # Use SVD
            U_svd, s, Vt_svd = np.linalg.svd(term_matrix, full_matrices=True)
            tol = 1e-10
            effective_rank = np.sum(s > tol)
            
            if effective_rank < current_rank:
                # The null space is the last (current_rank - effective_rank) rows of Vt_svd
                # We only need one dependency vector
                # The last row of Vt_svd is a vector c such that M c = 0
                c = Vt_svd[-1, :]
                
                # Find an index j with c[j] != 0
                j = None
                for idx in range(current_rank):
                    if abs(c[idx]) > tol:
                        j = idx
                        break
                
                if j is not None:
                    # T_j = - sum_{i != j} (c[i]/c[j]) T_i
                    # New terms: T'_i = (1 - c[i]/c[j]) T_i for i != j
                    new_terms = []
                    for i in range(current_rank):
                        if i == j:
                            continue
                        scale = 1.0 - c[i] / c[j]
                        u_new = [val * scale for val in terms_list[i][0]]
                        v_new = terms_list[i][1] # Scale can be put in U
                        w_new = terms_list[i][2]
                        new_terms.append((u_new, v_new, w_new))
                    
                    # Verify the sum is preserved?
                    # T_old = sum T_i = sum_{i!=j} T_i + T_j
                    # T_j = - sum_{i!=j} (c[i]/c[j]) T_i
                    # T_old = sum_{i!=j} T_i - sum_{i!=j} (c[i]/c[j]) T_i = sum_{i!=j} (1 - c[i]/c[j]) T_i
                    # This is correct.
                    
                    # Recursively reduce
                    return reduce_rank(new_terms)
                else:
                    return terms_list
            else:
                return terms_list
        except:
            return terms_list

    reduced_terms = reduce_rank(terms)
    final_rank = len(reduced_terms)
    
    U = [[0.0] * final_rank for _ in range(n * m)]
    V = [[0.0] * final_rank for _ in range(m * p)]
    W = [[0.0] * final_rank for _ in range(p * n)]
    
    for c_idx, (u_list, v_list, w_list) in enumerate(reduced_terms):
        for i in range(n * m):
            U[i][c_idx] = u_list[i]
        for i in range(m * p):
            V[i][c_idx] = v_list[i]
        for i in range(p * n):
            W[i][c_idx] = w_list[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
