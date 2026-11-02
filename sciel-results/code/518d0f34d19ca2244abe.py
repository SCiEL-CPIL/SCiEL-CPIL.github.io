"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin
    
    # Build the matrix multiplication tensor
    # T[a, b, c] where a = i*m+j (A index), b = j'*p+k (B index), c = k'*n+i' (C index)
    # T[a,b,c] = 1 iff j=j' and k=k' and i=i'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # For matrix multiplication, rank-1 components have special structure.
    # A rank-1 u⊗v⊗w is nonzero at (a,b,c) iff:
    #   u[i*m+j] * v[j'*p+k] * w[k'*n+i'] != 0
    #   and the tensor requires j=j', k=k', i=i'
    # So the support pattern is determined by sets of indices.
    
    # Strategy: Try structured rank-1 extractions where we fix subsets of indices
    # and extract a rank-1 that covers multiple tensor entries at once.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Helper: extract a structured rank-1 component
    def extract_structured_rank1(T_work, i_set, j_set, k_set):
        """
        Extract a rank-1 component that covers all (i,j,k) with i in i_set, j in j_set, k in k_set.
        u[i*m+j] = 1 for i in i_set, j in j_set
        v[j'*p+k] = 1 for j' in j_set, k in k_set  
        w[k'*n+i'] = 1 for k' in k_set, i' in i_set
        This covers |i_set| * |j_set| * |k_set| positions, but only if the sets are consistent.
        Actually, this only works if we're careful about the index matching.
        
        Let me reconsider. The tensor entry T[a,b,c] = 1 iff:
          a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j', k=k', i=i'
        
        A rank-1 u⊗v⊗w contributes to T[a,b,c] if:
          u[i*m+j] * v[j'*p+k] * w[k'*n+i'] is nonzero
        
        For this to match the tensor structure, we need that whenever the product is nonzero,
        the indices satisfy j=j', k=k', i=i'.
        
        One way: choose u supported on {(i,j) : i in I, j in J}, 
                   v supported on {(j,k) : j in J, k in K},
                   w supported on {(k,i) : k in K, i in I}
        Then the product is nonzero when j=j' in J, k=k' in K, i=i' in I,
        which gives exactly the tensor entries for those index sets.
        """
        if not i_set or not j_set or not k_set:
            return None, 0
        
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        
        for i in i_set:
            for j in j_set:
                u_vec[i*m+j] = 1.0
        
        for j in j_set:
            for k in k_set:
                v_vec[j*p+k] = 1.0
        
        for k in k_set:
            for i in i_set:
                w_vec[k*n+i] = 1.0
        
        # Compute the rank-1 tensor
        rank1 = np.outer(np.outer(u_vec, v_vec), w_vec).reshape(n*m, m*p, p*n)
        
        # Check if this is a valid subset of the remaining tensor
        # It should only have 1s where T_work has 1s (or at least where T has 1s)
        # Actually, we need to be careful: the rank-1 might have 1s where T_work has 0
        # which would be bad. Let's check.
        
        # The rank-1 has 1s at positions where i in i_set, j in j_set, k in k_set
        # and the index matching is consistent. This should match the tensor structure.
        
        # Subtract from T_work
        T_work_new = T_work - rank1
        
        # Verify no negative values (which would indicate over-subtraction)
        if np.any(T_work_new < -1e-10):
            return None, 0
        
        return (u_vec, v_vec, w_vec), len(i_set) * len(j_set) * len(k_set)
    
    # Greedy with structured extraction
    T_work = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Precompute all possible index sets
    all_i_sets = [set(range(n))]
    all_j_sets = [set(range(m))]
    all_k_sets = [set(range(p))]
    
    # Generate subsets
    def gen_subsets(n):
        subsets = []
        for mask in range(1, 1 << n):
            s = set()
            for bit in range(n):
                if mask & (1 << bit):
                    s.add(bit)
            subsets.append(s)
        return subsets
    
    i_subsets = gen_subsets(n)
    j_subsets = gen_subsets(m)
    k_subsets = gen_subsets(p)
    
    max_iters = 100
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Find the best structured rank-1 to extract
        best_extract = None
        best_coverage = 0
        
        # Try all combinations of index sets
        for i_set in i_subsets:
            for j_set in j_subsets:
                for k_set in k_set if False else k_subsets:
                    if not i_set or not j_set or not k_set:
                        continue
                    
                    result, coverage = extract_structured_rank1(T_work, i_set, j_set, k_set)
                    if result is not None and coverage > best_coverage:
                        best_coverage = coverage
                        best_extract = (result, i_set, j_set, k_set)
        
        if best_extract is None:
            # No more structured extractions possible, fall back to single-entry
            max_idx = np.unravel_index(np.argmax(np.abs(T_work)), T_work.shape)
            max_val = T_work[max_idx]
            if abs(max_val) < 1e-10:
                break
            
            a, b, c = max_idx
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            u_vec[i*m+j] = 1.0
            v_vec[j*p+k] = 1.0
            w_vec[k*n+i] = 1.0
            
            rank1 = np.outer(np.outer(u_vec, v_vec), w_vec).reshape(n*m, m*p, p*n)
            T_work -= rank1
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
            continue
        
        (u_vec, v_vec, w_vec), i_set, j_set, k_set = best_extract
        rank1 = np.outer(np.outer(u_vec, v_vec), w_vec).reshape(n*m, m*p, p*n)
        T_work -= rank1
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    rank_found = len(U_cols)
    
    if rank_found < best_rank:
        best_rank = rank_found
        best_U = np.array(U_cols).T
        best_V = np.array(V_cols).T
        best_W = np.array(W_cols).T
    
    # Convert to half-integers
    if best_U is not None:
        U_out = [[int(round(x * 2)) / 2 for x in row] for row in best_U]
        V_out = [[int(round(x * 2)) / 2 for x in row] for row in best_V]
        W_out = [[int(round(x * 2)) / 2 for x in row] for row in best_W]
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
