import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = 4, 4, 7
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    def verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        """Check if decomposition is valid"""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_arr.shape[1]):
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Build schoolbook decomposition as baseline (rank 112)
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                u_col = [0] * dim_u
                v_col = [0] * dim_v
                w_col = [0] * dim_w
                u_col[idx_u] = 1
                v_col[idx_v] = 1
                w_col[idx_w] = 1
                U_list.append(u_col)
                V_list.append(v_col)
                W_list.append(w_col)
    
    U_arr = np.array(U_list).T  # shape (dim_u, rank)
    V_arr = np.array(V_list).T  # shape (dim_v, rank)
    W_arr = np.array(W_list).T  # shape (dim_w, rank)
    
    if not verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Now try to find a lower-rank decomposition using rank-1 peeling
    # We'll work with the tensor directly
    
    # Build the full tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Rank-1 peeling algorithm
    def peel_rank1_decomposition(T, max_terms=112, time_limit=100):
        """
        Iteratively find rank-1 terms that match the residual tensor.
        """
        residual = T.copy()
        terms = []
        start = time.time()
        
        for _ in range(max_terms):
            if time.time() - start > time_limit:
                break
            
            # Find the entry with maximum absolute value in the residual
            abs_res = np.abs(residual)
            max_idx = np.unravel_index(np.argmax(abs_res), residual.shape)
            max_val = residual[max_idx]
            
            if abs(max_val) < 1e-10:
                break
            
            # We want to find u, v, w such that:
            # u[a] * v[b] * w[c] approximates residual[a,b,c]
            # where (a,b,c) = max_idx
            # 
            # Strategy: set u[max_idx[0]] = 1, then solve for v and w
            # using the structure of the residual
            
            a_max, b_max, c_max = max_idx
            
            # Initialize u, v, w with zeros
            u = np.zeros(dim_u)
            v = np.zeros(dim_v)
            w = np.zeros(dim_w)
            
            # Set u[a_max] = 1
            u[a_max] = 1.0
            
            # For entries where residual[a, b, c] != 0, we need:
            # u[a] * v[b] * w[c] = residual[a, b, c]
            # Since u[a_max] = 1: v[b_max] * w[c_max] = max_val
            
            # Let's try to find v and w that best fit the residual
            # Fix v[b_max] = 1, then w[c_max] = max_val
            v[b_max] = 1.0
            w[c_max] = max_val
            
            # Now try to extend this to other entries
            # For each (a, b, c) where residual[a,b,c] != 0:
            # We need u[a] * v[b] * w[c] = residual[a,b,c]
            
            # This is a system of equations. Let's use a greedy approach:
            # For each b, if v[b] is not set, try to determine it from entries
            # where u[a] and w[c] are known
            
            # First pass: determine v and w from entries where u is known
            # (i.e., entries with a = a_max)
            for b in range(dim_v):
                for c in range(dim_w):
                    if abs(residual[a_max, b, c]) > 1e-10:
                        if abs(v[b]) > 1e-10:
                            # v[b] is known, determine w[c]
                            if abs(w[c]) < 1e-10:
                                w[c] = residual[a_max, b, c] / v[b]
                        elif abs(w[c]) > 1e-10:
                            # w[c] is known, determine v[b]
                            v[b] = residual[a_max, b, c] / w[c]
            
            # Second pass: determine u from entries where v and w are known
            for a in range(dim_u):
                for b in range(dim_v):
                    for c in range(dim_w):
                        if abs(residual[a, b, c]) > 1e-10:
                            if abs(v[b]) > 1e-10 and abs(w[c]) > 1e-10:
                                if abs(u[a]) < 1e-10:
                                    u[a] = residual[a, b, c] / (v[b] * w[c])
            
            # Third pass: refine v and w using all entries
            for b in range(dim_v):
                for c in range(dim_w):
                    # Find an a where u[a] is known and residual[a,b,c] != 0
                    for a in range(dim_u):
                        if abs(u[a]) > 1e-10 and abs(residual[a, b, c]) > 1e-10:
                            if abs(v[b]) < 1e-10 and abs(w[c]) < 1e-10:
                                # Both unknown, skip
                                continue
                            elif abs(v[b]) < 1e-10:
                                v[b] = residual[a, b, c] / (u[a] * w[c])
                            elif abs(w[c]) < 1e-10:
                                w[c] = residual[a, b, c] / (u[a] * v[b])
                            break
            
            # Check if this term is valid
            # Compute the rank-1 tensor and check if it matches the residual
            term_tensor = np.einsum('a,b,c->abc', u, v, w)
            
            # Check the maximum absolute difference
            max_diff = np.max(np.abs(term_tensor - residual))
            
            # If the term doesn't match well, try a different approach
            # For now, just accept it and subtract
            terms.append((u, v, w))
            residual -= term_tensor
        
        return terms
    
    # Try peeling
    start_peel = time.time()
    peeled_terms = peel_rank1_decomposition(T, max_terms=112, time_limit=80)
    
    # Build the decomposition from peeled terms
    if len(peeled_terms) > 0:
        U_peel = np.array([term[0] for term in peeled_terms]).T
        V_peel = np.array([term[1] for term in peeled_terms]).T
        W_peel = np.array([term[2] for term in peeled_terms]).T
        
        # Verify
        if verify_decomp(U_peel, V_peel, W_peel, n, m, p):
            # This is a valid decomposition with rank len(peeled_terms)
            if len(peeled_terms) < U_arr.shape[1]:
                U_arr = U_peel
                V_arr = V_peel
                W_arr = W_peel
    
    # Convert to list format
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
