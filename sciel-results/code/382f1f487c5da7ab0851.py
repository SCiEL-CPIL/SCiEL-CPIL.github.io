import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    start_time = time.time()
    deadline = start_time + 135

    def _verify(U, V, W, n, m, p):
        """Verify that the tensor decomposition exactly reproduces the multiplication tensor."""
        rank = U.shape[1]
        T = np.ones((n, m, p))
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    result[i, j, k] = val
        return np.allclose(result, T, atol=1e-9)

    def _is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    def _build_decomposition(terms, n, m, p):
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            U[:, r] = terms[r][0]
            V[:, r] = terms[r][1]
            W[:, r] = terms[r][2]
        return U, V, W

    def _try_decomp(terms, n, m, p):
        U, V, W = _build_decomposition(terms, n, m, p)
        if _verify(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Schoolbook decomposition (rank = n*m*p = 30)
    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    terms.append((u, v, w))
        return terms

    def _tensor_to_vector(T, n, m, p):
        """Flatten tensor T[i,j,k] to vector of length n*m*p."""
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = T[i,j,k]
        return vec

    def _rank_reduction_search(n, m, p, max_rank, time_limit):
        """
        Greedy rank reduction: start from schoolbook, try to remove terms one by one.
        For each candidate removal, check if the remaining terms can still span the tensor.
        """
        terms = schoolbook_terms(n, m, p)
        current_rank = len(terms)
        
        # Build the "feature" matrices for each term
        # Term r has u_r (size n*m), v_r (size m*p), w_r (size p*n)
        # The tensor equation is: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
        
        # We'll try to reduce rank by checking if any term is "redundant"
        # A term is redundant if removing it and adjusting others still gives exact solution
        
        best_terms = list(terms)
        
        # Try removing terms one at a time, greedily
        while time.time() < deadline - 5:
            if current_rank <= max_rank:
                break
            
            removed = False
            for idx in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                # Try removing term idx
                candidate_terms = [t for i, t in enumerate(best_terms) if i != idx]
                cand_rank = len(candidate_terms)
                
                # Check if we can solve for the remaining terms
                # This is a linear system: for each (i,j,k), sum over remaining r of
                # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
                # This is nonlinear in general, but if we fix two of the three vectors,
                # it becomes linear in the third.
                
                # Strategy: Fix U and V, solve for W. Or fix U and W, solve for V.
                # Let's fix U and V, solve for W.
                
                U_cand = np.array([t[0] for t in candidate_terms]).T  # (n*m, cand_rank)
                V_cand = np.array([t[1] for t in candidate_terms]).T  # (m*p, cand_rank)
                
                # Build linear system for W
                # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
                # Then sum_r A[i,j,k,r] * W[k*n+i, r] = 1
                # But W has different indices for each (i,k) pair.
                # For fixed (i,k), the equation is: sum_r A[i,j,k,r] * W[k*n+i, r] = 1 for each j
                # This gives m equations for each (i,k) pair, with cand_rank unknowns.
                
                # Actually, let's set up the full linear system.
                # Unknowns: W[k*n+i, r] for k in range(p), i in range(n), r in range(cand_rank)
                # Total unknowns: p * n * cand_rank
                # Equations: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # Total equations: n * m * p
                
                num_unknowns = p * n * cand_rank
                num_eqs = n * m * p
                
                if num_unknowns < num_eqs:
                    # Underdetermined, skip
                    continue
                
                # Build system
                # For equation (i,j,k): unknowns are W[k*n+i, r] for r in range(cand_rank)
                # Coefficient of W[k*n+i, r] in equation (i,j,k) is U[i*m+j,r] * V[j*p+k,r]
                
                # We can write this as: for each (i,k), we have m equations (one per j)
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for j=0,...,m-1
                
                # Let's build the full system
                A_mat = np.zeros((num_eqs, num_unknowns))
                b_vec = np.ones(num_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Equation index: i*m*p + j*p + k
                            for r in range(cand_rank):
                                # Unknown index: (k*n + i) * cand_rank + r
                                unk_idx = (k * n + i) * cand_rank + r
                                A_mat[eq_idx, unk_idx] = U_cand[i*m+j, r] * V_cand[j*p+k, r]
                            eq_idx += 1
                
                # Solve
                try:
                    W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    continue
                
                # Check if solution is exact
                W_sol = W_sol.reshape(p * n, cand_rank)
                
                # Verify
                U_arr = U_cand
                V_arr = V_cand
                W_arr = W_sol
                
                T_check = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(cand_rank):
                                val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                            T_check[i,j,k] = val
                
                if np.allclose(T_check, np.ones((n,m,p)), atol=1e-9):
                    # Check half-integer
                    if _is_half_integer(U_arr) and _is_half_integer(V_arr) and _is_half_integer(W_arr):
                        new_terms = []
                        for r in range(cand_rank):
                            new_terms.append((U_arr[:,r].copy(), V_arr[:,r].copy(), W_arr[:,r].copy()))
                        best_terms = new_terms
                        current_rank = cand_rank
                        removed = True
                        break
            
            if not removed:
                break
        
        return best_terms, current_rank

    # Start with schoolbook
    school_terms = schoolbook_terms(n, m, p)
    school_result = _try_decomp(school_terms, n, m, p)
    
    if school_result is None:
        # Fallback: return schoolbook even if verification fails (shouldn't happen)
        U, V, W = _build_decomposition(school_terms, n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = school_result
    best_rank = len(school_terms)
    
    # Try rank reduction
    if time.time() < deadline - 10:
        reduced_terms, reduced_rank = _rank_reduction_search(n, m, p, best_rank - 1, deadline - 5)
        if reduced_rank < best_rank:
            result = _try_decomp(reduced_terms, n, m, p)
            if result is not None:
                best_result = result
                best_rank = reduced_rank
    
    return best_result

# EVOLVE-BLOCK-END
