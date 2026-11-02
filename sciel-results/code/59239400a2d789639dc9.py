import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Use Strassen-like decomposition for 2x2 blocks (rank 7 per block)
    - Use schoolbook for the remaining column
    - Total: 5 blocks * 7 + 20 schoolbook = 55
    - Try to reduce to rank 54 by exploiting n=2 linearity
    
    The output W has shape (p*n, rank) = (20, rank)
    Each W[k*n+i, t] gives coefficient for C[i,k] from term t
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Build the known rank-55 solution
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify the solution
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i, j, k] = np.sum(U55[i*m+j, :] * V55[j*p+k, :] * W55[k*n+i, :])
    
    assert np.allclose(T_check, 1.0), f"Rank-55 solution invalid"
    
    # Try to find rank-54 solution by exploiting n=2 structure
    # Idea: In the 2x2 Strassen blocks, the W matrix has entries for both rows.
    # Since n=2, we might be able to share a linear combination.
    # 
    # The 2x2x2 tensor has rank 7 (Strassen). For 2x3x10, we have:
    # - 5 blocks of 2x2x2 (using columns 0-8 of B, rows 0-1 of A)
    # - 20 schoolbook terms for column 2 of A
    #
    # Total: 5*7 + 20 = 55
    #
    # To get rank 54, we need to save 1 rank. One approach:
    # Look at the W matrix structure. For each 2x2 block, W has 4 non-zero entries
    # per term. Maybe we can find a linear dependency.
    
    def try_rank_54():
        """
        Try to find a rank-54 solution by looking for linear dependencies
        in the W matrix that allow merging two terms.
        """
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        # The key insight: for n=2, the output has only 2 rows.
        # In the 2x2x2 Strassen decomposition, each term contributes to both
        # output rows. We can try to find if any two terms can be combined.
        
        # Let's analyze the W matrix structure for one 2x2 block:
        # M1: W[c00] = 1, W[c11] = 1  -> row 0 col 0, row 1 col 1
        # M2: W[c10] = 1, W[c11] = -1 -> row 1 col 0, row 1 col 1
        # M3: W[c01] = 1, W[c11] = 1  -> row 0 col 1, row 1 col 1
        # M4: W[c00] = 1, W[c10] = 1  -> row 0 col 0, row 1 col 0
        # M5: W[c00] = -1, W[c01] = 1 -> row 0 col 0, row 0 col 1
        # M6: W[c11] = 1              -> row 1 col 1
        # M7: W[c00] = 1              -> row 0 col 0
        
        # For the full problem, we have 5 such blocks plus 20 schoolbook terms.
        # The schoolbook terms only affect one output row each.
        
        # One approach: try to find a linear combination of the 35 Strassen terms
        # that has a simple structure allowing us to merge with a schoolbook term.
        
        # Alternative: use optimization to find if rank 54 is possible.
        # We'll try a simpler heuristic: check if removing one term and adjusting
        # others can still give an exact decomposition.
        
        # For now, let's try a direct approach: 
        # In the 2x2x2 case, the 7 terms are:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # For the 2x3x10 case with 5 blocks, we have 35 Strassen terms.
        # Let's see if we can find a linear dependency among them.
        
        # Actually, let's try a different approach: 
        # Use the fact that n=2 to see if we can reduce the rank.
        
        # The tensor T has entries T[i,j,k] = 1 if (i,j,k) corresponds to a valid
        # multiplication term, 0 otherwise.
        
        # For 2x3x10:
        # i in {0,1}, j in {0,1,2}, k in {0,...,9}
        # T[i,j,k] = 1 for all valid (i,j,k)
        
        # The rank is the minimum number of rank-1 tensors needed to sum to T.
        
        # Let's try to find if rank 54 is possible by solving a linear system.
        # We'll fix U and V to have a specific structure and solve for W.
        
        # Strategy: Start with the rank-55 solution. Try to express one term as a
        # linear combination of the others, then remove it.
        
        # For each term t in 0..54, check if the tensor contribution of term t
        # can be written as a linear combination of the other terms' contributions.
        
        # The contribution of term t to the tensor is:
        # T_t[i,j,k] = U[i,j,t] * V[j,k,t] * W[k,i,t]
        
        # We want to check if there exist coefficients c_s (s != t) such that:
        # U[i,j,t] * V[j,k,t] * W[k,i,t] = sum_{s != t} c_s * U[i,j,s] * V[j,k,s] * W[k,i,s]
        # for all (i,j,k).
        
        # This is a linear system in the c_s variables.
        
        # However, this is a 60-variable system (54 unknowns) with 60 equations
        # (2*3*10 = 60). It might be solvable.
        
        # Let's build the system for each term t.
        
        n_terms = 55
        # Build the tensor contributions
        T_contrib = np.zeros((n_terms, n, m, p))
        for t in range(n_terms):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_contrib[t, i, j, k] = U55[i*m+j, t] * V55[j*p+k, t] * W55[k*n+i, t]
        
        # For each term t, check if it can be expressed as a linear combination of others
        best_solution = None
        for t_remove in range(n_terms):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Build the system: find c_s such that T_contrib[t_remove] = sum_{s != t_remove} c_s * T_contrib[s]
            # This is a linear system: A @ c = b
            # where A is (n*m*p) x (n_terms-1) and b is (n*m*p)
            
            indices = [s for s in range(n_terms) if s != t_remove]
            A = np.zeros((n*m*p, len(indices)))
            b = np.zeros(n*m*p)
            
            idx = 0
            for s in indices:
                A[:, idx] = T_contrib[s].flatten()
                idx += 1
            b = T_contrib[t_remove].flatten()
            
            # Solve the linear system
            try:
                c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                residual = np.linalg.norm(A @ c - b)
                if residual < 1e-10:
                    # We found an exact linear combination
                    # Build the new rank-54 solution
                    U54 = np.zeros((n*m, 54))
                    V54 = np.zeros((m*p, 54))
                    W54 = np.zeros((p*n, 54))
                    
                    col = 0
                    for s in indices:
                        U54[:, col] = U55[:, s] * c[col]
                        V54[:, col] = V55[:, s]
                        W54[:, col] = W55[:, s]
                        col += 1
                    
                    # Verify
                    T_check54 = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check54[i, j, k] = np.sum(U54[i*m+j, :] * V54[j*p+k, :] * W54[k*n+i, :])
                    
                    if np.allclose(T_check54, 1.0, atol=1e-10):
                        best_solution = (U54, V54, W54, 54)
                        break
            except np.linalg.LinAlgError:
                continue
        
        return best_solution
    
    # Try to find rank-54 solution
    result_54 = try_rank_54()
    if result_54 is not None:
        best_U, best_V, best_W, best_rank = result_54
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
