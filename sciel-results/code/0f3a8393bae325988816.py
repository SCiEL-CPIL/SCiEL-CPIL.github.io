"""Program to find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Strategy: Try ranks from a lower bound up to schoolbook (60)
    # For each rank r, try to find sparse support patterns (which entries are non-zero)
    # such that the resulting linear system has a solution
    
    # Lower bound: max of matrix ranks of slices
    # T[:,:,c] for each c gives an (n*m) x (m*p) matrix
    # The rank of the full tensor is at least the max rank of any slice
    lb = 0
    for c in range(p*n):
        slice_c = T[:, :, c]
        r_c = np.linalg.matrix_rank(slice_c)
        lb = max(lb, r_c)
    for a in range(n*m):
        slice_a = T[a, :, :]
        r_a = np.linalg.matrix_rank(slice_a)
        lb = max(lb, r_a)
    for b in range(m*p):
        slice_b = T[:, b, :]
        r_b = np.linalg.matrix_rank(slice_b)
        lb = max(lb, r_b)
    
    # Try to find a decomposition with rank starting from max(lb, 40) up to 60
    # We use a heuristic: randomly assign non-zero positions in U, V, W
    # then solve the linear system for the actual values
    
    def try_find_decomposition(rank, num_trials=500):
        """Try to find a rank-r decomposition by random sparsification."""
        for trial in range(num_trials):
            # Randomly choose which (row, col) pairs are non-zero in each matrix
            # Each column t of U,V,W defines a rank-1 term
            # We need to choose support patterns for each column
            
            # For each column t, choose a small set of non-zero entries
            # Strategy: each column has roughly n*m/r non-zero entries in U, etc.
            
            U_support = []
            V_support = []
            W_support = []
            
            for t in range(rank):
                # Choose random rows for U column t
                num_u_nonzero = max(1, int(np.ceil(n*m / rank)))
                u_rows = np.random.choice(n*m, size=min(num_u_nonzero, n*m), replace=False)
                U_support.append(set(u_rows))
                
                num_v_nonzero = max(1, int(np.ceil(m*p / rank)))
                v_rows = np.random.choice(m*p, size=min(num_v_nonzero, m*p), replace=False)
                V_support.append(set(v_rows))
                
                num_w_nonzero = max(1, int(np.ceil(p*n / rank)))
                w_rows = np.random.choice(p*n, size=min(num_w_nonzero, p*n), replace=False)
                W_support.append(set(w_rows))
            
            # Build the linear system:
            # For each (a,b,c) where T[a,b,c] = 1:
            #   sum_{t: a in U_support[t], b in V_support[t], c in W_support[t]} U[a,t]*V[b,t]*W[c,t] = 1
            # For each (a,b,c) where T[a,b,c] = 0:
            #   sum_{t: a in U_support[t], b in V_support[t], c in W_support[t]} U[a,t]*V[b,t]*W[c,t] = 0
            #
            # This is nonlinear (products of unknowns). Instead, we can fix one matrix
            # and solve for the others, or use a different approach.
            #
            # Alternative: Fix U and V to have entries 0 or 1, then solve for W.
            # Or: Use the fact that we can set U[a,t] = 1 if a in support, else 0
            # and similarly for V, then solve for W.
            #
            # Let's try: Set U[a,t] = 1 if a in U_support[t], else 0
            # Set V[b,t] = 1 if b in V_support[t], else 0
            # Then we need: T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # This is linear in W!
            
            # Build the system for W:
            # For each c, the equation is:
            # T[a,b,c] = sum_t (U[a,t]*V[b,t]) * W[c,t] for all (a,b)
            # But this needs to hold for ALL (a,b,c), not just where T=1
            
            # Actually, we need: for all (a,b,c):
            # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            
            # Let's build a linear system in W.
            # Unknowns: W[c,t] for c in [0, p*n), t in [0, rank)
            # Total unknowns: p*n * rank
            
            # Equations: for each (a,b,c):
            # sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            
            # But this is p*n * n*m * m*p equations, which is too many.
            # We only need equations where T[a,b,c] is non-zero, plus some random others.
            
            # Let's collect all (a,b,c) triples
            equations = []
            rhs = []
            
            # Include all non-zero entries
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if T[a, b, c] != 0:
                            # Equation: sum_t U[a,t]*V[b,t]*W[c,t] = 1
                            row = np.zeros(p*n * rank)
                            for t in range(rank):
                                if U[a, t] != 0 and V[b, t] != 0:
                                    row[c * rank + t] = U[a, t] * V[b, t]
                            equations.append(row)
                            rhs.append(T[a, b, c])
            
            # Add some random zero equations to constrain the solution
            num_zero_eqs = min(50, p*n * n*m * m*p - len(equations))
            for _ in range(num_zero_eqs):
                a = np.random.randint(n*m)
                b = np.random.randint(m*p)
                c = np.random.randint(p*n)
                if T[a, b, c] == 0:
                    row = np.zeros(p*n * rank)
                    for t in range(rank):
                        if U[a, t] != 0 and V[b, t] != 0:
                            row[c * rank + t] = U[a, t] * V[b, t]
                    equations.append(row)
                    rhs.append(0)
            
            if len(equations) == 0:
                continue
            
            A_mat = np.array(equations)
            b_vec = np.array(rhs)
            
            # Solve the underdetermined system
            try:
                # Use least squares to find a solution
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution actually satisfies the full tensor equation
            W_mat = W_flat.reshape(p*n, rank)
            
            # Verify: compute the reconstructed tensor
            error = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        recon = 0.0
                        for t in range(rank):
                            recon += U[a, t] * V[b, t] * W_mat[c, t]
                        if abs(recon - T[a, b, c]) > 1e-6:
                            error += abs(recon - T[a, b, c])
            
            if error < 1e-4:
                return U, V, W_mat
            
            # If not exact, try with different random supports
            continue
        
        return None
    
    # First, try to find a good rank
    # Start from a reasonable rank and go up
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from 45 to 59
    for rank in range(45, 60):
        result = try_find_decomposition(rank, num_trials=100)
        if result is not None:
            U, V, W = result
            best_rank = rank
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook decomposition
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

