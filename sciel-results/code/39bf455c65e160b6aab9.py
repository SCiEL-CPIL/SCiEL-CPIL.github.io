"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Check that the factor matrices exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build tensor T[i, j, k] = delta_{i,k} * delta_{j,j} = 1 for all (i,j,k) in schoolbook
    # Actually the tensor is T[i,j,k] where output C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the standard basis)
    # Wait no. The multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # The decomposition is: for each term c, sum over c: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # T[i,j,k] = 1 for all valid i,j,k (it's the rank-n*m*p all-ones in the appropriate structure)
    # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Let me verify: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(n, m, p, target_rank, max_iters=2000, time_limit=80.0):
    """Try to find a decomposition of given rank using iterative refinement."""
    import time
    start_time = time.time()
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor as a matrix: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a 3-tensor decomposition. We can use a simple iterative method.
    
    best = None
    np.random.seed(42)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random sparse matrices
        density = 0.3
        U = np.random.choice([0, 1, -1], size=(nm, target_rank)) * (np.random.rand(nm, target_rank) < density)
        V = np.random.choice([0, 1, -1], size=(mp, target_rank)) * (np.random.rand(mp, target_rank) < density)
        W = np.random.choice([0, 1, -1], size=(pn, target_rank)) * (np.random.rand(pn, target_rank) < density)
        
        # Ensure at least one non-zero per row
        for row in range(nm):
            if np.all(U[row] == 0):
                U[row, 0] = 1
        for row in range(mp):
            if np.all(V[row] == 0):
                V[row, 0] = 1
        for row in range(pn):
            if np.all(W[row] == 0):
                W[row, 0] = 1
        
        # Iterate: fix two matrices, solve for the third
        for sub_iter in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Fix U, V; solve for W
            # For each (k, i), we need: sum_c W[k*n+i, c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) = 1
            # Let M_ki[c] = sum_j U[i*m+j,c]*V[j*p+k,c]
            # Then we need W[k*n+i,:] dot M_ki = 1 for each (k,i)
            # This is an underdetermined system. We want integer/half-integer solutions.
            # Use least squares as a proxy.
            
            # Build system for W
            # Rows indexed by (k, i), columns by c
            A_W = np.zeros((pn, target_rank))
            for k in range(p):
                for i in range(n):
                    for c in range(target_rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, c] * V[j*p+k, c]
                        A_W[k*n+i, c] = s
            
            # Solve A_W @ W_col = 1 for each column? No, W is a row vector for each (k,i).
            # We want: for each row r (corresponding to (k,i)), sum_c W[r,c] * A_W[r,c] = 1
            # This is: W @ A_W.T = ones(pn)
            # So A_W.T @ W.T = ones(pn), where W.T is (target_rank, pn)
            # Or equivalently, for each c, sum over r: W[r,c] * A_W[r,c] ... no.
            
            # Let me reformulate. W has shape (pn, target_rank).
            # Constraint: for each r in [0, pn): sum_c W[r,c] * A_W[r,c] = 1
            # This means A_W * W^T = ones(pn), where A_W is (pn, target_rank) and W^T is (target_rank, pn)
            # So (A_W * W^T)[r] = sum_c A_W[r,c] * W^T[c, r]... no that's not right either.
            
            # A_W has shape (pn, target_rank). W has shape (pn, target_rank).
            # We want: for each r, sum_c W[r,c] * A_W[r,c] = 1
            # This is: W @ A_W^T = ones(pn) where A_W^T is (target_rank, pn)
            # So (W @ A_W^T)[r] = sum_c W[r,c] * A_W^T[c, r] = sum_c W[r,c] * A_W[r, c] = 1. Yes!
            # So we need W @ A_W^T = ones(pn)
            # This is equivalent to: for each column c of A_W^T (which is row c of A_W), we have...
            # Actually, let's just solve row by row. For each r, we need a vector w_r of length target_rank
            # such that w_r dot A_W[r,:] = 1. We want w_r to be simple (integer/half-integer).
            # Use the minimum-norm solution: w_r = A_W[r,:] / (A_W[r,:] dot A_W[r,:])
            # But we want it to be simple. Let's just use least squares with regularization.
            
            # Actually, let's use a different approach. Fix W and U, solve for V, etc.
            # This is getting complex. Let me use a simpler heuristic.
            
            # Simple approach: just check if current (U,V,W) works, and if not, do a greedy update.
            # For each (i,j,k) where the constraint isn't satisfied, adjust one of the matrices.
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (s - 1.0) ** 2
            
            if error < 1e-12:
                # Check for integer/half-integer entries
                U_int = U * 2
                V_int = V * 2
                W_int = W * 2
                if (np.abs(U_int - np.round(U_int)) < 1e-9 and
                    np.abs(V_int - np.round(V_int)) < 1e-9 and
                    np.abs(W_int - np.round(W_int)) < 1e-9):
                    best = (U.copy(), V.copy(), W.copy())
                    return best
            
            # Greedy: pick the worst (i,j,k) and adjust
            max_err = 0
            worst = (0, 0, 0)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        e = abs(s - 1.0)
                        if e > max_err:
                            max_err = e
                            worst = (i, j, k)
            
            i, j, k = worst
            # Adjust W[k*n+i, :] to better satisfy the constraint
            # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let a[c] = U[i*m+j,c] * V[j*p+k,c]
            # We need: a dot W_row = 1
            a = U[i*m+j, :] * V[j*p+k, :]
            if np.sum(a**2) > 1e-12:
                # Minimum norm solution: w = a / (a dot a)
                w_new = a / np.dot(a, a)
                # Round to nearest half-integer
                w_new = np.round(w_new * 2) / 2
                W[k*n+i, :] = w_new
        
        # Check if we have a valid solution
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error += (s - 1.0) ** 2
        
        if error < 1e-12:
            U_int = U * 2
            V_int = V * 2
            W_int = W * 2
            if (np.abs(U_int - np.round(U_int)) < 1e-9 and
                np.abs(V_int - np.round(V_int)) < 1e-9 and
                np.abs(W_int - np.round(W_int)) < 1e-9):
                best = (U.copy(), V.copy(), W.copy())
                return best
    
    return best

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # For 3x5x7, the schoolbook rank is 105.
    # Try ranks from a reasonable lower bound up to schoolbook-1.
    # Known lower bounds: at least n*m + m*p + p*n - 2 or similar, but let's be pragmatic.
    # Try a few specific ranks.
    
    candidate_ranks = [104, 103, 102, 100, 95, 90]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:  # Leave safety margin
            break
        
        result = _try_random_search(n, m, p, rank, max_iters=500, time_limit=min(60, deadline - time.time() - 10))
        if result is not None:
            U_np, V_np, W_np = result
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U_np]
            V_list = [[float(x) for x in row] for row in V_np]
            W_list = [[float(x) for x in row] for row in W_np]
            
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
