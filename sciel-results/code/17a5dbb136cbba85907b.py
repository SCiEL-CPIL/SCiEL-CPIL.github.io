import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = Fraction(1, 1)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        # U, V, W are float arrays
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        T_float = T.astype(float)
        return np.allclose(T_rec, T_float, atol=tol)

    def is_half_integer(arr, tol=1e-9):
        doubled = 2 * arr
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)

    # Rank-1 peeling with exact arithmetic
    # At each step, find the best rank-1 approximation of the remaining tensor,
    # subtract it, and repeat.
    
    def peel_rank1(T_rem):
        """Find best rank-1 approximation of T_rem using SVD on matricizations."""
        # T_rem shape: (n*m, m*p, p*n) = (4, 14, 14)
        # Try all 3 matricizations and pick the one with best rank-1 fit
        # For a 3-tensor, the best rank-1 is given by the top singular vectors
        # of any matricization (for CP, this is an approximation)
        
        # Matricization 1: reshape to (4, 14*14)
        M1 = T_rem.reshape(4, 14*14)
        # Matricization 2: reshape to (14, 4*14)
        M2 = T_rem.transpose(1, 0, 2).reshape(14, 4*14)
        # Matricization 3: reshape to (14, 4*14)
        M3 = T_rem.transpose(2, 0, 1).reshape(14, 4*14)
        
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        # Try matricization 1
        try:
            U_svd, S, Vt = np.linalg.svd(M1, full_matrices=False)
            u_vec = U_svd[:, 0]
            v_vec = Vt[0, :].reshape(14, 14)
            # Check: T_approx = u_vec * v_vec (outer product in mat-1 space)
            # But we need T[a,b,c] = u[a] * v[b,c]
            # This is not a valid CP term unless v is rank-1
            # So this approach doesn't directly give CP rank-1 terms
        except:
            pass
        
        # Alternative: use the fact that for small tensors, we can search
        # for rank-1 terms by trying all possible sparse patterns
        # Since T is sparse (only 28 non-zero entries out of 4*14*14=784),
        # a rank-1 term that matches many non-zeros is likely useful.
        
        # For each non-zero entry, try to find a rank-1 term that covers it
        # and as many other non-zeros as possible.
        
        nonzero_entries = []
        for a in range(4):
            for b in range(14):
                for c in range(14):
                    if T_rem[a, b, c] != 0:
                        nonzero_entries.append((a, b, c))
        
        if not nonzero_entries:
            return None, None, None, 0
        
        best_cov = 0
        best_u = None
        best_v = None
        best_w = None
        best_val = None
        
        # For each non-zero entry, try to construct a rank-1 term
        # A rank-1 term is defined by u (4-dim), v (14-dim), w (14-dim)
        # T[a,b,c] = u[a]*v[b]*w[c]
        # For the term to match entry (a0,b0,c0) with value 1:
        # u[a0]*v[b0]*w[c0] = 1
        # We can set u[a0]=1, v[b0]=1, w[c0]=1 and see what other entries it matches.
        
        for (a0, b0, c0) in nonzero_entries:
            # Try u with only a0 non-zero, v with only b0 non-zero, w with only c0 non-zero
            # This is the schoolbook term - covers exactly 1 entry
            # To cover more, we need u, v, w to have multiple non-zeros.
            
            # Try: u has non-zeros at positions that are "compatible" with a0
            # For entry (a,b,c) to be covered: u[a]*v[b]*w[c] must equal T_rem[a,b,c]
            # If T_rem[a,b,c] = 1, then u[a]*v[b]*w[c] = 1
            # If T_rem[a,b,c] = -1, then u[a]*v[b]*w[c] = -1
            
            # Heuristic: try small integer vectors
            # u in {-1, 0, 1}^4, v in {-1, 0, 1}^14, w in {-1, 0, 1}^14
            # But that's 3^4 * 3^14 * 3^14 = way too many
            
            # Better: for a given (a0, b0, c0), try u with support S_a, v with support S_b, w with support S_c
            # such that for all (a,b,c) in S_a x S_b x S_c, T_rem[a,b,c] = u[a]*v[b]*w[c]
            
            # Start with minimal support: u[a0]=1, v[b0]=1, w[c0]=1
            # Then try to expand support by adding entries that are consistent.
            
            u = np.zeros(4)
            v = np.zeros(14)
            w = np.zeros(14)
            u[a0] = 1
            v[b0] = 1
            w[c0] = 1
            
            # Try to expand: for each other non-zero (a,b,c), check if we can
            # add it by setting u[a], v[b], w[c] appropriately.
            # This is complex. Let's just use the schoolbook terms as fallback
            # and try a few structured patterns.
            
            pass
        
        # Fallback: return a schoolbook term for the first non-zero entry
        a0, b0, c0 = nonzero_entries[0]
        u = np.zeros(4)
        v = np.zeros(14)
        w = np.zeros(14)
        u[a0] = 1
        v[b0] = 1
        w[c0] = 1
        return u, v, w, 1
    
    # Given the complexity of peeling, let's try a different structural approach:
    # Use the known algebraic structure. The 2x2x7 tensor can be decomposed
    # by treating the 7 columns as 7 instances of 2x2x1, but sharing the A-side.
    # 
    # Key insight: C[:,k] = A @ B[:,k] for k=0..6
    # This is a linear map from (A, B[:,k]) to C[:,k]
    # The tensor is T = sum_k T_k where T_k is the 2x2x1 tensor for column k
    # 
    # For 2x2x1, the minimal rank is 2 (since it's just 2 scalar multiplications
    # per output, but actually 2x2x1 has rank 2: C1 = A11*B1 + A12*B2, C2 = A21*B1 + A22*B2)
    # Wait, 2x2x1 means multiplying 2x2 by 2x1 to get 2x1. That's rank 4 in general?
    # No: T[a,b,c] where a in {0,1} (row of A), b in {0,1} (col of A = row of B), c in {0,1} (row of C)
    # T[a,b,c] = 1 if c = a (output row = input row of A) and b is the column of A = row of B
    # Actually for 2x2x1: C[c] = sum_b A[c,b] * B[b]
    # So T[c,b,?] ... this is getting confusing.
    
    # Let me just implement a direct search for rank r decompositions using
    # a randomized approach with half-integer rounding.
    
    def try_random_rank(r, num_trials=1000):
        """Try to find a rank-r decomposition using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random U and V with half-integer entries
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            
            # Solve for W: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:,:,c] = sum_r (U * V_r) W[c,r]
            # Let M_r = U * V_r (element-wise), shape (4, 14)
            # T[:,:,c] = sum_r M_r * W[c,r]
            # This is a linear system in W[c, :]
            
            # Build the system: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # Let's vectorize: T_flat[c] = A_mat @ W[c]
            # where A_mat has entries A_mat[(a,b), r] = U[a,r]*V[b,r]
            
            A_mat = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    for r_idx in range(r):
                        A_mat[a * m*p + b, r_idx] = U[a, r_idx] * V[b, r_idx]
            
            # For each c, solve A_mat @ W[c] = T_flat[c]
            # T_flat[c] is the vector of T[a,b,c] for all (a,b)
            
            W = np.zeros((p*n, r))
            valid = True
            for c in range(p*n):
                T_col = np.zeros(n*m * m*p)
                for a in range(n*m):
                    for b in range(m*p):
                        T_col[a * m*p + b] = float(T[a, b, c])
                
                # Solve least squares
                try:
                    W_c, residuals, rank, sv = np.linalg.lstsq(A_mat, T_col, rcond=None)
                except:
                    valid = False
                    break
                
                W[c, :] = W_c
            
            if not valid:
                continue
            
            # Check if W is half-integer
            if not is_half_integer(W):
                continue
            
            # Verify
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks from 14 to 27
    for r in range(14, min(best_rank, 28)):
        if time.time() > deadline:
            break
        
        result = try_random_rank(r, num_trials=200)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    # Return best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
