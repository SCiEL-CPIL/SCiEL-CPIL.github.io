import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 130.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Build rank-42 Strassen decomposition using 6 2x2x2 blocks
    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        # 6 blocks: (a_rows, a_cols, b_rows, b_cols)
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        # Strassen 7-term decomposition for 2x2x2
        strassen = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
            ([(0,1)],        [(1,1),(3,-1)],[(1,1),(3,1)]),
            ([(3,1)],        [(2,1),(0,-1)],[(0,1),(2,1)]),
            ([(0,1),(1,1)],  [(3,1)],       [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1],
                     2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1],
                     2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0],
                     2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
            
            for u_c, v_c, w_c in strassen:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return U, V, W
    
    U, V, W = make_rank42()
    
    if not check_decomposition(U, V, W):
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return to_lists(U, V, W)
    
    # Try to reduce rank by finding linear dependencies
    def try_reduce_rank(U, V, W):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            
            eliminated = False
            for r in range(rank-1, -1, -1):
                if time.time() > deadline - 15:
                    break
                
                if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    eliminated = True
                    break
                
                for s in range(r):
                    if np.allclose(U[:, r], U[:, s]) and np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        rank -= 1
                        eliminated = True
                        break
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return U, V, W
    
    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W)
    
    if check_decomposition(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced
    
    # Advanced rank reduction: find linear dependencies using tensor structure
    def advanced_rank_reduction(U, V, W, T):
        """
        Try to reduce rank by finding that some columns are linear combinations
        of others when considering the full tensor structure.
        """
        if time.time() > deadline - 30:
            return U, V, W
        
        U_orig = U.copy()
        V_orig = V.copy()
        W_orig = W.copy()
        rank = U.shape[1]
        
        # Try removing one column at a time and checking if the remaining columns
        # can still span the tensor
        for remove_idx in range(rank-1, -1, -1):
            if time.time() > deadline - 15:
                break
            
            # Skip this column
            U_test = np.delete(U_orig, remove_idx, axis=1)
            V_test = np.delete(V_orig, remove_idx, axis=1)
            W_test = np.delete(W_orig, remove_idx, axis=1)
            
            # Check if the reduced set still decomposes T
            if check_decomposition(U_test, V_test, W_test):
                U_orig = U_test
                V_orig = V_test
                W_orig = W_test
                rank -= 1
        
        if check_decomposition(U_orig, V_orig, W_orig):
            return U_orig, V_orig, W_orig
        
        return U, V, W
    
    U_adv, V_adv, W_adv = advanced_rank_reduction(U, V, W, T)
    
    if U_adv.shape[1] < U.shape[1] and check_decomposition(U_adv, V_adv, W_adv):
        U, V, W = U_adv, V_adv, W_adv
    
    # Parity-check / integer Gaussian elimination approach for rank minimization
    # Construct the multiplication tensor as a matrix and find its rank
    def parity_check_rank_reduction(U, V, W, T):
        """
        Use the tensor structure to find if we can reduce rank further.
        The multiplication tensor T can be viewed as a matrix where each entry
        T[i,j,k] corresponds to a coefficient. We check if the columns of
        (U, V, W) have linear dependencies that allow rank reduction.
        """
        if time.time() > deadline - 20:
            return U, V, W
        
        U_orig = U.copy()
        V_orig = V.copy()
        W_orig = W.copy()
        rank = U.shape[1]
        
        # Build the "parity check" matrix from the tensor structure
        # For each output position (i,j,k), we need sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # This is a system of equations. We can try to find a smaller set of columns
        # that still satisfies all equations.
        
        # Flatten the tensor
        T_flat = T.flatten()
        num_constraints = len(T_flat)
        
        # For each column r, define the vector v_r in R^(nm*mp*pn) where
        # v_r[i*mp+j*pn+k] = U[i,r]*V[j,r]*W[k,r]
        # We need sum_r v_r = T_flat
        
        # Try greedy elimination: remove columns that are "redundant"
        # A column r is redundant if v_r can be expressed as a linear combination
        # of the other columns (approximately, for integer/half-integer coefficients)
        
        for _ in range(30):  # Limit iterations for time
            if time.time() > deadline - 10:
                break
            
            eliminated = False
            for r in range(rank-1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Check if column r is redundant
                # Build the system: can we find coefficients c_s for s != r such that
                # sum_{s != r} c_s * v_s = v_r?
                
                # This is a linear system. Let's set it up.
                # For each constraint (i,j,k): sum_{s != r} c_s * U[i,s]*V[j,s]*W[k,s] = U[i,r]*V[j,r]*W[k,r]
                
                # Build the matrix A and vector b
                A = np.zeros((num_constraints, rank-1))
                b = np.zeros(num_constraints)
                
                col_idx = 0
                for s in range(rank):
                    if s == r:
                        continue
                    for i in range(n*m):
                        for j in range(m*p):
                            for k in range(p*n):
                                idx = i * (m*p) + j * p*n + k
                                A[idx, col_idx] = U[i, s] * V[j, s] * W[k, s]
                    col_idx += 1
                
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            idx = i * (m*p) + j * p*n + k
                            b[idx] = U[i, r] * V[j, r] * W[k, r]
                
                # Solve the linear system
                try:
                    # Use least squares since the system may be overdetermined
                    coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Check if the solution is close to integer/half-integer
                    # and if it satisfies the equations well
                    if np.allclose(A @ coeffs, b, atol=1e-8):
                        # Check if coefficients are approximately rational with small denominators
                        # For now, just check if the solution works
                        # Try removing column r and adding the linear combination
                        U_test = U_orig.copy()
                        V_test = V_orig.copy()
                        W_test = W_orig.copy()
                        
                        # The new column r would be a linear combination of others
                        # But since we're just removing, we need to verify the remaining
                        # columns still span the tensor
                        U_test = np.delete(U_test, r, axis=1)
                        V_test = np.delete(V_test, r, axis=1)
                        W_test = np.delete(W_test, r, axis=1)
                        
                        if check_decomposition(U_test, V_test, W_test):
                            U_orig = U_test
                            V_orig = V_test
                            W_orig = W_test
                            rank -= 1
                            eliminated = True
                            break
                except Exception:
                    pass
            
            if not eliminated:
                break
        
        if check_decomposition(U_orig, V_orig, W_orig):
            return U_orig, V_orig, W_orig
        
        return U, V, W
    
    U_pc, V_pc, W_pc = parity_check_rank_reduction(U, V, W, T)
    
    if U_pc.shape[1] < U.shape[1] and check_decomposition(U_pc, V_pc, W_pc):
        U, V, W = U_pc, V_pc, W_pc
    
    return to_lists(U, V, W)
