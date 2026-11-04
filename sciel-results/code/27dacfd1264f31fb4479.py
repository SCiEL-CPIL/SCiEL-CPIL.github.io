import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 135
    
    # Helper to check if a matrix is close to half-integers
    def is_half_integer(mat, tol=1e-6):
        # 2 * mat should be close to integer
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False

    def verify_solution(U, V, W, T, tol=1e-9):
        # Check exactness
        # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient check:
        # For each r, add outer product
        # Or use einsum
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Search for lower ranks
    # Strategy: Try block-diagonal decomposition with rank 6 first.
    # Then search upwards.
    
    for r in range(6, rank_schoolbook + 1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        found = False
        # Try multiple random integer initializations for V and W
        # We fix V and W to small integers, solve for U, check if U is half-integer.
        for trial in range(200):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + r * 1000 + trial)
            # Random integer matrices with small entries
            # Using -1, 0, 1 is a good start for structured tensors
            V = np.random.randint(-1, 2, size=(mp, r))
            W = np.random.randint(-1, 2, size=(pn, r))
            
            # Construct the matrix M where M[b*pn+c, r] = V[b, r] * W[c, r]
            # M is (mp*pn) x r
            # We need to solve for U: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])^T
            # Let m_r be the vector V[:, r] * W[:, r] flattened.
            # T[a, :, :] flattened = sum_r U[a, r] m_r
            # So T_flat[a] = U[a] @ M^T ?
            # T_flat[a, b*pn+c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let M be matrix with columns m_r.
            # Then T_flat[a] = U[a] @ M^T is wrong dimension.
            # T_flat[a] is a vector of length mp*pn.
            # U[a] is a vector of length r.
            # M is (mp*pn) x r.
            # (U[a] @ M^T) gives (1 x r) @ (r x mp*pn) = 1 x mp*pn.
            # So T_flat[a] = U[a] @ M^T.
            # We want U[a].
            # U[a] = T_flat[a] @ (M^T)^{-1} = T_flat[a] @ (M M^T)^{-1} M ? No.
            # If r < mp*pn, M is tall.
            # We need M to have full column rank r.
            # Then U[a] is the unique solution to U[a] @ M^T = T_flat[a] if T_flat[a] is in col space of M^T?
            # Actually, if M has full column rank, the system U[a] M^T = T_flat[a] has a unique solution
            # IF T_flat[a] is in the row space of M^T (which is the column space of M).
            # Since we are looking for an exact decomposition, we assume it exists.
            # We can solve using least squares: U[a] = argmin || U[a] M^T - T_flat[a] ||.
            # If a solution exists, the residual is 0.
            
            # Build M efficiently
            # M[b*pn+c, r] = V[b, r] * W[c, r]
            M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
            
            # Check if M has full column rank
            if np.linalg.matrix_rank(M) < r:
                continue
                
            # Solve for U
            # U is (nm x r)
            U = np.zeros((nm, r))
            for a in range(nm):
                target = T[a, :, :].flatten()
                # Solve U[a] @ M^T = target
                # This is equivalent to M @ U[a]^T = target^T
                # So U[a]^T = M^{-1} target^T (in least squares sense)
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                    U[a, :] = sol
                except:
                    continue
                    
            # Check if U is half-integer
            if is_half_integer(U):
                # Snap to nearest half-integer
                U_snapped = np.round(2.0 * U) / 2.0
                # Verify exactness
                if verify_solution(U_snapped, V, W, T):
                    best_rank = r
                    best_U, best_V, best_W = U_snapped, V, W
                    found = True
                    break
                    
        if found:
            break
            
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
