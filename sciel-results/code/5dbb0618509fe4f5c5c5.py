import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor positions
    T_positions = []
    T_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_positions.append((a, b, c))
                T_set.add((a, b, c))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for col, (a, b, c) in enumerate(T_positions):
            U[a, col] = 1
            V[b, col] = 1
            W[c, col] = 1
        return U, V, W
    
    U_school, V_school, W_school = schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    def als_decompose(r, max_iters=100, tol=1e-8):
        """ALS for CP decomposition of the matrix multiplication tensor."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        prev_error = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for a in range(nm):
                # Build the system: for each (b,c), the equation is:
                # sum_col U[a,col] * V[b,col] * W[c,col] = T[a,b,c]
                # This is a linear system in U[a,:]
                # Let's build the matrix A and vector b
                # A[row, col] = V[b_row, col] * W[c_row, col]
                # b[row] = T[a, b_row, c_row]
                rows = []
                for b in range(mp):
                    for c in range(pn):
                        rows.append((b, c))
                A_mat = np.zeros((len(rows), r))
                b_vec = np.zeros(len(rows))
                for idx, (b, c) in enumerate(rows):
                    A_mat[idx, :] = V[b, :] * W[c, :]
                    b_vec[idx] = 1.0 if (a, b, c) in T_set else 0.0
                # Solve A_mat @ U[a,:] = b_vec
                try:
                    U[a, :], residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            # Update V
            for b in range(mp):
                rows = []
                for a in range(nm):
                    for c in range(pn):
                        rows.append((a, c))
                A_mat = np.zeros((len(rows), r))
                b_vec = np.zeros(len(rows))
                for idx, (a, c) in enumerate(rows):
                    A_mat[idx, :] = U[a, :] * W[c, :]
                    b_vec[idx] = 1.0 if (a, b, c) in T_set else 0.0
                try:
                    V[b, :], residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            # Update W
            for c in range(pn):
                rows = []
                for a in range(nm):
                    for b in range(mp):
                        rows.append((a, b))
                A_mat = np.zeros((len(rows), r))
                b_vec = np.zeros(len(rows))
                for idx, (a, b) in enumerate(rows):
                    A_mat[idx, :] = U[a, :] * V[b, :]
                    b_vec[idx] = 1.0 if (a, b, c) in T_set else 0.0
                try:
                    W[c, :], residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    return None
            
            # Compute error
            error = 0.0
            for a, b, c in T_positions:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                error += (val - 1.0) ** 2
            # Sample zero positions
            for _ in range(50):
                a = np.random.randint(0, nm)
                b = np.random.randint(0, mp)
                c = np.random.randint(0, pn)
                if (a, b, c) not in T_set:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error += val ** 2
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        return U, V, W
    
    # Try to find a lower rank
    for r in range(best_rank - 1, max(30, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        result = als_decompose(r)
        if result is not None:
            U, V, W = result
            # Check exactness
            exact = True
            for a, b, c in T_positions:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    exact = False
                    break
            if exact:
                # Check zero positions more thoroughly
                zero_ok = True
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if (a, b, c) not in T_set:
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                if abs(val) > 1e-6:
                                    zero_ok = False
                                    break
                        if not zero_ok:
                            break
                    if not zero_ok:
                        break
                if zero_ok:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
