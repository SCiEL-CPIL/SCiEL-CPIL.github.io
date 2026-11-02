"""Matrix multiplication tensor decomposition 3x5x7 with structured search."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition satisfies T[i*m+j, j*p+k, k*n+i] = 1 for all valid triples."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 105
    
    def time_left():
        return deadline - time.time()
    
    # Build the trivial decomposition as fallback
    U_trivial = np.zeros((n*m, trivial_rank), dtype=float)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=float)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1.0
                V_trivial[j*p+k, col] = 1.0
                W_trivial[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = trivial_rank
    
    # Precompute valid triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    
    def try_rank(r, max_iters=300, seed=0):
        """Try to find a rank-r decomposition using alternating least squares."""
        if time_left() < 15:
            return None
        
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Check convergence
            if it % 20 == 0:
                max_err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-8:
                    return U, V, W
            
            # Update U: for each row a, solve linear system
            # For row a = i*m+j, the constraints are for all k:
            # sum_t U[a,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            # This is: U[a,:] @ (V[j*p+k,:] * W[k*n+i,:]) = 1 for each k
            # So we have p equations in r unknowns for U[a,:]
            
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Build system: for each k, V[j*p+k,:] * W[k*n+i,:] gives a row
                    A_mat = np.zeros((p, r))
                    b_vec = np.ones(p)
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A_mat @ U[a,:] = b_vec (least squares if overdetermined)
                    if r <= p:
                        # Underdetermined or square: use lstsq
                        U[a, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    else:
                        # Overdetermined: use lstsq
                        U[a, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update V: for each row b = j*p+k, solve linear system
            # For row b = j*p+k, the constraints are for all i:
            # sum_t U[i*m+j,t] * V[b,t] * W[k*n+i,t] = 1
            # This is: V[b,:] @ (U[i*m+j,:] * W[k*n+i,:]) = 1 for each i
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    A_mat = np.zeros((n, r))
                    b_vec = np.ones(n)
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    V[b, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Update W: for each row c = k*n+i, solve linear system
            # For row c = k*n+i, the constraints are for all j:
            # sum_t U[i*m+j,t] * V[j*p+k,t] * W[c,t] = 1
            # This is: W[c,:] @ (U[i*m+j,:] * V[j*p+k,:]) = 1 for each j
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    A_mat = np.zeros((m, r))
                    b_vec = np.ones(m)
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W[c, :], residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Final check
        max_err = 0.0
        for (a, b, c) in triples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            max_err = max(max_err, abs(val - 1.0))
        if max_err < 1e-6:
            return U, V, W
        return None
    
    # Try ranks from theoretical lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 35
    for r in range(35, trivial_rank + 1):
        if time_left() < 20:
            break
        
        found = False
        for seed in range(5):
            if time_left() < 10:
                break
            result = try_rank(r, max_iters=200, seed=seed)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    # Convert to lists and round to clean values
    U_list = [[best_U[i, t] for t in range(best_rank)] for i in range(n*m)]
    V_list = [[best_V[i, t] for t in range(best_rank)] for i in range(m*p)]
    W_list = [[best_W[i, t] for t in range(best_rank)] for i in range(p*n)]
    
    # Round to nearest 0.5 if close
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for j in range(len(row)):
                val = row[j]
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) < 1e-6:
                    row[j] = nearest_half
                else:
                    row[j] = round(val, 10)
    
    return {"U": U_list, "V": V_list, "W": W_list}
