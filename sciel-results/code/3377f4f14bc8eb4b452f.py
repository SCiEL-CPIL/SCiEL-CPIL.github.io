"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_general(n, m, p, rank, max_iters=500, seed=0, deadline=None):
    """Find rank-r decomposition for n x m x p multiplication tensor using vectorized ALS."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Update U: for each (i,j), solve A_{(i,j)}^T u_{(i,j)} = 1_p
        # A_{(i,j)}[k,:] = V[j*p+k,:] * W[k*n+i,:]  (element-wise product)
        for i in range(n):
            for j in range(m):
                # Build A: shape (p, rank)
                # V[j*p:(j+1)*p, :] is (p, rank)
                # W.reshape(p, n, rank)[:, i, :] is (p, rank)
                Vj = V[j*p:(j+1)*p, :]  # (p, rank)
                Wi = W.reshape(p, n, rank)[:, i, :]  # (p, rank)
                A = Vj * Wi  # (p, rank)
                try:
                    sol = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), solve B_{(j,k)}^T v_{(j,k)} = 1_n
        # B_{(j,k)}[i,:] = U[i*m+j,:] * W[k*n+i,:]
        for j in range(m):
            for k in range(p):
                # U.reshape(n, m, rank)[:, j, :] is (n, rank)
                # W[k*n:(k+1)*n, :] is (n, rank)
                Uj = U.reshape(n, m, rank)[:, j, :]  # (n, rank)
                Wk = W[k*n:(k+1)*n, :]  # (n, rank)
                B = Uj * Wk  # (n, rank)
                try:
                    sol = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), solve C_{(k,i)}^T w_{(k,i)} = 1_m
        # C_{(k,i)}[j,:] = U[i*m+j,:] * V[j*p+k,:]
        for k in range(p):
            for i in range(n):
                # U.reshape(n, m, rank)[i, :, :] is (m, rank)
                # V.reshape(m, p, rank)[:, k, :] is (m, rank)
                Ui = U.reshape(n, m, rank)[i, :, :]  # (m, rank)
                Vk = V.reshape(m, p, rank)[:, k, :]  # (m, rank)
                C = Ui * Vk  # (m, rank)
                try:
                    sol = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_err < 1e-8:
                break
                
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _polish_rank(U, V, W, n, m, p, target_rank, deadline=None):
    """Try to reduce rank by projecting onto a lower-dimensional subspace."""
    if target_rank >= U.shape[1]:
        return U, V, W, U.shape[1]
    
    current_rank = U.shape[1]
    # Try SVD-based projection to target_rank
    # Center and find principal components
    U_c = U - U.mean(axis=0)
    V_c = V - V.mean(axis=0)
    W_c = W - W.mean(axis=0)
    
    # Use the combined data to find a low-rank subspace
    all_data = np.vstack([U_c, V_c, W_c])
    try:
        U_svd, S, Vt = np.linalg.svd(all_data, full_matrices=False)
        # Project onto target_rank components
        P = Vt[:target_rank, :].T  # (current_rank, target_rank)
        U_new = U @ P
        V_new = V @ P
        W_new = W @ P
    except:
        return U, V, W, current_rank
    
    # Verify the projected solution still works
    if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
        return U_new, V_new, W_new, target_rank
    return U, V, W, current_rank

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, n*m*p
    
    # Strategy 1: Find a low rank decomposition for 3x3x3 and double it
    # This gives rank 2*r3 for 3x3x6
    # Try r3 from 26 down to 20 (wider range with faster ALS)
    for r3 in range(26, 19, -1):
        if time.time() > deadline - 20.0:
            break
        found = False
        for seed in range(15):
            if time.time() > deadline - 20.0:
                break
            U3, V3, W3, err = _als_general(3, 3, 3, r3, max_iters=200, seed=seed, deadline=deadline)
            if U3 is not None and err < 1e-6:
                # Construct 3x3x6 factors
                total_rank = 2 * r3
                if total_rank < best_rank:
                    U = np.zeros((n*m, total_rank))
                    V = np.zeros((m*p, total_rank))
                    W = np.zeros((p*n, total_rank))
                    
                    # Copy slice 0 (k=0,1,2)
                    U[:, 0:r3] = U3
                    for j in range(m):
                        for k in range(3):
                            V[j*p+k, 0:r3] = V3[j*3+k, 0:r3]
                    for k in range(3):
                        for i in range(n):
                            W[k*n+i, 0:r3] = W3[k*3+i, 0:r3]
                            
                    # Copy slice 1 (k=3,4,5)
                    U[:, r3:total_rank] = U3
                    for j in range(m):
                        for k in range(3):
                            V[j*p+(3+k), r3:total_rank] = V3[j*3+k, 0:r3]
                    for k in range(3):
                        for i in range(n):
                            W[(3+k)*n+i, r3:total_rank] = W3[k*3+i, 0:r3]
                            
                    # Verify
                    if _verify(U, V, W, n, m, p, tol=1e-5):
                        best_U, best_V, best_W, best_rank = U, V, W, total_rank
                        found = True
                        break
        if found:
            break

    # Strategy 2: Direct ALS for 3x3x6
    # Try ranks from best_rank - 1 down to 40
    for R in range(best_rank - 1, 39, -1):
        if time.time() > deadline - 8.0:
            break
        found = False
        for seed in range(10):
            if time.time() > deadline - 8.0:
                break
            U, V, W, err = _als_general(n, m, p, R, max_iters=100, seed=seed, deadline=deadline)
            if U is not None and err < 1e-6:
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_U, best_V, best_W, best_rank = U, V, W, R
                    found = True
                    break
        if found:
            break
    
    # Strategy 3: Try to polish/reduce rank from current best
    if best_rank > 40 and time.time() < deadline - 3.0:
        for target_r in range(best_rank - 1, max(39, best_rank - 5), -1):
            if time.time() > deadline - 3.0:
                break
            U_p, V_p, W_p, r_p = _polish_rank(best_U, best_V, best_W, n, m, p, target_r, deadline)
            if r_p < best_rank:
                best_U, best_V, best_W, best_rank = U_p, V_p, W_p, r_p
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
