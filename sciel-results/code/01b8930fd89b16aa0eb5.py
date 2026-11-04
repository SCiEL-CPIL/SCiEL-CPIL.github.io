import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target tensor for 3x3x6
    # U: 9xR, V: 18xR, W: 18xR
    # Indices: u=3i+j, v=3j+k, w=6k+i
    n, m, p = 3, 3, 6
    # Target tensor is sparse.
    # We want to find U, V, W such that sum_r U[:,r] V[:,r] W[:,r] == T
    
    # We split into two 3x3x3 blocks
    # Block 1: k in 0..2
    # Block 2: k in 3..5
    
    def get_333_tensor():
        # Returns the target tensor for 3x3x3
        # Shape (9, 9, 9)
        # u=3i+j, v=3j+k, w=3k+i
        T = np.zeros((9, 9, 9), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = 3*i + j
                    v = 3*j + k
                    w = 3*k + i
                    T[u, v, w] = 1
        return T

    def solve_333(rank, timeout=5):
        # Try to find a rank 'rank' decomposition of the 3x3x3 tensor
        T = get_333_tensor()
        # Variables: U (9xrank), V (9xrank), W (9xrank)
        # Total vars: 27*rank
        # We use a simple iterative solver
        
        # Initialize
        U = np.random.randint(-2, 3, (9, rank))
        V = np.random.randint(-2, 3, (9, rank))
        W = np.random.randint(-2, 3, (9, rank))
        
        best_err = np.inf
        best_U, best_V, best_W = U, V, W
        
        # Alternating Least Squares
        # Fix V, W, solve for U.
        # T = sum_r U_r V_r W_r
        # This is linear in U if V, W are fixed.
        # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # For each u, T[u, :, :] = sum_r U[u, r] (V[:, r] W[:, r]^T)
        # This is a linear system for U[u, :].
        
        for it in range(50):
            if time.time() > timeout:
                break
            
            # Update U
            for u in range(9):
                # T[u, :, :] is a 9x9 matrix
                # We want sum_r U[u, r] M_r = T[u, :, :]
                # where M_r = V[:, r] W[:, r]^T
                # Stack M_r into a matrix M of shape (81, rank)
                # T_flat = T[u, :, :].flatten()
                # Solve M @ U[u, :] = T_flat
                M = np.zeros((81, rank))
                for r in range(rank):
                    M[:, r] = (V[:, r] * W[:, r]).flatten()
                T_flat = T[u, :, :].flatten()
                # Least squares
                sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                U[u, :] = sol
                
            # Update V
            for v in range(9):
                M = np.zeros((81, rank))
                for r in range(rank):
                    M[:, r] = (U[:, r] * W[:, r]).flatten()
                # T[:, v, :] is not contiguous in the same way
                # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
                # For fixed v, T[:, v, :] = sum_r (U[:, r] W[:, r]^T) V[v, r]
                # This is also linear in V[v, :]
                T_flat = T[:, v, :].flatten()
                sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                V[v, :] = sol
                
            # Update W
            for w in range(9):
                M = np.zeros((81, rank))
                for r in range(rank):
                    M[:, r] = (U[:, r] * V[:, r]).flatten()
                T_flat = T[:, :, w].flatten()
                sol, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
                W[w, :] = sol
            
            # Check error
            err = 0
            for r in range(rank):
                pass # Just compute full tensor error
            T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-6:
                break
        
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None

    # Try to solve 3x3x3 with rank 23
    sol_333 = None
    if time.time() < deadline:
        sol_333 = solve_333(23, timeout=30)
    
    if sol_333:
        U1, V1, W1 = sol_333
        # Construct 3x3x6
        # U is 9x46, V is 18x46, W is 18x46
        rank = 46
        U = np.zeros((9, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        # Block 1: k in 0..2
        # U1 is 9x23
        # V1 is 9x23 (indices 3j+k for k in 0..2)
        # W1 is 9x23 (indices 3k+i for k in 0..2)
        
        # Map to 3x3x6 indices
        # U is same for both blocks? No, U depends on i, j.
        # In 3x3x6, U index is 3i+j. Same as 3x3x3.
        # So U for block 1 is U1.
        U[:, 0:23] = U1
        
        # V index is 3j+k.
        # For block 1, k is 0..2. V1 index is 3j+k.
        # So V[:, 0:23] corresponds to V1.
        V[:, 0:23] = V1
        
        # W index is 6k+i.
        # For block 1, k is 0..2. W1 index is 3k+i.
        # We need to map W1[3k+i] to W[6k+i].
        for k in range(3):
            for i in range(3):
                W[6*k + i, 0:23] = W1[3*k + i, :]
                
        # Block 2: k in 3..5
        # Let k' = k - 3.
        # U is same structure.
        U[:, 23:46] = U1
        
        # V index is 3j+k.
        # For block 2, k is 3..5.
        # V1 index is 3j+k'.
        # We need to map V1[3j+k'] to V[3j+(k'+3)].
        for j in range(3):
            for k_prime in range(3):
                V[3*j + (k_prime + 3), 23:46] = V1[3*j + k_prime, :]
                
        # W index is 6k+i.
        # For block 2, k is 3..5.
        # W1 index is 3k'+i.
        # We need to map W1[3k'+i] to W[6(k'+3)+i].
        for k_prime in range(3):
            for i in range(3):
                W[6*(k_prime + 3) + i, 23:46] = W1[3*k_prime + i, :]
                
        # Verify
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        # Construct target T for 3x3x6
        T_target = np.zeros((9, 18, 18), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    u = 3*i + j
                    v = 3*j + k
                    w = 6*k + i
                    T_target[u, v, w] = 1
                    
        if np.allclose(T_calc, T_target):
            # Convert to list of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback: Schoolbook
    rank = 54
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(6):
                U[3*i+j][col] = 1
                V[3*j+k][col] = 1
                W[6*k+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
