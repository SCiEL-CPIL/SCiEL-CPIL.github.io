import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Keep safety margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Try to find a low rank decomposition
    # We know R(2, 4, 4) = 9.
    # R(2, 4, 6) is likely around 15-18.
    # Let's try ranks from 15 to 30.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Initialize best with schoolbook
    U_school = np.zeros((U_dim, best_rank), dtype=np.int32)
    V_school = np.zeros((V_dim, best_rank), dtype=np.int32)
    W_school = np.zeros((W_dim, best_rank), dtype=np.int32)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    for target_rank in range(15, 35):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(attempt + target_rank)
            U = np.random.randn(U_dim, target_rank)
            V = np.random.randn(V_dim, target_rank)
            W = np.random.randn(W_dim, target_rank)
            
            # Normalize to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            success = False
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                    
                # Fix U, V, solve for W
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For each c, T[:, :, c] is a matrix.
                # Let M_c be the matrix of shape (U_dim * V_dim, target_rank)
                # M_c[(a, b), r] = U[a, r] * V[b, r]
                # Then T_flat[:, c] = M_c @ W[:, c]
                
                # We can solve for all c simultaneously?
                # The system is linear in W.
                # Let's stack the equations.
                # Total equations: U_dim * V_dim * W_dim
                # Unknowns: W_dim * target_rank
                
                # This is a large system. Let's use least squares.
                # Construct A and b.
                # A has shape (U_dim * V_dim * W_dim, W_dim * target_rank)
                # b has shape (U_dim * V_dim * W_dim,)
                
                # This might be too slow for large ranks.
                # Alternative: iterate over c?
                # For each c, solve for W[:, c].
                
                for c in range(W_dim):
                    # Build M_c
                    # M_c has shape (U_dim * V_dim, target_rank)
                    # M_c[idx, r] = U[a, r] * V[b, r] where idx = a * V_dim + b
                    M_c = np.zeros((U_dim * V_dim, target_rank))
                    for a in range(U_dim):
                        for b in range(V_dim):
                            idx = a * V_dim + b
                            M_c[idx, :] = U[a, :] * V[b, :]
                    
                    # T_flat_c has shape (U_dim * V_dim,)
                    T_flat_c = T[:, :, c].ravel()
                    
                    # Solve M_c @ W_c = T_flat_c
                    # W_c is a vector of size target_rank
                    try:
                        W_c, residuals, rank, sv = np.linalg.lstsq(M_c, T_flat_c, rcond=None)
                        W[:, c] = W_c
                    except:
                        pass
                
                # Fix V, W, solve for U
                for a in range(U_dim):
                    M_a = np.zeros((V_dim * W_dim, target_rank))
                    for b in range(V_dim):
                        for c in range(W_dim):
                            idx = b * W_dim + c
                            M_a[idx, :] = V[b, :] * W[c, :]
                    T_flat_a = T[a, :, :].ravel()
                    try:
                        U_a, residuals, rank, sv = np.linalg.lstsq(M_a, T_flat_a, rcond=None)
                        U[a, :] = U_a
                    except:
                        pass
                
                # Fix U, W, solve for V
                for b in range(V_dim):
                    M_b = np.zeros((U_dim * W_dim, target_rank))
                    for a in range(U_dim):
                        for c in range(W_dim):
                            idx = a * W_dim + c
                            M_b[idx, :] = U[a, :] * W[c, :]
                    T_flat_b = T[:, b, :].ravel()
                    try:
                        V_b, residuals, rank, sv = np.linalg.lstsq(M_b, T_flat_b, rcond=None)
                        V[b, :] = V_b
                    except:
                        pass
                
                # Check error
                error = 0
                for a in range(U_dim):
                    for b in range(V_dim):
                        for c in range(W_dim):
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - T[a, b, c]) ** 2
                
                if error < 1e-6:
                    # Round to integers or half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness
                    exact = True
                    for a in range(U_dim):
                        for b in range(V_dim):
                            for c in range(W_dim):
                                val = np.dot(U_int[a, :], V_int[b, :] * W_int[c, :])
                                if not np.isclose(val, T[a, b, c], atol=1e-9):
                                    exact = False
                                    break
                            if not exact:
                                break
                        if not exact:
                            break
                    
                    if exact:
                        found = True
                        best_rank = target_rank
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        break
            
            if found:
                break
    
    # Convert to lists of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
