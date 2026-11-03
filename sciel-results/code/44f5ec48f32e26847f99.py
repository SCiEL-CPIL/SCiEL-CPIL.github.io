import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # a index: i*m + j, size n*m = 9
    # b index: j*p + k, size m*p = 24
    # c index: k*n + i, size p*n = 24
    
    # Construct the target tensor T as a 3D array or flattened
    # T[a, b, c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin

    # Try to find a lower rank
    # Lower bound for 3x3x3 is 19. For 3x3x8, maybe higher.
    # Let's try ranks from 50 down to 30? Or just a few random searches for R < 72.
    # Given the difficulty, maybe just try a few specific ranks.
    
    # We will try ranks from 71 down to 40.
    # For each rank, try a few random restarts.
    
    for R in range(71, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # For each a, solve min || A_a U_a - t_a ||^2
                # A_a is (m*p * p*n) x R = 576 x R
                # t_a is (m*p * p*n) x 1
                for a in range(n*m):
                    # Construct A_a
                    # A_a[(b,c), r] = V[b, r] * W[c, r]
                    # We can form this efficiently
                    # V is (m*p, R), W is (p*n, R)
                    # We need outer product of V and W reshaped?
                    # A_a is not independent of a, but the coefficients V[b,r]W[c,r] are.
                    # Let's precompute M = V^T @ W? No.
                    # We need a matrix of size (576, R) where row (b,c) is V[b,:] * W[c,:]
                    # This is the Khatri-Rao product of V and W, but flattened.
                    # Let's just build it. 576 * R is small.
                    A_a = np.zeros((m*p * p*n, R))
                    t_a = np.zeros(m*p * p*n)
                    idx = 0
                    for b in range(m*p):
                        for c in range(p*n):
                            A_a[idx, :] = V[b, :] * W[c, :]
                            t_a[idx] = T[a, b, c]
                            idx += 1
                            
                    # Solve least squares
                    # U_a = argmin || A_a U_a - t_a ||
                    try:
                        U[a, :] = np.linalg.lstsq(A_a, t_a, rcond=None)[0]
                    except:
                        pass
                        
                # Update V
                for b in range(m*p):
                    A_b = np.zeros((n*m * p*n, R))
                    t_b = np.zeros(n*m * p*n)
                    idx = 0
                    for a in range(n*m):
                        for c in range(p*n):
                            A_b[idx, :] = U[a, :] * W[c, :]
                            t_b[idx] = T[a, b, c]
                            idx += 1
                    try:
                        V[b, :] = np.linalg.lstsq(A_b, t_b, rcond=None)[0]
                    except:
                        pass
                        
                # Update W
                for c in range(p*n):
                    A_c = np.zeros((n*m * m*p, R))
                    t_c = np.zeros(n*m * m*p)
                    idx = 0
                    for a in range(n*m):
                        for b in range(m*p):
                            A_c[idx, :] = U[a, :] * V[b, :]
                            t_c[idx] = T[a, b, c]
                            idx += 1
                    try:
                        W[c, :] = np.linalg.lstsq(A_c, t_c, rcond=None)[0]
                    except:
                        pass
                
                # Check error
                # Compute reconstructed tensor
                # Rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # We can do this efficiently using einsum or loops
                # Rec = np.einsum('ar,br,cr->abc', U, V, W)
                # This might be slow if done every iteration.
                # Let's check error every 10 iterations or at the end.
                
            if it % 10 == 0 or it == 99:
                Rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(Rec - T)
                if err < 1e-9:
                    # Try to round to half-integers
                    # The problem states integer/half-integer entries.
                    # We can try rounding to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    Rec_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    err_rounded = np.linalg.norm(Rec_rounded - T)
                    
                    if err_rounded < 1e-9:
                        # Valid decomposition found
                        best_rank = R
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        # Convert to list of lists
                        return {
                            "U": best_U.tolist(),
                            "V": best_V.tolist(),
                            "W": best_W.tolist()
                        }
                    else:
                        # Maybe try rounding to integers?
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        Rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                        err_int = np.linalg.norm(Rec_int - T)
                        if err_int < 1e-9:
                            best_rank = R
                            best_U = U_int
                            best_V = V_int
                            best_W = W_int
                            return {
                                "U": best_U.tolist(),
                                "V": best_V.tolist(),
                                "W": best_W.tolist()
                            }

    # Return schoolbook if no better found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
