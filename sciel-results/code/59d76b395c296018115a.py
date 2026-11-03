import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We search for integer decompositions of 8*T
    T_target = 8.0 * T

    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook decomposition
    U_school = np.zeros((n*m, best_rank), dtype=int)
    V_school = np.zeros((m*p, best_rank), dtype=int)
    W_school = np.zeros((p*n, best_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Search for lower ranks
    for rank in range(24, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(2000):
            if time.time() > deadline:
                break
                
            U = np.random.choice([-2, -1, 0, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(p*n, rank))
            
            # Local search
            # Calculate initial residual
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
            R = T_target - T_decomp
            err = np.linalg.norm(R)
            
            # Iterative local search
            for iter in range(500):
                if time.time() > deadline:
                    break
                    
                # Pick a random matrix and entry to update
                mat = np.random.choice(['U', 'V', 'W'])
                t = np.random.randint(rank)
                
                if mat == 'U':
                    a = np.random.randint(n*m)
                    # Gradient: sum_{b,c} R[a,b,c] V[t,b] W[t,c]
                    grad = np.dot(V[:, t], np.dot(W[:, t], R[a, :, :])) # Wait, R is (a, b, c)
                    # R[a, :, :] is (b, c)
                    # V[:, t] is (b)
                    # W[:, t] is (c)
                    # grad = sum_b sum_c R[a,b,c] V[b,t] W[c,t]
                    # This is a scalar.
                    # Let's compute it properly.
                    # R_a = R[a, :, :]
                    # grad = np.sum(R_a * np.outer(V[:, t], W[:, t]))
                    
                    # Try flipping or adding 1
                    # We want to minimize || R + delta * (e_a \otimes V_t \otimes W_t) ||
                    # Delta R = delta * (e_a \otimes V_t \otimes W_t)
                    # Change in squared norm: 2 <R, Delta R> + ||Delta R||^2
                    # <R, Delta R> = delta * sum_{b,c} R[a,b,c] V[b,t] W[c,t]
                    # ||Delta R||^2 = delta^2 * ||V_t||^2 * ||W_t||^2
                    
                    # Let's just try delta = -1, 1
                    best_delta = 0
                    best_err = err
                    for delta in [-1, 1]:
                        new_err_approx = err + 2 * delta * np.sum(R[a, :, :] * np.outer(V[:, t], W[:, t])) + delta*delta * np.sum(V[:, t]**2) * np.sum(W[:, t]**2)
                        if new_err_approx < best_err:
                            best_err = new_err_approx
                            best_delta = delta
                            
                    if best_delta != 0:
                        U[a, t] += best_delta
                        R[a, :, :] += best_delta * np.outer(V[:, t], W[:, t])
                        err = best_err # Approximate, recalc occasionally?
                        # For exactness check, we need to be careful.
                        # Let's recalculate err every 100 steps or if we want to be safe.
                        # But for speed, we use the approximation.
                        
                elif mat == 'V':
                    b = np.random.randint(m*p)
                    # grad = sum_{a,c} R[a,b,c] U[a,t] W[c,t]
                    # Delta R = delta * (U_t \otimes e_b \otimes W_t)
                    # <R, Delta R> = delta * sum_{a,c} R[a,b,c] U[a,t] W[c,t]
                    # ||Delta R||^2 = delta^2 * ||U_t||^2 * ||W_t||^2
                    best_delta = 0
                    best_err = err
                    for delta in [-1, 1]:
                        # R[:, b, :] is (a, c)
                        # U[:, t] is (a)
                        # W[:, t] is (c)
                        # sum = np.sum(R[:, b, :] * np.outer(U[:, t], W[:, t]))
                        new_err_approx = err + 2 * delta * np.sum(R[:, b, :] * np.outer(U[:, t], W[:, t])) + delta*delta * np.sum(U[:, t]**2) * np.sum(W[:, t]**2)
                        if new_err_approx < best_err:
                            best_err = new_err_approx
                            best_delta = delta
                            
                    if best_delta != 0:
                        V[b, t] += best_delta
                        R[:, b, :] += best_delta * np.outer(U[:, t], W[:, t])
                        err = best_err
                        
                else: # W
                    c = np.random.randint(p*n)
                    # grad = sum_{a,b} R[a,b,c] U[a,t] V[b,t]
                    # Delta R = delta * (U_t \otimes V_t \otimes e_c)
                    # <R, Delta R> = delta * sum_{a,b} R[a,b,c] U[a,t] V[b,t]
                    # ||Delta R||^2 = delta^2 * ||U_t||^2 * ||V_t||^2
                    best_delta = 0
                    best_err = err
                    for delta in [-1, 1]:
                        # R[:, :, c] is (a, b)
                        # U[:, t] is (a)
                        # V[:, t] is (b)
                        # sum = np.sum(R[:, :, c] * np.outer(U[:, t], V[:, t]))
                        new_err_approx = err + 2 * delta * np.sum(R[:, :, c] * np.outer(U[:, t], V[:, t])) + delta*delta * np.sum(U[:, t]**2) * np.sum(V[:, t]**2)
                        if new_err_approx < best_err:
                            best_err = new_err_approx
                            best_delta = delta
                            
                    if best_delta != 0:
                        W[c, t] += best_delta
                        R[:, :, c] += best_delta * np.outer(U[:, t], V[:, t])
                        err = best_err
                        
                # Periodic exact check
                if iter % 50 == 0:
                    T_decomp = np.einsum('at,bt,ct->abc', U, V, W)
                    if np.allclose(T_decomp, T_target, atol=1e-6):
                        found = True
                        break
                    # Recalculate err to correct drift
                    R = T_target - T_decomp
                    err = np.linalg.norm(R)
                    
            if found:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break

    if best_U is not None:
        U_final = (best_U / 2.0).tolist()
        V_final = (best_V / 2.0).tolist()
        W_final = (best_W / 2.0).tolist()
        
        T_check = np.einsum('at,bt,ct->abc', np.array(U_final), np.array(V_final), np.array(W_final))
        if not np.allclose(T_check, T, atol=1e-6):
            U_final = (U_school / 2.0).tolist()
            V_final = (V_school / 2.0).tolist()
            W_final = (W_school / 2.0).tolist()
            
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        U_final = (U_school / 2.0).tolist()
        V_final = (V_school / 2.0).tolist()
        W_final = (W_school / 2.0).tolist()
        return {"U": U_final, "V": V_final, "W": W_final}
