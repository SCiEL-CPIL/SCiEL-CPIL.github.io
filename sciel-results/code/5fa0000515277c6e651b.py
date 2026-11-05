import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Target ranks to try, from low to high
    # Based on heuristics and small cases, 20 is a lower bound candidate.
    # Schoolbook is 40.
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40]
    
    best_result = None
    best_rank = n * m * p
    
    # Initialize best with schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find better ranks
    for r in target_ranks:
        if r >= best_rank:
            continue
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank r
        # We use a random initialization and gradient descent on the error
        # Error = sum (sum_c U_ic V_jc W_kc - delta_jk)^2
        # This is a non-convex problem.
        
        # Given the small size, we can try a few random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + attempt)
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Use simple gradient descent
            # Derivatives are straightforward
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                # Compute tensor T = sum_c U[:,c] V[:,c] W[:,c]
                # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                # Reshape for efficient computation
                # U: (nm, r), V: (mp, r), W: (pn, r)
                # We need T[i, j, k]
                # Let's compute the error directly
                # This is O(n*m*p*r)
                # 2*4*5*20 = 800 operations per step. Very fast.
                
                T = np.zeros((n, m, p))
                for c in range(r):
                    u_c = U[:, c].reshape(n, m)
                    v_c = V[:, c].reshape(m, p)
                    w_c = W[:, c].reshape(p, n)
                    # Outer product of u_c, v_c, w_c? No, it's element-wise product summed over c?
                    # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                    # This is not a standard outer product of 3 vectors.
                    # It is the sum of rank-1 tensors.
                    # T = sum_c np.einsum('i,j,k->ijk', U[:,c].reshape(n,m), V[:,c].reshape(m,p), W[:,c].reshape(p,n))
                    # Wait, U[i,c] is scalar for fixed i,c.
                    # So T[i,j,k] += U[i,c] * V[j,c] * W[k,c]
                    
                    # Efficient way:
                    # T += np.einsum('i,j,k->ijk', u_c, v_c, w_c) is wrong because u_c is n x m.
                    # U[i,c] depends on i. V[j,c] depends on j. W[k,c] depends on k.
                    # So for fixed c, the tensor is outer product of vector U[:,c] (length nm), V[:,c] (length mp), W[:,c] (length pn)?
                    # No. U[i*m+j, c].
                    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
                    # This couples i and j in U, j and k in V, k and i in W.
                    
                    # Let's just compute the full T for this c
                    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # This is O(nmp)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Target tensor
                # delta_jk
                # T_target[i,j,k] = 1 if j==k else 0
                # Note: i is free.
                
                error = 0
                # Compute gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target = 1.0 if j == k else 0.0
                            diff = T[i,j,k] - target
                            error += diff * diff
                            # d(T[i,j,k])/d(U[i*m+j, c]) = V[j*p+k, c] * W[k*n+i, c]
                            # d(T[i,j,k])/d(V[j*p+k, c]) = U[i*m+j, c] * W[k*n+i, c]
                            # d(T[i,j,k])/d(W[k*n+i, c]) = U[i*m+j, c] * V[j*p+k, c]
                            
                            for c in range(r):
                                vu = V[j*p+k, c] * W[k*n+i, c]
                                uv = U[i*m+j, c] * W[k*n+i, c]
                                uw = U[i*m+j, c] * V[j*p+k, c]
                                
                                dU[i*m+j, c] += 2 * diff * vu
                                dV[j*p+k, c] += 2 * diff * uv
                                dW[k*n+i, c] += 2 * diff * uw
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if error < 1e-10:
                    # Check if we can round to half-integers
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                target = 1.0 if j == k else 0.0
                                val = 0.0
                                for c in range(r):
                                    val += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                                if abs(val - target) > 1e-9:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                    if valid:
                        best_rank = r
                        best_result = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        found = True
                        break
                        
        if found:
            # If we found a valid one, we can stop or try lower? 
            # We are iterating from low to high, so if we found one, it's the lowest in this list.
            # But we might have skipped some.
            # Since we go 20, 21, ..., if 20 fails and 21 succeeds, we return 21.
            break

    return best_result
