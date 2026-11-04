import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # Define the target tensor T
    # T[i, j, k] is non-zero if it corresponds to a multiplication term
    # The tensor is T_{(i,j), (j,k), (k,i)} = 1
    # We check the equality: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k]
    # T[i,j,k] is 1 for all i,j,k? No.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that T(A,B) = C.
    # The bilinear form is <C, T(A,B)> = sum_{i,k} C[k,i] sum_j A[i,j] B[j,k]
    # So T_{i,j, j',k', k,i'} = 1 if j=j', k=k', i=i'.
    # In our indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We sum over r.
    # This sum must equal 1 if the indices match the multiplication structure?
    # Actually, for a fixed (i,j,k), the term in the sum is non-zero only if r contributes.
    # The target value for the triple (i,j,k) is 1?
    # Let's check schoolbook.
    # For schoolbook, each r corresponds to a specific (i,j,k).
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # The product is 1.
    # Sum is 1.
    # So the target T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    # Wait, is T[i,j,k] always 1?
    # If I change i to i', the term W[k*n+i', r] is 0 for the schoolbook r of (i,j,k).
    # So the sum is 0 for i' != i.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # So we need to solve:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    num_eqs = n * m * p
    # num_eqs = 56
    
    def residual(params, R):
        # params is a flat array of size R * (nm + mp + pn)
        # Split into U, V, W
        nm = n * m
        mp = m * p
        pn = p * n
        size_U = R * nm
        size_V = R * mp
        size_W = R * pn
        
        U = params[:size_U].reshape(R, nm)
        V = params[size_U:size_U+size_V].reshape(R, mp)
        W = params[size_U+size_V:].reshape(R, pn)
        
        res = np.zeros(num_eqs)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[:, i*m + j]
                    v_val = V[:, j*p + k]
                    w_val = W[:, k*n + i]
                    res[idx] = np.sum(u_val * v_val * w_val) - 1.0
                    idx += 1
        return res

    best_rank = n * m * p
    best_params = None
    
    # Try ranks from 24 to best_rank
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers
            # Scale: values should be small
            np.random.seed(attempt)
            params = np.random.randint(-2, 3, size=R * (n*m + m*p + p*n))
            
            # Use scipy.optimize.least_squares
            try:
                from scipy.optimize import least_squares
                # Bounds are not strictly necessary if we check later
                res = least_squares(residual, params, args=(R,), method='lm', max_nfev=5000)
                
                # Check if solution is close to 0
                if np.max(np.abs(res.fun)) < 1e-6:
                    # Check if we can round to half-integers
                    U = res.x[:R*n*m].reshape(R, n*m)
                    V = res.x[R*n*m:R*(n*m+m*p)].reshape(R, m*p)
                    W = res.x[R*(n*m+m*p):].reshape(R, p*n)
                    
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify exactness
                    res_exact = residual(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]), R)
                    if np.max(np.abs(res_exact)) < 1e-9:
                        best_rank = R
                        best_params = (U_r, V_r, W_r)
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            break

    if best_params is not None:
        U, V, W = best_params
        # Convert to lists of lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
