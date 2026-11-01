import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # Target ranks to try (try to beat 42, then 39, etc.)
    # The schoolbook rank is n*m*p = 42.
    # Strassen-like improvements might get us to 39 or lower.
    # Let's try a few target ranks in descending order of likelihood of success.
    target_ranks = [41, 40, 39]

    best_result = None
    best_rank = n * m * p  # 42

    num_u = n * m
    num_v = m * p
    num_w = p * n

    # Precompute indices for the error function
    # The tensor T[i,j,k; i',k'] = delta_{i,i'} delta_{k,k'}
    # We want U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = T[i,j,k; i',k']
    # Wait, the prompt says W[k*n+i, r]. Let's re-read carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the index of W is k*n + i.
    # In the standard decomposition:
    # C[i,k] = sum_{j,r} A[i,j] B[j,k] U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's verify the indices.
    # U has shape (n*m, rank). Index i*m+j.
    # V has shape (m*p, rank). Index j*p+k.
    # W has shape (p*n, rank). Index k*n+i.
    
    # The equation is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if i'=i and k'=k?
    # No, the multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form:
    # sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should be 1 for all i,k?
    # Actually, the tensor M_{i,j,k} = 1.
    # The decomposition is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = M_{i,j,k}
    # But M_{i,j,k} is just 1 for all i,j,k? No.
    # The matrix multiplication tensor is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This is equivalent to:
    # sum_{j,r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} A_{i,j} B_{j,k} = C_{i,k}
    # This must hold for all A, B.
    # This implies that for each i, k:
    # sum_{j,r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 if j is summed out?
    # No. The coefficient of A_{i,j} B_{j,k} in the expansion of C_{i,k} must be 1.
    # The coefficient of A_{i',j'} B_{j',k'} in the expansion of C_{i,k} must be 0 if (i',j') != (i,j) or j' != j or k' != k.
    # Actually, it's simpler. The tensor is T_{i,j,k} = 1.
    # We want sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k?
    # No, that's not right.
    # Let's look at the standard definition.
    # The rank of matrix multiplication [m,n,p] is the smallest R such that
    # sum_{r=1}^R U^{(r)}_{i,j} V^{(r)}_{j,k} W^{(r)}_{i,k} = 1
    # Wait, the indices are:
    # U depends on i, j.
    # V depends on j, k.
    # W depends on i, k.
    # So sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1 for all i,j,k.
    # Yes, this is the correct condition.
    
    # So the error function is:
    # For all i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1 = 0
    
    # Note: W index is k*n+i.
    
    def error_func(params, R):
        # params is flat array of size R * (n*m + m*p + p*n)
        U = params[:R*num_u].reshape(R, num_u).T
        V = params[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
        W = params[R*(num_u+num_v):].reshape(R, num_w).T
        
        errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # This is a dot product of U[u_idx] and V[v_idx] * W[w_idx]
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    errors[idx] = val - 1.0
                    idx += 1
                    
        return errors

    # We use scipy.optimize.least_squares
    from scipy.optimize import least_squares

    found_solution = False
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        total_vars = R * (num_u + num_v + num_w)
        
        # Try a few random initializations
        # Since the problem is non-convex, random restarts are important.
        # Also, we can try starting from a known good solution (e.g., schoolbook) 
        # and perturbing it, or using a structured initialization.
        # For now, random restarts with different seeds.
        
        # Number of restarts depends on time.
        # Let's estimate time per restart.
        # n=2, m=3, p=7.
        # num_u=6, num_v=21, num_w=14.
        # total_vars for R=41: 41 * (6+21+14) = 41 * 41 = 1681.
        # Errors: n*m*p = 42.
        # This is a small problem. Least squares should be fast.
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(1000 + restart)
            # Initialize with small random values
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                res = least_squares(error_func, x0, args=(R,), method='lm', max_nfev=5000)
            except Exception as e:
                # print(f"Exception: {e}")
                continue
                
            if res.cost < 1e-12:
                # Check if we can snap to half-integers
                U = res.x[:R*num_u].reshape(R, num_u).T
                V = res.x[R*num_u:R*(num_u+num_v)].reshape(R, num_v).T
                W = res.x[R*(num_u+num_v):].reshape(R, num_w).T
                
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                # We need to check if the snapped solution is still exact.
                # Due to floating point errors, the solution might be slightly off.
                # If it's very close to half-integers, snapping should work.
                if np.allclose(U_snap, U, atol=1e-9) and np.allclose(V_snap, V, atol=1e-9) and np.allclose(W_snap, W, atol=1e-9):
                    # Re-verify with snapped values
                    params_snapped = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                    err = error_func(params_snapped, R)
                    if np.max(np.abs(err)) < 1e-9:
                        # Success
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                else:
                    # If snap is bad, maybe return float if exact enough?
                    # Prompt says "integer/half-integer entries".
                    # If we can't snap, we might not be able to return this.
                    # But if the float solution is exact, maybe we can find a rational representation?
                    # For now, if snap fails, we might discard or try to use float if allowed?
                    # "exactly decompose" implies exactness.
                    # If float is 0.4999999, it's effectively 0.5.
                    # Let's try to return the float values if they are very close to half-integers.
                    if np.max(np.abs(U - U_snap)) < 1e-8 and np.max(np.abs(V - V_snap)) < 1e-8 and np.max(np.abs(W - W_snap)) < 1e-8:
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }

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
