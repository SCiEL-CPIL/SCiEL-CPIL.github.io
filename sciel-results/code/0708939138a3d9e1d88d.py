import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Use numerical optimization
    import numpy as np
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Build the tensor T[i,j,k] = 1 for all valid indices
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Try ranks from a lower estimate up to schoolbook-1
    # Lower bound: at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actually the tensor rank is likely higher
    
    for target_rank in range(28, rank_school):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            break
            
        # Try random restarts
        found = False
        for restart in range(50):
            if time.time() > deadline - 30:
                break
                
            np.random.seed(42 + target_rank * 1000 + restart)
            # Initialize with small random values
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Use iterative refinement
            # The condition is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a nonlinear system. Use gradient descent on the loss.
            
            loss_fn = lambda Ur, Vr, Wr: 0.0
            # Actually, let's use a different approach: alternating minimization
            
            # For fixed V and W, solve for U, etc. But it's not linear in each.
            
            # Let's try a simpler approach: use the fact that this is a tensor decomposition
            # and use a numerical solver.
            
            # Actually, for small tensors, we can set up the equations and use scipy.optimize
            from scipy.optimize import least_squares
            
            def residuals(params):
                # params: [U_flat, V_flat, W_flat]
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                
                Ur = U_flat.reshape(n*m, target_rank)
                Vr = V_flat.reshape(m*p, target_rank)
                Wr = W_flat.reshape(p*n, target_rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += Ur[i*m+j, c] * Vr[j*p+k, c] * Wr[k*n+i, c]
                            res.append(val - 1.0)
                return np.array(res)
            
            x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
            try:
                result = least_squares(residuals, x0, max_nfev=5000)
                if result.cost < 1e-8:
                    # Found a solution!
                    U_flat = result.x[:n*m*target_rank]
                    V_flat = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = result.x[n*m*target_rank + m*p*target_rank:]
                    
                    Ur = U_flat.reshape(n*m, target_rank)
                    Vr = V_flat.reshape(m*p, target_rank)
                    Wr = W_flat.reshape(p*n, target_rank)
                    
                    # Check if we can round to half-integers
                    # Try rounding each entry to nearest half-integer
                    Ur_h = np.round(Ur * 2) / 2
                    Vr_h = np.round(Vr * 2) / 2
                    Wr_h = np.round(Wr * 2) / 2
                    
                    # Verify the rounded solution
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(target_rank):
                                    val += Ur_h[i*m+j, c] * Vr_h[j*p+k, c] * Wr_h[k*n+i, c]
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-9:
                        # Convert to lists
                        U_list = [[float(x) for x in row] for row in Ur_h]
                        V_list = [[float(x) for x in row] for row in Vr_h]
                        W_list = [[float(x) for x in row] for row in Wr_h]
                        
                        # Clean up: convert 0.5 to 0.5, etc.
                        def clean_val(v):
                            if abs(v - round(v)) < 1e-9:
                                return int(round(v))
                            elif abs(v*2 - round(v*2)) < 1e-9:
                                return v
                            return v
                        
                        U_clean = [[clean_val(x) for x in row] for row in U_list]
                        V_clean = [[clean_val(x) for x in row] for row in V_list]
                        W_clean = [[clean_val(x) for x in row] for row in W_list]
                        
                        best = {"U": U_clean, "V": V_clean, "W": W_clean}
                        best_rank = target_rank
                        found = True
                        break
            except Exception as e:
                pass
        
        if found:
            # Try even lower ranks
            continue
    
    return best
