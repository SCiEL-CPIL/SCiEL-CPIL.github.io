import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    # Shapes
    U_shape = (n * m, rank)
    V_shape = (m * p, rank)
    W_shape = (p * n, rank)
    
    # Indices for the tensor entries
    # T[i, j, k] = 1
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def unpack(params):
        u_flat = params[:n*m*rank]
        v_flat = params[n*m*rank : n*m*rank + m*p*rank]
        w_flat = params[n*m*rank + m*p*rank :]
        
        U = u_flat.reshape(U_shape)
        V = v_flat.reshape(V_shape)
        W = w_flat.reshape(W_shape)
        return U, V, W

    def residuals(params):
        U, V, W = unpack(params)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res.append(val - 1.0)
        return np.array(res)

    # Try to find a solution
    best_solution = None
    best_loss = np.inf
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Let's try small values.
        scale = 1.0
        params = np.random.normal(0, scale, size=(n*m + m*p + p*n) * rank)
        
        try:
            result = least_squares(residuals, params, method='lm', max_nfev=1000)
            if result.success or result.cost < best_loss:
                best_loss = result.cost
                best_solution = result.x
        except Exception:
            pass

    if best_solution is not None and best_loss < 1e-6:
        U, V, W = unpack(best_solution)
        
        # Convert to half-integers
        # Multiply by 2, round, divide by 2
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        # Verify exactness
        # We need to check if the rounded values satisfy the equations exactly
        # Using integer arithmetic: 2*U, 2*V, 2*W are integers.
        # Let U2 = 2*U_half, etc.
        # Condition: Sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = 8 ?
        # No, (U*V*W) = 1.
        # (U2/2 * V2/2 * W2/2) = 1 => U2*V2*W2 = 8.
        
        U2 = (U_half * 2).astype(int)
        V2 = (V_half * 2).astype(int)
        W2 = (W_half * 2).astype(int)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r]
                    if val != 8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Return as lists of floats (half-integers)
            U_list = U_half.tolist()
            V_list = V_half.tolist()
            W_list = W_half.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook if optimization fails
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
