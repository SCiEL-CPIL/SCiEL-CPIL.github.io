import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T of shape (dim_u, dim_v, dim_w)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) are consistent, else 0
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a decomposition of rank r
    # We start with a rank slightly lower than the trivial one
    # Trivial rank is dim_u * dim_v / m ? No, trivial is n*m*p = 44.
    # Let's try ranks from 30 down to 20.
    
    def objective(params, r):
        # Unpack params
        # U: dim_u x r, V: dim_v x r, W: dim_w x r
        # Total params: (dim_u + dim_v + dim_w) * r
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        u_flat = params[:size_u]
        v_flat = params[size_u:size_u + size_v]
        w_flat = params[size_u + size_v:]
        
        U = u_flat.reshape((dim_u, r))
        V = v_flat.reshape((dim_v, r))
        W = w_flat.reshape((dim_w, r))
        
        # Reconstruct tensor
        # Rec[a, b, c] = sum_t U[a, t] * V[b, t] * W[c, t]
        # Efficient calculation:
        # Rec = sum_t np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
        # This is slow for large r.
        # Alternative: 
        # Rec = np.einsum('at,bt,ct->abc', U, V, W)
        
        Rec = np.einsum('at,bt,ct->abc', U, V, W)
        diff = Rec - T
        return np.sum(diff * diff)

    def grad(params, r):
        # Gradient of objective
        size_u = dim_u * r
        size_v = dim_v * r
        size_w = dim_w * r
        
        u_flat = params[:size_u]
        v_flat = params[size_u:size_u + size_v]
        w_flat = params[size_u + size_v:]
        
        U = u_flat.reshape((dim_u, r))
        V = v_flat.reshape((dim_v, r))
        W = w_flat.reshape((dim_w, r))
        
        Rec = np.einsum('at,bt,ct->abc', U, V, W)
        diff = Rec - T
        
        # Gradient w.r.t U:
        # dObj/dU[a, t] = 2 * sum_{b,c} diff[a, b, c] * V[b, t] * W[c, t]
        # grad_U[a, t] = 2 * (diff @ (V[:, t] * W[:, t]))? No.
        # diff is (a, b, c). V[:, t] is (b,). W[:, t] is (c,).
        # sum_{b,c} diff[a, b, c] V[b, t] W[c, t]
        # This is like contracting diff with outer product of V_t and W_t.
        
        # We can compute this efficiently.
        # Let M_t = diff contracted with V[:, t] and W[:, t]?
        # grad_U = 2 * np.einsum('abc,bt,ct->at', diff, V, W)
        # grad_V = 2 * np.einsum('abc,at,ct->bt', diff, U, W)
        # grad_W = 2 * np.einsum('abc,at,bt->ct', diff, U, V)
        
        grad_U = 2.0 * np.einsum('abc,bt,ct->at', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,at,ct->bt', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,at,bt->ct', diff, U, V)
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # We will try to find a solution with rank 30
    # If it fails, we might need to adjust.
    # Given the time limit, we do a few random restarts.
    
    best_result = None
    best_rank = None
    best_error = float('inf')
    
    # Try a few ranks
    for r in [30, 35, 40]:
        for attempt in range(5):
            # Random initialization
            np.random.seed(42 + attempt + r)
            params = np.random.randn((dim_u + dim_v + dim_w) * r) * 0.1
            
            res = minimize(objective, params, args=(r,), method='BFGS', jac=grad, options={'maxiter': 100, 'gtol': 1e-5})
            
            if res.success:
                # Check if we can snap to half-integers
                size_u = dim_u * r
                size_v = dim_v * r
                size_w = dim_w * r
                
                u_flat = res.x[:size_u]
                v_flat = res.x[size_u:size_u + size_v]
                w_flat = res.x[size_u + size_v:]
                
                U = u_flat.reshape((dim_u, r))
                V = v_flat.reshape((dim_v, r))
                W = w_flat.reshape((dim_w, r))
                
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify exactness
                Rec = np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap)
                error = np.max(np.abs(Rec - T))
                
                if error < 1e-9:
                    if r < (best_rank if best_rank else 1000):
                        best_rank = r
                        best_result = (U_snap, V_snap, W_snap)
                        break # Found a valid solution for this rank, try lower rank?
            
            # If we found a solution, we can break and try lower rank?
            # The loop iterates r from 30 to 40.
            # We want the lowest rank.
            # So we should try 30 first. If it works, we are done.
            # If 30 fails, try 35, etc.
            
    if best_result:
        U, V, W = best_result
        # Convert to list of lists of ints/floats
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to trivial decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
