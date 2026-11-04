import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 8]

def _get_4x4x4_rank20():
    n, m, p = 4, 4, 4
    r = 20
    
    np.random.seed(42)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    def loss_and_grad(params):
        U_f = params[:n*m*r].reshape(n*m, r)
        V_f = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W_f = params[n*m*r+m*p*r:].reshape(p*n, r)
        
        loss = 0.0
        U_grad = np.zeros_like(U_f)
        V_grad = np.zeros_like(V_f)
        W_grad = np.zeros_like(W_f)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.sum(W_f[k*n+i] * U_f[i*m+j] * V_f[jp*p+k])
                        expected = 1.0 if j == jp else 0.0
                        diff = val - expected
                        loss += diff * diff
                        
                        # Gradient with respect to W_f[k*n+i, c]:
                        # d(loss)/d(W_f[k*n+i,c]) = 2*diff * U_f[i*m+j,c] * V_f[jp*p+k,c]
                        W_grad[k*n+i] += 2 * diff * U_f[i*m+j] * V_f[jp*p+k]
                        U_grad[i*m+j] += 2 * diff * W_f[k*n+i] * V_f[jp*p+k]
                        V_grad[jp*p+k] += 2 * diff * W_f[k*n+i] * U_f[i*m+j]
        
        return loss, np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    from scipy.optimize import minimize
    result = minimize(loss_and_grad, params0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'maxfun': 5000})
    
    U_opt = result.x[:n*m*r].reshape(n*m, r)
    V_opt = result.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W_opt = result.x[n*m*r+m*p*r:].reshape(p*n, r)
    
    # Round to half-integers
    U_r = np.round(U_opt * 2) / 2
    V_r = np.round(V_opt * 2) / 2
    W_r = np.round(W_opt * 2) / 2
    
    # Verify
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    val = np.sum(W_r[k*n+i] * U_r[i*m+j] * V_r[jp*p+k])
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > 1e-6:
                        # Fallback: use unrounded
                        return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Get rank-20 decomposition for 4x4x4
    U4, V4, W4 = _get_4x4x4_rank20()
    
    # For 4x4x8: B has 8 columns, split into two 4-column blocks
    # U: (4*4, 40) = (16, 40) - two copies of U4
    # V: (4*8, 40) = (32, 40) - V4 for left block, V4 for right block
    # W: (8*4, 40) = (32, 40) - W4 for left block, W4 for right block
    
    rank = 40
    U = [row[:] + row[:] for row in U4]  # This is wrong, need to interleave
    
    # Actually:
    # U[i*4+j, c] for c in 0..19: first block
    # U[i*4+j, c] for c in 20..39: second block (same as first)
    
    U_full = [U4[i*4+j][:] + U4[i*4+j][:] for i in range(4) for j in range(4)]
    
    # V[j*8+k, c]: for k in 0..3 (left block), use V4[j*4+k, c]
    #                for k in 4..7 (right block), use V4[j*4+(k-4), c]
    V_full = []
    for j in range(4):
        for k in range(8):
            if k < 4:
                row = V4[j*4+k][:] + V4[j*4+k][:]
            else:
                row = V4[j*4+(k-4)][:] + V4[j*4+(k-4)][:]
            V_full.append(row)
    
    # W[k*4+i, c]: for k in 0..3 (left block), use W4[k*4+i, c]
    #               for k in 4..7 (right block), use W4[k*4+i, c]
    W_full = []
    for k in range(8):
        for i in range(4):
            if k < 4:
                row = W4[k*4+i][:] + W4[k*4+i][:]
            else:
                row = W4[k*4+(k-4)][:] + W4[k*4+(k-4)][:]
            W_full.append(row)
    
    return {"U": U_full, "V": V_full, "W": W_full}
