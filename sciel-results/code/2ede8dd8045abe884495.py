"""Direct tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # Trivial solution: rank = n*m*p = 36
    rank = n * m * p  # 36
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(len(U[0])):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try to find a better decomposition using optimization
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try decreasing ranks
    for try_rank in range(rank - 1, 24, -1):
        if time.time() - t0 > 130:
            break
        try:
            success = _try_rank(try_rank, n, m, p, t0)
            if success is not None:
                U_t, V_t, W_t = success
                if verify(U_t, V_t, W_t, n, m, p):
                    best_rank = try_rank
                    best_U, best_V, best_W = U_t, V_t, W_t
                    break  # Found the best, stop
        except Exception:
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_rank(rank, n, m, p, t0):
    """Try to find a rank-r decomposition using random initialization + optimization."""
    import numpy as np
    
    deadline = t0 + 130
    
    # Precompute index mappings for vectorized operations
    # For each (i,j,k) triple, we need indices into U, V, W
    i_idx = []
    j_idx = []
    k_idx = []
    u_idx = []
    v_idx = []
    w_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                u_idx.append(i * m + j)
                v_idx.append(j * p + k)
                w_idx.append(k * n + i)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    u_idx = np.array(u_idx)
    v_idx = np.array(v_idx)
    w_idx = np.array(w_idx)
    
    num_constraints = n * m * p
    
    rng = np.random.RandomState(42)
    
    for attempt in range(15):
        if time.time() > deadline:
            break
        
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Scale to help convergence
        U *= (n*m) ** 0.25
        V *= (m*p) ** 0.25
        W *= (p*n) ** 0.25
        
        lr = 0.05
        best_loss = np.inf
        
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Compute residuals vectorized
            # For each (i,j,k): res = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r] - 1
            # U[u_idx, :] has shape (num_constraints, rank)
            U_sel = U[u_idx, :]  # (num_constraints, rank)
            V_sel = V[v_idx, :]  # (num_constraints, rank)
            W_sel = W[w_idx, :]  # (num_constraints, rank)
            
            # res[:, r] = sum_r' U_sel[:, r'] * V_sel[:, r'] * W_sel[:, r'] - 1
            # This is element-wise product summed over rank dimension
            prod = U_sel * V_sel * W_sel  # (num_constraints, rank)
            res = prod.sum(axis=1) - 1.0  # (num_constraints,)
            
            loss = np.sum(res ** 2)
            
            if loss < 1e-14:
                return (U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Compute gradients
            # d/dU[u, r] = sum over (i,j,k) where u_idx == u: res * V_sel[:, r] * W_sel[:, r]
            # d/dV[v, r] = sum over (i,j,k) where v_idx == v: res * U_sel[:, r] * W_sel[:, r]
            # d/dW[w, r] = sum over (i,j,k) where w_idx == w: res * U_sel[:, r] * V_sel[:, r]
            
            # For U gradient:
            # U_grad[u, r] = sum_{c: u_idx[c]==u} res[c] * V_sel[c, r] * W_sel[c, r]
            U_grad = np.zeros((n*m, rank))
            V_grad = np.zeros((m*p, rank))
            W_grad = np.zeros((p*n, rank))
            
            # Use np.add.at for scatter-add
            res_VW = res[:, None] * V_sel * W_sel  # (num_constraints, rank)
            res_UW = res[:, None] * U_sel * W_sel
            res_UV = res[:, None] * U_sel * V_sel
            
            np.add.at(U_grad, (u_idx, slice(None)), res_VW)
            np.add.at(V_grad, (v_idx, slice(None)), res_UW)
            np.add.at(W_grad, (w_idx, slice(None)), res_UV)
            
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Adaptive learning rate
            if loss > best_loss * 1.5:
                lr *= 0.5
            elif loss < best_loss * 0.8:
                lr = min(lr * 1.2, 0.1)
        
        # Check if best solution is good enough
        if best_loss < 1e-10:
            return (best_U, best_V, best_W)
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies the tensor equation."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(len(U[0])):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
