import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition using NumPy tensor contraction."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
    U_np = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_np = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_np = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute T[i, j, j', k, k'] = sum_r U[i,j,r]*V[j',k,r]*W[k',i',r] for all i'
    # We need T[i, j, j', k, k'] to equal delta(i,i')*delta(j,j')*delta(k,k')
    # Actually the tensor is M[i, j, j', k] = delta(i,i')*delta(j,j')*delta(k,k') where i' is output row
    # Wait, let me re-read: the multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i, j, k, j', k'] ... no.
    # The standard matrix multiplication tensor has indices (i, j, k) for C[i,k] = sum_j A[i,j] B[j,k]
    # But here we have A(n x m) and B(m x p), so C(n x p).
    # T[i, j, k, j', k'] doesn't make sense. Let me think again.
    
    # The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has three index sets: (i,j) for A, (j,k) for B, (i,k) for C.
    # T[i, j, j', k, k'] ... no, it's T[i, j, k] where the three modes are:
    #   mode 1: (i, j) - input A indices, size n*m
    #   mode 2: (j', k) - input B indices, size m*p  
    #   mode 3: (i', k') - output C indices, size n*p
    # And T[i,j, j',k, i',k'] = 1 iff i=i', j=j', k=k'
    
    # So the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] summed over r should give T[i,j, j',k, i',k']
    
    # T[i,j, j',k, i',k'] = 1 iff i==i' and j==j' and k==k'
    
    # Let's verify by computing the full tensor and comparing.
    # For efficiency, we can check specific entries.
    
    # The number of (i,j,j',k,i',k') combinations is n*m*m*p*n*p = 2*5*5*5*2*5 = 2500
    # For each, we compute sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    U_flat = np.array(U, dtype=np.float64)  # shape (n*m, rank)
    V_flat = np.array(V, dtype=np.float64)  # shape (m*p, rank)
    W_flat = np.array(W, dtype=np.float64)  # shape (n*p, rank)
    
    # Check a subset of entries for speed, or all if feasible
    # 2500 entries * rank ops. If rank is ~30, that's 75000 ops, very fast.
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            u_idx = i * m + j
                            v_idx = jp * p + k
                            w_idx = kp * n + ip
                            val = np.dot(U_flat[u_idx, :], V_flat[v_idx, :] * W_flat[w_idx, :])
                            if abs(val - target) > 1e-9:
                                return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute total squared error of the decomposition."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return float('inf')
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            u_idx = i * m + j
                            v_idx = jp * p + k
                            w_idx = kp * n + ip
                            val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                            total_error += (val - target) ** 2
    return total_error

def _gradient_descent(U, V, W, n, m, p, r, lr=0.01, max_iters=200, timeout=20):
    """Gradient descent to minimize the error."""
    start = time.time()
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for it in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Compute error
        err = 0.0
        grad_U = np.zeros_like(U_np)
        grad_V = np.zeros_like(V_np)
        grad_W = np.zeros_like(W_np)
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                u_idx = i * m + j
                                v_idx = jp * p + k
                                w_idx = kp * n + ip
                                # val = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                                prod = U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :]
                                val = np.sum(prod)
                                diff = val - target
                                err += diff * diff
                                # Gradient w.r.t. U[u_idx, r]: diff * V[v_idx,r] * W[w_idx,r]
                                grad_U[u_idx, :] += 2 * diff * V_np[v_idx, :] * W_np[w_idx, :]
                                grad_V[v_idx, :] += 2 * diff * U_np[u_idx, :] * W_np[w_idx, :]
                                grad_W[w_idx, :] += 2 * diff * U_np[u_idx, :] * V_np[v_idx, :]
        
        # Update
        U_np -= lr * grad_U
        V_np -= lr * grad_V
        W_np -= lr * grad_W
        
        if err < 1e-12:
            break
    
    return U_np.tolist(), V_np.tolist(), W_np.tolist(), err

def _try_rank(r, n, m, p, deadline, values):
    """Try to find a rank-r decomposition using gradient descent with random restarts."""
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        U = [[random.choice(values) for _ in range(r)] for _ in range(n * m)]
        V = [[random.choice(values) for _ in range(r)] for _ in range(m * p)]
        W = [[random.choice(values) for _ in range(r)] for _ in range(p * n)]
        
        # Run gradient descent
        U, V, W, err = _gradient_descent(U, V, W, n, m, p, r, lr=0.05, max_iters=300, timeout=15)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-9:
            break
    
    if best_err < 1e-9 and best_U is not None:
        if _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook is a valid baseline
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    values = [-1, -0.5, 0, 0.5, 1]
    
    # Try ranks from 40 down to 25
    for r in range(40, 24, -1):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        result = _try_rank(r, n, m, p, deadline, values)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
