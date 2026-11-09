import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _schoolbook_decomp():
    """Standard (schoolbook) decomposition of 2x4x7 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
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

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication.
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen formulas for C = A*B where A,B,C are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[c11,c12],[c21,c22]]
    # M1 = (c+d)*e
    # M2 = (c+d)*(f-h)  
    # M3 = a*(e-f)
    # M4 = d*(g-h)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # c11 = M5 + M4 - M2 + M6
    # c12 = M1 + M2
    # c21 = M3 + M4
    # c22 = M1 - M3 + M5 - M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h
    # W rows: 0=c11, 1=c12, 2=c21, 3=c22
    
    # M1 = (c+d)*e
    U[2,0] = 1; U[3,0] = 1
    V[0,0] = 1
    W[1,0] = 1  # c12
    
    # M2 = (c+d)*(f-h)
    U[2,1] = 1; U[3,1] = 1
    V[1,1] = 1; V[3,1] = -1
    W[0,1] = -1  # c11
    W[1,1] = 1   # c12
    
    # M3 = a*(e-f)
    U[0,2] = 1
    V[0,2] = 1; V[1,2] = -1
    W[2,2] = 1   # c21
    W[3,2] = -1  # c22
    
    # M4 = d*(g-h)
    U[3,3] = 1
    V[2,3] = 1; V[3,3] = -1
    W[0,3] = 1   # c11
    W[2,3] = 1   # c21
    
    # M5 = (a+b)*h
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1   # c11
    W[3,4] = 1   # c22
    
    # M6 = (c-a)*(e+f)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # c11
    
    # M7 = (b-d)*(g+h)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = -1  # c22
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0
                rhs = 0.0
                for l in range(rank):
                    u_val = U[i*m + j, l]
                    v_val = V[j*p + k, l]
                    w_val = W[k*n + i, l]
                    rhs += u_val * v_val * w_val
                if abs(lhs - rhs) > tol:
                    return False
    return True

def _block_decomp_init(n, m, p):
    """Create a block-structured initialization using Strassen for 2x2x2 sub-blocks.
    Partition the 7 output columns into [2,2,2,1].
    For each 2-column group, use a 2x4x2 decomposition.
    For the 1-column group, use a 2x4x1 decomposition.
    """
    # We'll build U, V, W as numpy arrays
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Partition p=7 into [2,2,2,1]
    p_groups = [2, 2, 2, 1]
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for pg in p_groups:
        if pg == 2:
            # 2x4x2 problem: split m=4 into [2,2]
            # Two 2x2x2 Strassen blocks
            for j_offset in range(0, m, 2):
                s = _strassen_2x2x2()
                U_b = np.zeros((n*m, 7))
                V_b = np.zeros((m*p, 7))
                W_b = np.zeros((p*n, 7))
                
                # Map Strassen 2x2x2 to our indexing
                # Strassen U: rows 0=a(i=0,j=0), 1=b(i=0,j=1), 2=c(i=1,j=0), 3=d(i=1,j=1)
                # Our U: row i*m+j
                # For j_offset=0: a=U[0*2+0]=U[0], b=U[0*2+1]=U[1], c=U[1*2+0]=U[2], d=U[1*2+1]=U[3]
                # For j_offset=2: a=U[0*2+2]=U[2], b=U[0*2+3]=U[3], c=U[1*2+2]=U[4], d=U[1*2+3]=U[5]
                
                u_map = [j_offset, j_offset+1, m+j_offset, m+j_offset+1]
                v_map = [j_offset*p, (j_offset+1)*p, (j_offset+2)*p, (j_offset+3)*p]
                
                # Wait, V indexing: V[j*p+k]. For 2x2x2 with j_offset:
                # j=0: V[0*p+k] = V[k] for k=0,1
                # j=1: V[1*p+k] = V[p+k] for k=0,1
                # So v_map should be [0, 1, p, p+1] for j_offset=0
                # And [2, 3, p+2, p+3] for j_offset=2
                
                v_map = [j_offset, j_offset+1, p+j_offset, p+j_offset+1]
                
                # W: W[k*n+i]. For 2x2x2 with k_offset=0:
                # k=0: W[0*n+i] = W[i] for i=0,1
                # k=1: W[1*n+i] = W[n+i] for i=0,1
                # So w_map = [0, 1, n, n+1]
                
                w_map = [0, 1, n, n+1]
                
                for l in range(7):
                    for local_u, global_u in enumerate(u_map):
                        U_b[global_u, l] = s['U'][local_u, l]
                    for local_v, global_v in enumerate(v_map):
                        V_b[global_v, l] = s['V'][local_v, l]
                    for local_w, global_w in enumerate(w_map):
                        W_b[global_w, l] = s['W'][local_w, l]
                
                U_blocks.append(U_b)
                V_blocks.append(V_b)
                W_blocks.append(W_b)
        else:
            # pg == 1: 2x4x1 problem
            # Schoolbook: rank = n*m*pg = 2*4*1 = 8
            U_b = np.zeros((n*m, 8))
            V_b = np.zeros((m*p, 8))
            W_b = np.zeros((p*n, 8))
            col = 0
            for i in range(n):
                for j in range(m):
                    U_b[i*m+j, col] = 1
                    V_b[j*p, col] = 1  # k=0 only
                    W_b[0*n+i, col] = 1  # k=0
                    col += 1
            U_blocks.append(U_b)
            V_blocks.append(V_b)
            W_blocks.append(W_b)
    
    # Concatenate all blocks
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the residual of the tensor decomposition."""
    rank = U.shape[1]
    residual = np.zeros((n, m, p))
    for l in range(rank):
        u_col = U[:, l]
        v_col = V[:, l]
        w_col = W[:, l]
        # Reshape and outer product
        u_reshaped = u_col.reshape(n, m)
        v_reshaped = v_col.reshape(m, p)
        w_reshaped = w_col.reshape(p, n)
        # Contribution to tensor: for each (i,j,k), add U[i,j,l]*V[j,k,l]*W[k,i,l]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
    return residual

def _optimize_decomp(U, V, W, n, m, p, max_iter=200, lr=0.01, time_limit=30):
    """Optimize the decomposition using gradient descent on the residual."""
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    
    # Compute initial residual
    target = np.ones((n, m, p))
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        residual = _compute_residual(U, V, W, n, m, p)
        error = residual - target
        
        # Compute gradients
        # d/dU[i,j,l] = sum_{k} error[i,j,k] * V[j,k,l] * W[k,i,l]
        # d/dV[j,k,l] = sum_{i} error[i,j,k] * U[i,j,l] * W[k,i,l]
        # d/dW[k,i,l] = sum_{j} error[i,j,k] * U[i,j,l] * V[j,k,l]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for l in range(rank):
            u_col = U[:, l].reshape(n, m)
            v_col = V[:, l].reshape(m, p)
            w_col = W[:, l].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = error[i, j, k]
                        dU[i*m+j, l] += e * v_col[j, k] * w_col[k, i]
                        dV[j*p+k, l] += e * u_col[i, j] * w_col[k, i]
                        dW[k*n+i, l] += e * u_col[i, j] * v_col[j, k]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if np.max(np.abs(error)) < 1e-10:
            break
    
    return U, V, W

def _try_optimization(n, m, p, time_limit=60):
    """Try to find a lower rank decomposition using optimization."""
    start_time = time.time()
    
    # Start with block-structured initialization
    U, V, W = _block_decomp_init(n, m, p)
    initial_rank = U.shape[1]
    
    # Verify initial decomposition
    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
        pass  # Good, we have a valid starting point
    
    # Optimize
    U, V, W = _optimize_decomp(U, V, W, n, m, p, max_iter=100, lr=0.005, 
                                time_limit=min(time_limit - (time.time() - start_time), 40))
    
    # Verify result
    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try with smaller learning rate and more iterations
    U, V, W = _block_decomp_init(n, m, p)
    U, V, W = _optimize_decomp(U, V, W, n, m, p, max_iter=200, lr=0.001,
                                time_limit=min(time_limit - (time.time() - start_time), 30))
    
    if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    deadline = time.time() + 145.0  # Leave 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomp()
    best_rank = n * m * p
    
    # Try optimization-based approach
    try:
        time_remaining = deadline - time.time()
        if time_remaining > 10:
            result = _try_optimization(n, m, p, time_limit=time_remaining - 5)
            if result is not None:
                if _verify_decomp(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = len(result["U"][0])
    except Exception:
        pass
    
    return best_result
