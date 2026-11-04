import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return True
        
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _try_rank_structured(n, m, p, rank, time_limit=5.0, seed=42):
    """
    Try to find a valid decomposition with given rank using structured search.
    Exploits the n=2 structure for W.
    """
    rng = random.Random(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    values = [0, 0.5, -0.5, 1, -1]
    
    start_time = time.time()
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Initialize with sparse random values, biased toward 1
        for _ in range(rank * 3):
            U[rng.randrange(nm)][rng.randrange(rank)] = rng.choice(values)
            V[rng.randrange(mp)][rng.randrange(rank)] = rng.choice(values)
            W[rng.randrange(pn)][rng.randrange(rank)] = rng.choice(values)
        
        # Local search with targeted mutations
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
            # Choose which matrix to mutate and which entry
            var_idx = rng.randrange(3)
            c = rng.randrange(rank)
            
            if var_idx == 0:
                idx = rng.randrange(nm)
                U[idx][c] = rng.choice(values)
            elif var_idx == 1:
                idx = rng.randrange(mp)
                V[idx][c] = rng.choice(values)
            else:
                idx = rng.randrange(pn)
                W[idx][c] = rng.choice(values)
                
    return None

def _try_rank_grad_descent(n, m, p, rank, time_limit=5.0, seed=42):
    """
    Use gradient descent on a smooth relaxation, then round to half-integers.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute gradients
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            # Compute current tensor values and errors
            errors = np.zeros((n, m, p))
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                        t_val = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                        err = t_val - 1.0
                        errors[i, j, k] = err
                        
                        # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c]
                        dU[i*m+j, :] += err * V_3d[j, k, :] * W_3d[k, i, :]
                        
                        # dT/dV[j,k,c] = U[i,j,c] * W[k,i,c]
                        dV[j*p+k, :] += err * U_3d[i, j, :] * W_3d[k, i, :]
                        
                        # dT/dW[k,i,c] = U[i,j,c] * V[j,k,c]
                        dW[k*n+i, :] += err * U_3d[i, j, :] * V_3d[j, k, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check if close to solution
            max_err = np.max(np.abs(errors))
            if max_err < 1e-6:
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding failed, try to fix with small adjustments
                for fix_step in range(100):
                    if time.time() - start_time > time_limit:
                        break
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # Pick a random entry and adjust by 0.5
                    var_idx = rng.randrange(3)
                    if var_idx == 0:
                        r, c = rng.randrange(nm), rng.randrange(rank)
                        U_list[r][c] += rng.choice([-0.5, 0.5])
                    elif var_idx == 1:
                        r, c = rng.randrange(mp), rng.randrange(rank)
                        V_list[r][c] += rng.choice([-0.5, 0.5])
                    else:
                        r, c = rng.randrange(pn), rng.randrange(rank)
                        W_list[r][c] += rng.choice([-0.5, 0.5])
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill
    
    # First, get the schoolbook baseline
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use both structured search and gradient descent
    for rank in range(49, 39, -1):
        if time.time() > deadline:
            break
        
        # Try structured search first
        result = _try_rank_structured(n, m, p, rank, time_limit=3.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
        
        # Try gradient descent
        result = _try_rank_grad_descent(n, m, p, rank, time_limit=3.0, seed=rank+1000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
