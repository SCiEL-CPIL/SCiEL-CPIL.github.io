import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def _make_schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    # Check integer/half-integer
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0, atol=tol):
                    return False
    return True

def _snap_and_check(U, V, W, tol=1e-9):
    # Snap to nearest half-integer
    U2 = np.round(U * 2) / 2.0
    V2 = np.round(V * 2) / 2.0
    W2 = np.round(W * 2) / 2.0
    
    # Check if the snap was close enough
    if np.linalg.norm(U - U2) > 1e-6 or np.linalg.norm(V - V2) > 1e-6 or np.linalg.norm(W - W2) > 1e-6:
        return None, None, None
        
    if _verify(U2, V2, W2, tol):
        return U2, V2, W2
    return None, None, None

def _compute_loss_and_grad(U, V, W, n, m, p, R, penalty_weight=1.0):
    """
    Compute loss and gradients for the tensor decomposition problem.
    Loss = sum_{i,j,k} (sum_r U_ijr V_jkr W_ikr - 1)^2 + penalty_weight * (half-integer violation)
    """
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Compute residuals and gradients
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # residual = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
                residual = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]) - 1.0
                loss += residual ** 2
                
                # Gradient w.r.t U[i*m+j, :]
                grad_U[i * m + j, :] += 2 * residual * (V[j * p + k, :] * W[k * n + i, :])
                
                # Gradient w.r.t V[j*p+k, :]
                grad_V[j * p + k, :] += 2 * residual * (U[i * m + j, :] * W[k * n + i, :])
                
                # Gradient w.r.t W[k*n+i, :]
                grad_W[k * n + i, :] += 2 * residual * (U[i * m + j, :] * V[j * p + k, :])
    
    # Add penalty for deviation from half-integers
    # Penalty: sum over all entries of (2*x - round(2*x))^2 * penalty_weight
    for mat, grad in [(U, grad_U), (V, grad_V), (W, grad_W)]:
        doubled = 2.0 * mat
        rounded = np.round(doubled)
        violation = doubled - rounded
        loss += penalty_weight * np.sum(violation ** 2)
        grad += penalty_weight * 2.0 * violation * 2.0  # d/dx (2x - round(2x))^2 = 2*(2x-round)*2
    
    return loss, grad_U, grad_V, grad_W

def _gradient_descent(U, V, W, n, m, p, R, max_iter=500, lr_init=0.01, lr_decay=0.995, 
                       penalty_weight_init=0.1, penalty_growth=1.05, time_budget=60.0):
    """
    Gradient descent with adaptive learning rate and penalty scheduling.
    """
    start_time = time.time()
    lr = lr_init
    penalty_weight = penalty_weight_init
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Compute loss and gradients
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grad(U, V, W, n, m, p, R, penalty_weight)
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Update parameters
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= lr_decay
        
        # Grow penalty weight slowly to encourage half-integer values
        penalty_weight *= penalty_growth
        
        # Check convergence
        if abs(loss - best_loss) < 1e-12:
            break
        
        # Try snapping if loss is small
        if loss < 1e-4:
            U_s, V_s, W_s = _snap_and_check(U, V, W)
            if U_s is not None:
                return U_s, V_s, W_s, True
    
    return best_U, best_V, best_W, False

def _try_gradient_search(R, n, m, p, seed, time_budget=30.0):
    """
    Try gradient-based search for a valid decomposition of rank R.
    """
    np.random.seed(seed)
    
    # Initialize with random values scaled to be near half-integers
    # Start with values in [-1, 1] and snap to nearest half-integer
    U = np.random.randn(n * m, R) * 0.5
    V = np.random.randn(m * p, R) * 0.5
    W = np.random.randn(p * n, R) * 0.5
    
    # Snap initial values to half-integers
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    
    # Run gradient descent
    U_out, V_out, W_out, found = _gradient_descent(U, V, W, n, m, p, R, 
                                                     max_iter=300, 
                                                     lr_init=0.02, 
                                                     lr_decay=0.997,
                                                     penalty_weight_init=0.05,
                                                     penalty_growth=1.03,
                                                     time_budget=time_budget)
    
    if found:
        return U_out, V_out, W_out
    
    # Final check
    U_s, V_s, W_s = _snap_and_check(U_out, V_out, W_out)
    if U_s is not None:
        return U_s, V_s, W_s
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Schoolbook baseline
    U_sb, V_sb, W_sb = _make_schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # 2. Try gradient-based search for lower ranks
    # Start from rank 119 down to 100
    seeds = [42, 123, 999, 777, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for R in range(119, 99, -1):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        
        # Determine how many seeds to try based on time
        if time_left > 60:
            max_seeds = 4
        elif time_left > 30:
            max_seeds = 2
        else:
            max_seeds = 1
        
        for seed in seeds[:max_seeds]:
            if time.time() > deadline:
                break
            
            U_out, V_out, W_out = _try_gradient_search(R, n, m, p, seed, 
                                                        time_budget=min(20.0, time_left - 5.0))
            
            if U_out is not None:
                # Found a valid decomposition with rank R
                best_result = {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                best_rank = R
                # Continue searching for even lower ranks
                break
        
        # If we found a valid decomposition, we could stop early if time is tight
        if time.time() > deadline - 10.0:
            break
    
    return best_result

# Fallback in case construct is not called or for direct testing
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    n, m, p = DIMENSIONS
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0):
                    ok = False
                    print(f"Fail at {i},{j},{k}: {val}")
                    break
            if not ok: break
        if not ok: break
    if ok:
        print("Verification Passed")
    else:
        print("Verification Failed")
