"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    # U, V, W are expected to be exact (int or Fraction)
                    # But for speed in verify during search, we might have floats?
                    # The contract says integer/half-integer. 
                    # Let's assume U, V, W are passed as lists of numbers.
                    # To be safe, if they are floats, check closeness.
                    # If they are Fractions, check exact equality.
                    try:
                        if isinstance(U[i * m + j][c], Fraction):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        else:
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    except:
                        return False
                
                # Check if s is 1
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1.0) > 1e-9:
                        return False
    return True

def _schoolbook():
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

def _gradient_step(U, V, W, n, m, p, lr):
    """
    Perform one step of gradient descent on the objective:
    sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
    """
    n, m, p = n, m, p
    rank = len(U[0])
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Residual R[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
    # We need dL/dU, dL/dV, dL/dW
    # L = sum_{i,j,k} R_{ijk}^2
    # dL/dU[i,j,c] = 2 * sum_{i,j,k} R_{ijk} * V[j,k,c] * W[k,i,c]
    
    # Efficient computation using Einstein summation or loops
    # R shape: (n, m, p)
    # U shape: (n, m, rank)
    # V shape: (m, p, rank)
    # W shape: (p, n, rank)
    
    # Let's compute R first
    # R[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a tensor contraction.
    # We can do this by looping over c and accumulating?
    # Or using np.einsum
    # R = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np) - 1.0
    
    # Let's try einsum
    R = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np) - 1.0
    
    # Gradients
    # dL/dU[i,j,c] = 2 * sum_{k} R[i,j,k] * V[j,k,c] * W[k,i,c]
    # dL/dV[j,k,c] = 2 * sum_{i} R[i,j,k] * U[i,j,c] * W[k,i,c]
    # dL/dW[k,i,c] = 2 * sum_{j} R[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # Note: W is indexed [k, i, c] in the formula above?
    # Wait, W shape is (p, n, rank). So W[k, i, c].
    # In the product U[i,j,c] * V[j,k,c] * W[k,i,c], W is indeed [k,i,c].
    
    # dU = 2 * np.einsum('ijk,jkc,kic->ijc', R, V_np, W_np)
    # dV = 2 * np.einsum('ijk,ijc,kic->jkc', R, U_np, W_np)
    # dW = 2 * np.einsum('ijk,ijc,jkc->kic', R, U_np, V_np)
    
    dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', R, V_np, W_np)
    dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', R, U_np, W_np)
    dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', R, U_np, V_np)
    
    # Update
    U_np -= lr * dU
    V_np -= lr * dV
    W_np -= lr * dW
    
    return U_np, V_np, W_np

def _try_gradient_descent(rank, n, m, p, max_iter=1000, lr=0.01):
    """
    Initialize U, V, W with small random noise and run gradient descent.
    """
    np.random.seed(42) # Deterministic for reproducibility in this function, though outer loop might vary
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for i in range(max_iter):
        U, V, W = _gradient_step(U, V, W, n, m, p, lr)
        
        # Check convergence
        # Compute residual norm
        U_np = U
        V_np = V
        W_np = W
        R = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np) - 1.0
        err = np.linalg.norm(R)
        
        if err < 1e-6:
            # Found a good candidate, try to recover exact values
            return U, V, W
        if i % 100 == 0 and i > 0:
             # Decay learning rate
             lr *= 0.9
             
    return U, V, W

def _recover_exact(U, V, W, n, m, p, max_denom=100):
    """
    Try to find exact rational (integer or half-integer) values close to the float solution.
    """
    # For each entry, find the closest rational with small denominator
    # The problem states "integer/half-integer". So denominators 1 or 2.
    # Let's check if rounding to nearest 0.5 works.
    
    U_exact = [[0]*len(U[0]) for _ in range(len(U))]
    V_exact = [[0]*len(V[0]) for _ in range(len(V))]
    W_exact = [[0]*len(W[0]) for _ in range(len(W))]
    
    for i in range(len(U)):
        for c in range(len(U[0])):
            val = U[i][c]
            # Nearest half-integer
            nearest = round(val * 2) / 2.0
            U_exact[i][c] = nearest
            if abs(val - nearest) > 1e-3:
                return None # Too far from half-integer grid
                
    for j in range(len(V)):
        for c in range(len(V[0])):
            val = V[j][c]
            nearest = round(val * 2) / 2.0
            V_exact[j][c] = nearest
            if abs(val - nearest) > 1e-3:
                return None

    for k in range(len(W)):
        for c in range(len(W[0])):
            val = W[k][c]
            nearest = round(val * 2) / 2.0
            W_exact[k][c] = nearest
            if abs(val - nearest) > 1e-3:
                return None

    # Verify exactness with these rounded values
    if _verify(U_exact, V_exact, W_exact, n, m, p):
        return U_exact, V_exact, W_exact
    else:
        # Maybe it's integers?
        U_int = [[int(round(x)) for x in row] for row in U_exact]
        V_int = [[int(round(x)) for x in row] for row in V_exact]
        W_int = [[int(round(x)) for x in row] for row in W_exact]
        if _verify(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
        return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best solution starts as schoolbook
    best = _schoolbook()
    best_rank = n * m * p  # 90
    
    # Theoretically, lower bound is max(n*p, m*p, n*m) = 18.
    # Strassen gives 7 for 2x2. For 3x3, best is 23 (Winograd).
    # For 3x5x6, we don't know the exact minimum.
    # Let's try ranks from 30 to 89.
    
    # We will try gradient descent for a few ranks.
    # If GD finds a solution, we verify it.
    
    # To increase chance, we can try multiple initializations or ranks.
    # But GD is expensive.
    
    # Let's try rank 40, 50, 60.
    ranks_to_try = [40, 50, 60, 70, 80]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
            
        # Try a few restarts for this rank
        found = False
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Change seed for restart
            np.random.seed(42 + restart)
            U, V, W = _try_gradient_descent(rank, n, m, p, max_iter=2000, lr=0.05)
            
            # Check if we got close to a solution
            U_np = U
            V_np = V
            W_np = W
            R = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np) - 1.0
            err = np.linalg.norm(R)
            
            if err < 1e-4:
                exact = _recover_exact(U, V, W, n, m, p)
                if exact:
                    U_e, V_e, W_e = exact
                    best = {"U": U_e, "V": V_e, "W": W_e}
                    best_rank = rank
                    found = True
                    break # Move to next rank
                    
    return best
# EVOLVE-BLOCK-END
