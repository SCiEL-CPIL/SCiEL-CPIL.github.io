"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    
    The tensor T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For matrix multiplication, T[i, j, k] = 1 if we're computing (A*B)[i,k] which involves sum over j,
    but actually the standard bilinear form for matrix mult is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor for this is T[i, j, k] = 1 for all i,j,k? No.
    
    Actually, the matrix multiplication tensor T has entries T[i, j, k] = 1 for all valid i,j,k in the sense that
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    
    So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank_of(U, V, W):
    return len(U[0]) if U and U[0] else 0

def _make_identity_decomposition(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_gradient_descent(n, m, p, target_rank, time_budget, rng=None):
    """
    Try to find a decomposition with given rank using gradient descent on a loss function.
    The loss is the sum of squared errors over all (i,j,k) entries.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    
    # Initialize with random values
    U = np.array([[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)])
    V = np.array([[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)])
    W = np.array([[rng.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)])
    
    # Learning rate
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(10000):
        if time.time() - start_time > time_budget * 0.9:
            break
        
        # Compute current loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = s - 1.0
                    for c in range(target_rank):
                        dU[i*m+j][c] += 2 * err * V[j*p+k][c] * W[k*n+i][c]
                        dV[j*p+k][c] += 2 * err * U[i*m+j][c] * W[k*n+i][c]
                        dW[k*n+i][c] += 2 * err * U[i*m+j][c] * V[j*p+k][c]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid solution
    if best_loss < 1e-10:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_structured_patterns(n, m, p, time_budget, rng=None):
    """
    Try some structured patterns that might exploit the algebraic structure
    of matrix multiplication.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    
    # Pattern 1: Try to use the fact that we're multiplying 2x4 by 4x5
    # Maybe we can share some computation across the two rows of A
    
    # Let's try a decomposition where some of the rank-1 terms are shared
    # between the two i values (rows of A)
    
    for target_rank in range(n*m*p - 1, max(20, n*m*p - 15), -1):
        if time.time() - start_time > time_budget * 0.7:
            break
        
        # Try a few structured initializations
        for init_type in range(3):
            if time.time() - start_time > time_budget * 0.95:
                break
            
            if init_type == 0:
                # Start from identity and perturb
                base = _make_identity_decomposition(n, m, p)
                U = [row[:] for row in base["U"][:n*m]]
                V = [row[:] for row in base["V"][:m*p]]
                W = [row[:] for row in base["W"][:p*n]]
                
                # Truncate to target_rank and add small noise
                U = [row[:target_rank] + [rng.gauss(0, 0.1) for _ in range(target_rank - len(row[:target_rank]))] for row in U]
                V = [row[:target_rank] + [rng.gauss(0, 0.1) for _ in range(target_rank - len(row[:target_rank]))] for row in V]
                W = [row[:target_rank] + [rng.gauss(0, 0.1) for _ in range(target_rank - len(row[:target_rank]))] for row in W]
                
                # Ensure correct length
                U = [row[:target_rank] for row in U]
                V = [row[:target_rank] for row in V]
                W = [row[:target_rank] for row in W]
                
            elif init_type == 1:
                # Random sparse initialization
                U = [[rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(n*m)]
                V = [[rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(m*p)]
                W = [[rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0 for _ in range(target_rank)] for _ in range(p*n)]
                
            else:
                # Integer/half-integer initialization
                U = [[rng.choice([0, 0.5, 1, 1.5, 2]) if rng.random() < 0.5 else 0.0 for _ in range(target_rank)] for _ in range(n*m)]
                V = [[rng.choice([0, 0.5, 1, 1.5, 2]) if rng.random() < 0.5 else 0.0 for _ in range(target_rank)] for _ in range(m*p)]
                W = [[rng.choice([0, 0.5, 1, 1.5, 2]) if rng.random() < 0.5 else 0.0 for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 150s wall-clock, finish well before
    
    # Start with the known schoolbook decomposition
    best_result = _make_identity_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Verify the initial decomposition
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), \
        "Initial decomposition failed verification"
    
    rng = random.Random(12345)
    
    # Strategy 1: Try structured patterns first (faster, more likely to succeed if anything works)
    result = _try_structured_patterns(n, m, p, time_budget=60, rng=rng)
    if result is not None:
        best_result = result
        best_rank = len(result["U"][0])
    
    # Strategy 2: Try gradient descent for lower ranks
    for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
        if time.time() > deadline - 10.0:
            break
        
        result = _try_gradient_descent(n, m, p, target_rank, time_budget=15, rng=rng)
        if result is not None:
            best_result = result
            best_rank = target_rank
    
    return best_result
