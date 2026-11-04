"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
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

def _gradient_descent(U, V, W, n, m, p, max_iter=5000, lr=0.01, tol=1e-10, deadline=None, momentum=0.9):
    """Gradient descent with momentum to minimize ||S - 1||^2 where S = U ⊙ V ⊙ W."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Momentum velocities
    vU = np.zeros_like(U_arr)
    vV = np.zeros_like(V_arr)
    vW = np.zeros_like(W_arr)
    
    # Adaptive learning rate
    current_lr = lr
    prev_loss = np.inf
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute S = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] (transposed indices)
        # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        
        # Loss = sum (S - 1)^2
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
            patience = 0
        else:
            patience += 1
        
        if loss < tol:
            break
        
        # Adaptive learning rate: reduce if not improving
        if patience >= max_patience:
            current_lr *= 0.5
            patience = 0
            if current_lr < 1e-8:
                break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * (S[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Momentum update
        vU = momentum * vU - current_lr * dU
        vV = momentum * vV - current_lr * dV
        vW = momentum * vW - current_lr * dW
        
        U_arr += vU
        V_arr += vV
        W_arr += vW
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify. Also try integers."""
    # First try integers (subset of half-integers)
    U_int = [[int(round(x)) for x in row] for row in U]
    V_int = [[int(round(x)) for x in row] for row in V]
    W_int = [[int(round(x)) for x in row] for row in W]
    if _verify(U_int, V_int, W_int, n, m, p, tol=tol):
        return U_int, V_int, W_int
    
    # Try half-integers
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    
    # Try floor and ceil for each entry independently (limited search)
    # For small ranks, try all combinations of floor/ceil for each entry
    U_f = [[int(x * 2) / 2 for x in row] for row in U]
    V_f = [[int(x * 2) / 2 for x in row] for row in V]
    W_f = [[int(x * 2) / 2 for x in row] for row in W]
    U_c = [[int(x * 2 + 1) / 2 for x in row] for row in U]
    V_c = [[int(x * 2 + 1) / 2 for x in row] for row in V]
    W_c = [[int(x * 2 + 1) / 2 for x in row] for row in W]
    
    # Try all 8 combinations
    for ui in [U_f, U_c]:
        for vi in [V_f, V_c]:
            for wi in [W_f, W_c]:
                if _verify(ui, vi, wi, n, m, p, tol=tol):
                    return ui, vi, wi
    
    return None

def _structured_init(n, m, p, rank, seed=None):
    """Structured initialization based on partial schoolbook patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Distribute schoolbook terms across rank dimensions with random signs
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Assign to a random column
                c = np.random.randint(0, rank)
                sign_u = np.random.choice([-1, 1])
                sign_v = np.random.choice([-1, 1])
                sign_w = np.random.choice([-1, 1])
                U[i * m + j, c] = sign_u
                V[j * p + k, c] = sign_v
                W[k * n + i, c] = sign_w
    
    # Add small random noise to break symmetry
    U += np.random.randn(n * m, rank) * 0.1
    V += np.random.randn(m * p, rank) * 0.1
    W += np.random.randn(p * n, rank) * 0.1
    
    return U.tolist(), V.tolist(), W.tolist()

def _polish(U, V, W, n, m, p, deadline=None, max_iter=10000):
    """High-precision polish pass with very small learning rate."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
        
        if loss < 1e-14:
            break
        
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Very small step
        lr = 1e-6
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Phase 1: Random initializations
        max_random_attempts = 50
        for attempt in range(max_random_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = target_rank * 10000 + attempt
            U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed)
            
            U_opt, V_opt, W_opt = _gradient_descent(
                U_init, V_init, W_init, n, m, p,
                max_iter=3000, lr=0.05, tol=1e-12,
                deadline=deadline - 5, momentum=0.9
            )
            
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9)
                if result is not None:
                    U_r, V_r, W_r = result
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    found = True
                    break
        
        # Phase 2: Structured initializations
        if not found and time.time() < deadline - 10:
            max_struct_attempts = 30
            for attempt in range(max_struct_attempts):
                if time.time() > deadline - 10:
                    break
                
                seed = target_rank * 20000 + attempt
                U_init, V_init, W_init = _structured_init(n, m, p, target_rank, seed=seed)
                
                U_opt, V_opt, W_opt = _gradient_descent(
                    U_init, V_init, W_init, n, m, p,
                    max_iter=3000, lr=0.05, tol=1e-12,
                    deadline=deadline - 5, momentum=0.9
                )
                
                if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                    result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9)
                    if result is not None:
                        U_r, V_r, W_r = result
                        best = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = target_rank
                        found = True
                        break
        
        # Phase 3: Polish near-solutions
        if not found and time.time() < deadline - 5:
            # Try polishing the best solutions from phases 1 and 2
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                
                seed = target_rank * 30000 + attempt
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed)
                
                U_opt, V_opt, W_opt = _gradient_descent(
                    U_init, V_init, W_init, n, m, p,
                    max_iter=2000, lr=0.05, tol=1e-8,
                    deadline=deadline - 10, momentum=0.9
                )
                
                # If we're close, try polishing
                S_check = np.einsum('ijc,jkc,kic->ijk', 
                    np.array(U_opt).reshape(n, m, -1),
                    np.array(V_opt).reshape(m, p, -1),
                    np.array(W_opt).reshape(p, n, -1))
                loss_check = np.sum((S_check - 1.0) ** 2)
                
                if loss_check < 1e-4:
                    U_pol, V_pol, W_pol = _polish(
                        U_opt, V_opt, W_opt, n, m, p,
                        deadline=deadline - 3, max_iter=5000
                    )
                    
                    if _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-8):
                        result = _try_round_and_verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-9)
                        if result is not None:
                            U_r, V_r, W_r = result
                            best = {"U": U_r, "V": V_r, "W": W_r}
                            best_rank = target_rank
                            found = True
                            break
        
        if best_rank == target_rank:
            break
    
    # Return the best found
    return best
