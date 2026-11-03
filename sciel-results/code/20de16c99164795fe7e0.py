import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
        
    def is_valid_decomposition(U, V, W, T, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def try_snap_and_validate(U, V, W, T):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snap is close to current
        if not (np.allclose(U, U_snap, atol=0.1) and 
                np.allclose(V, V_snap, atol=0.1) and 
                np.allclose(W, W_snap, atol=0.1)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def adam_optimize(U, V, W, R, max_iter=300, lr_init=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        prev_loss = float('inf')
        best_loss = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            t += 1
            
            # Periodic snap check
            if it % 15 == 0:
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    return True, U_snap, V_snap, W_snap
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam update
            lr_t = lr_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU * gradU)
            U -= lr_t * mU / (np.sqrt(vU) + epsilon)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV * gradV)
            V -= lr_t * mV / (np.sqrt(vV) + epsilon)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW * gradW)
            W -= lr_t * mW / (np.sqrt(vW) + epsilon)
            
            loss = compute_loss(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-8:
                break
            
            if loss < prev_loss - 1e-8:
                prev_loss = loss
            else:
                # If stuck, break early
                if it > 50 and loss > prev_loss * 1.05:
                    break

        # Final snap check on best and current
        for U_c, V_c, W_c in [(best_U, best_V, best_W), (U, V, W)]:
            valid, U_snap, V_snap, W_snap = try_snap_and_validate(U_c, V_c, W_c, T)
            if valid:
                return True, U_snap, V_snap, W_snap
        
        return False, None, None, None

    def random_init(U_dim, V_dim, W_dim, R, seed_val):
        rng = np.random.RandomState(seed_val)
        # Initialize with values close to half-integers
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        U = rng.choice(vals, size=(U_dim, R))
        V = rng.choice(vals, size=(V_dim, R))
        W = rng.choice(vals, size=(W_dim, R))
        # Add small noise
        U += rng.randn(U_dim, R) * 0.1
        V += rng.randn(V_dim, R) * 0.1
        W += rng.randn(W_dim, R) * 0.1
        return U, V, W

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 45 to 49 (lower is better, so start from lower)
    ranks_to_try = list(range(45, 50))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts - use more restarts for lower ranks
        num_restarts = 8 if target_rank <= 46 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            seed_val = 42 + restart * 1000 + R
            U, V, W = random_init(U_dim, V_dim, W_dim, R, seed_val)
            
            found, U_snap, V_snap, W_snap = adam_optimize(U, V, W, R)
            
            if found:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()
