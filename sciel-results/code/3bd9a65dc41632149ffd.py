import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s limit to be safe (150s total)
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2
        
    def is_valid_decomposition(U, V, W, T, atol=1e-6):
        """
        Checks if U, V, W (assumed to be snapped or close to snapped)
        form a valid decomposition of T.
        """
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def try_snap_and_validate(U, V, W, T):
        """
        Snaps U, V, W to half-integers and checks if the snapped version is valid.
        Returns (True, U_snap, V_snap, W_snap) if valid, else (False, None, None, None).
        """
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if the snap is close to the current values (to avoid wild jumps)
        # We allow some tolerance here because gradient descent might not be perfectly aligned
        if not (np.allclose(U, U_snap, atol=0.1) and 
                np.allclose(V, V_snap, atol=0.1) and 
                np.allclose(W, W_snap, atol=0.1)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def polish_solution(U, V, W, max_iter=100, lr_init=0.001):
        """
        High-precision polish pass. 
        """
        lr = lr_init
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            U_new = U - lr * gradU
            V_new = V - lr * gradV
            W_new = W - lr * gradW
            
            loss_new = compute_loss(U_new, V_new, W_new)
            
            if loss_new < prev_loss:
                U, V, W = U_new, V_new, W_new
                prev_loss = loss_new
                lr *= 1.01 # Slight increase if improving
            else:
                lr *= 0.9
                U, V, W = U_new, V_new, W_new # Accept anyway to explore? No, revert.
                U, V, W = U + lr * gradU, V + lr * gradV, W + lr * gradW # Revert
                if lr < 1e-8:
                    break
        
        return U, V, W

    # Schoolbook decomposition as fallback
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

    def gf2_init(U_dim, V_dim, W_dim, R, seed_val):
        """
        Initialize using GF(2) structure: start with binary-like patterns
        and add small perturbations.
        """
        rng = np.random.RandomState(seed_val)
        
        # Create base patterns inspired by GF(2) linear algebra
        U = np.zeros((U_dim, R))
        V = np.zeros((V_dim, R))
        W = np.zeros((W_dim, R))
        
        # For each rank component, assign a "signature" pattern
        for r in range(R):
            # U signature: which (i,j) pairs are active
            u_active = rng.choice(U_dim, size=min(3, U_dim), replace=False)
            for idx in u_active:
                U[idx, r] = 1.0 if rng.random() < 0.5 else -1.0
            
            # V signature: which (j,k) pairs are active  
            v_active = rng.choice(V_dim, size=min(5, V_dim), replace=False)
            for idx in v_active:
                V[idx, r] = 1.0 if rng.random() < 0.5 else -1.0
            
            # W signature: which (k,i) pairs are active
            w_active = rng.choice(W_dim, size=min(3, W_dim), replace=False)
            for idx in w_active:
                W[idx, r] = 1.0 if rng.random() < 0.5 else -1.0
        
        # Add small Gaussian noise to break symmetry
        U += rng.randn(U_dim, R) * 0.1
        V += rng.randn(V_dim, R) * 0.1
        W += rng.randn(W_dim, R) * 0.1
        
        return U, V, W

    def random_init(U_dim, V_dim, W_dim, R, seed_val):
        """Standard random initialization."""
        rng = np.random.RandomState(seed_val)
        U = rng.randn(U_dim, R) * 0.5
        V = rng.randn(V_dim, R) * 0.5
        W = rng.randn(W_dim, R) * 0.5
        return U, V, W

    def optimize(U, V, W, max_iter=300, lr_init=0.01, momentum=0.9):
        """
        Gradient descent with momentum and periodic snap checks.
        Returns (found, U_snap, V_snap, W_snap) if a valid snapped solution is found.
        """
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        lr = lr_init
        no_improve = 0
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Periodic snap check
            if it % 15 == 0:
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    return True, U_snap, V_snap, W_snap
            
            # Compute gradient
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Update with momentum
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            # Compute loss
            loss = compute_loss(U, V, W)
            
            if loss < prev_loss - 1e-8:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    break
            
            if loss < 1e-6:
                break
        
        # Final snap check after optimization
        valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
        if valid:
            return True, U_snap, V_snap, W_snap
        
        return False, None, None, None

    best_solution = None
    best_rank = n * m * p  # schoolbook rank as initial best

    # Try ranks from 45 to 49
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts: more for lower ranks as they are harder
        num_restarts = 8 if R <= 46 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            seed_val = 42 + restart * 1000 + R
            
            # Alternate between GF(2)-inspired and random initialization
            if restart % 2 == 0:
                U, V, W = gf2_init(U_dim, V_dim, W_dim, R, seed_val)
            else:
                U, V, W = random_init(U_dim, V_dim, W_dim, R, seed_val)
            
            # Run optimization
            found, U_snap, V_snap, W_snap = optimize(U, V, W)
            
            if found:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                break
        
        if found:
            break
    
    # If still not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
