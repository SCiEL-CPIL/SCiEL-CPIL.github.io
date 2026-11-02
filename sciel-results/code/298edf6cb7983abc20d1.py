# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s total)
    
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

    def validate(U, V, W, atol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def discrete_refine(U_snap, V_snap, W_snap, max_iter=100):
        current_error = np.sum((np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True) - T) ** 2)
        U_curr = U_snap.copy()
        V_curr = V_snap.copy()
        W_curr = W_snap.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if current_error < 1e-9:
                break
                
            mat_choice = random.choice([0, 1, 2])
            if mat_choice == 0:
                r, c = random.randint(0, U_curr.shape[0]-1), random.randint(0, U_curr.shape[1]-1)
                old_val = U_curr[r, c]
                # Try +0.5, -0.5
                delta = random.choice([0.5, -0.5])
                U_curr[r, c] = old_val + delta
            elif mat_choice == 1:
                r, c = random.randint(0, V_curr.shape[0]-1), random.randint(0, V_curr.shape[1]-1)
                old_val = V_curr[r, c]
                delta = random.choice([0.5, -0.5])
                V_curr[r, c] = old_val + delta
            else:
                r, c = random.randint(0, W_curr.shape[0]-1), random.randint(0, W_curr.shape[1]-1)
                old_val = W_curr[r, c]
                delta = random.choice([0.5, -0.5])
                W_curr[r, c] = old_val + delta
                
            prod = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr, optimize=True)
            new_error = np.sum((prod - T) ** 2)
            
            if new_error < current_error:
                current_error = new_error
            else:
                if mat_choice == 0:
                    U_curr[r, c] = old_val
                elif mat_choice == 1:
                    V_curr[r, c] = old_val
                else:
                    W_curr[r, c] = old_val
                    
        return U_curr, V_curr, W_curr, current_error

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

    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        R = target_rank
        found = False
        
        # More restarts for lower ranks
        num_restarts = 10 if R == 45 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(42 + restart * 1000 + R)
            np.random.seed(42 + restart * 1000 + R)
            
            # Initialization Strategy 1: Sparse Random
            U = np.zeros((U_dim, R))
            V = np.zeros((V_dim, R))
            W = np.zeros((W_dim, R))
            
            for r in range(R):
                # Pick a few random non-zeros
                for _ in range(random.randint(1, 3)):
                    idx_u = random.randint(0, U_dim - 1)
                    U[idx_u, r] = random.choice([-1.0, 1.0])
                    idx_v = random.randint(0, V_dim - 1)
                    V[idx_v, r] = random.choice([-1.0, 1.0])
                    idx_w = random.randint(0, W_dim - 1)
                    W[idx_w, r] = random.choice([-1.0, 1.0])
            
            # Initialization Strategy 2: Gaussian
            U2 = np.random.normal(0, 1, (U_dim, R))
            V2 = np.random.normal(0, 1, (V_dim, R))
            W2 = np.random.normal(0, 1, (W_dim, R))
            
            # Choose better init
            loss1 = compute_loss(U, V, W)
            loss2 = compute_loss(U2, V2, W2)
            
            if loss2 < loss1:
                U, V, W = U2, V2, W2
            
            # Gradient Descent
            lr = 0.05 # Higher LR for sparse init?
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 200
            patience = 50
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                loss = compute_loss(U, V, W)
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                if loss < 1e-8:
                    break
            
            # Snap and Refine
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            
            # Discrete Refinement
            U_ref, V_ref, W_ref, ref_error = discrete_refine(U_snap, V_snap, W_snap)
            
            if ref_error < 1e-6:
                best_solution = (U_ref, V_ref, W_ref)
                best_rank = R
                found = True
                break
            
            # If refinement didn't solve it, check if original snap was better (unlikely but possible)
            valid, U_v, V_v, W_v = validate(U, V, W)
            if valid:
                best_solution = (U_v, V_v, W_v)
                best_rank = R
                found = True
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

# EVOLVE-BLOCK-END
