"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))

    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Function to check validity and snap to half-integers
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2

        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)

        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None

        # Convert back to lists
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]

        return {"U": U_out, "V": V_out, "W": W_out}

    # Kronecker product structure approach
    # 2x5 * 5x5 = (2x1) ⊗ (1x5) * (5x5)
    # For the 2x1 * 1x1 part: rank 2 (trivial, just two scalars)
    # For the 1x5 * 5x5 part: this is row vector times matrix, rank 5
    # Total rank via Kronecker: 2 * 5 = 10
    
    # Let's construct the Kronecker decomposition explicitly
    # The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # With n=2, m=5, p=5
    
    # Structure: A is 2x5, B is 5x5, C = A*B is 2x5
    # A[i,j] * B[j,k] -> C[i,k]
    # 
    # Kronecker view: A = a ⊗ I_5 where a is 2x1 column vector, I_5 is... 
    # Actually, let's think differently.
    # 
    # The tensor T[i,j,k,l] where i∈{0,1}, j∈{0..4}, k∈{0..4}, l∈{0..4}
    # T[i,j,k,l] = 1 if the indices form a valid multiplication term
    # 
    # For rank-1 term r: U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+l]
    # 
    # Kronecker approach: decompose into (2x1) ⊗ (1x5) structure
    # The 2x1 matrix multiplication with 1x5 gives rank 2*5=10
    # 
    # Let me construct a rank-10 decomposition using the product structure
    # 
    # For each "Kronecker factor" combination:
    # Factor 1: 2x1 * 1x1 (scalar) - contributes 2 terms
    # Factor 2: 1x5 * 5x5 - this is outer product of row vector with matrix
    # 
    # Actually, let me think about this more carefully.
    # 
    # The standard decomposition has rank n*m*p = 2*5*5 = 50
    # 
    # For the Kronecker product structure:
    # If we write A (2x5) as having rows that are linear combinations,
    # and B (5x5) similarly...
    #
    # Let me try a direct construction:
    # For rank r, we need U (10 x r), V (25 x r), W (10 x r)
    # 
    # Key insight: The multiplication (2x5)*(5x5) can be seen as:
    # Row i of C = Row i of A * B
    # Row i of A is a 1x5 vector, B is 5x5
    # So each row computation is (1x5)*(5x5) -> (1x5)
    # The rank for (1x5)*(5x5) is 5 (just the 5 columns of B weighted)
    # For 2 rows, if independent: rank 2*5 = 10
    #
    # Construction:
    # For i in {0,1}, k in {0..4}:
    #   Term r = i*5 + k
    #   U[0*5+j, r] = 1 for all j (row i of A, all columns)
    #   U[1*5+j, r] = 0 for all j (if i=0) or 1 (if i=1)
    #   V[j*5+k, r] = 1 if j matches... 
    #
    # Wait, let me be more careful with indices.
    # U has shape (n*m, r) = (10, r)
    # V has shape (m*p, r) = (25, r)  
    # W has shape (p*n, r) = (10, r)
    #
    # U index: i*m + j = i*5 + j
    # V index: j*p + k = j*5 + k
    # W index: k*n + i = k*2 + i
    #
    # For term r = i*5 + k (i∈{0,1}, k∈{0..4}):
    # We want: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1 for all i,j,k
    #
    # Let's set:
    # U[i*5+j, r'] = 1 if r' = i*5 + k' for some k', else 0
    #   More specifically: U[i*5+j, i*5+k'] = 1 for all j, k'
    #   So U[i*5+j, r] = 1 if floor(r/5) == i, else 0
    #
    # V[j*5+k, r] = ?
    #   We need V[j*5+k, i*5+k'] to be 1 when k=k', else 0
    #   V[j*5+k, i*5+k'] = 1 if k == k', else 0
    #   So V[j*5+k, r] = 1 if k == r%5, else 0
    #
    # W[k*2+i, r] = ?
    #   We need W[k*2+i, i*5+k'] = 1 when k==k', else 0
    #   W[k*2+i, r] = 1 if k == r%5 and i == floor(r/5), else 0
    #
    # Let's verify: For fixed i,j,k:
    # sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r]
    # = sum_{r: floor(r/5)==i} 1 * V[j*5+k, r] * W[k*2+i, r]
    # = sum_{k'=0}^{4} V[j*5+k, i*5+k'] * W[k*2+i, i*5+k']
    # = sum_{k'=0}^{4} (1 if k==k' else 0) * (1 if k==k' and i==floor((i*5+k')/5) else 0)
    # = sum_{k'=0}^{4} (1 if k==k' else 0) * (1 if k==k' else 0)  [since floor((i*5+k')/5)=i always]
    # = 1 (when k'=k)
    # = 1 ✓
    #
    # This gives rank 10!
    #
    # But wait - the entries are all 0 or 1, which are integers. 
    # Let me verify this is correct and then check if we can do better.

    def try_kronecker_rank10():
        r = 10
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for i in range(n):
            for k in range(p):
                r_idx = i * p + k  # 0..9 for n=2, p=5
                # U[i*m+j, r_idx] = 1 for all j
                for j in range(m):
                    U[i * m + j, r_idx] = 1.0
                # V[j*p+k, r_idx] = 1 for all j (k is fixed as r_idx % p)
                for j in range(m):
                    V[j * p + k, r_idx] = 1.0
                # W[k*n+i, r_idx] = 1
                W[k * n + i, r_idx] = 1.0
        
        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_recon, T_target, atol=1e-9):
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
        return None

    # Try the Kronecker rank-10 construction
    kronecker_result = try_kronecker_rank10()
    if kronecker_result:
        return kronecker_result

    # Optimization function with better initialization and more robust search
    def try_rank(r, seed=42):
        if time.time() > deadline:
            return None

        np.random.seed(seed)

        # Multiple initialization strategies
        inits = []

        # Strategy 1: Random initialization
        U1 = np.random.randn(n * m, r) * 0.5
        V1 = np.random.randn(m * p, r) * 0.5
        W1 = np.random.randn(p * n, r) * 0.5
        inits.append((U1, V1, W1))

        # Strategy 2: Sparse initialization with some structure
        U2 = np.zeros((n * m, r))
        V2 = np.zeros((m * p, r))
        W2 = np.zeros((p * n, r))
        for col in range(r):
            # Randomly assign non-zero entries
            for _ in range(3):
                idx = np.random.randint(0, n * m)
                U2[idx, col] = np.random.choice([-1, 1])
            for _ in range(3):
                idx = np.random.randint(0, m * p)
                V2[idx, col] = np.random.choice([-1, 1])
            for _ in range(3):
                idx = np.random.randint(0, p * n)
                W2[idx, col] = np.random.choice([-1, 1])
        inits.append((U2, V2, W2))

        # Strategy 3: Block-structured initialization inspired by Strassen
        U3 = np.zeros((n * m, r))
        V3 = np.zeros((m * p, r))
        W3 = np.zeros((p * n, r))
        for col in range(min(r, n * m * m * p // 3)):
            i = col % n
            j = (col // n) % m
            k = (col // (n * m)) % p
            U3[i * m + j, col] = 1.0
            V3[j * p + k, col] = 1.0
            W3[k * n + i, col] = 1.0
        U3 += np.random.randn(n * m, r) * 0.1
        V3 += np.random.randn(m * p, r) * 0.1
        W3 += np.random.randn(p * n, r) * 0.1
        inits.append((U3, V3, W3))

        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]

            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)

            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err

        best_result = None
        best_loss = np.inf

        for init_idx, (U_init, V_init, W_init) in enumerate(inits):
            if time.time() > deadline:
                break

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                res = minimize(loss, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12})
            except Exception:
                continue

            if res.fun < best_loss:
                best_loss = res.fun
                if res.fun < 1e-6:
                    size_U = (n * m) * r
                    size_V = (m * p) * r
                    U_sol = res.x[:size_U].reshape((n * m), r)
                    V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                    W_sol = res.x[size_U + size_V:].reshape((p * n), r)

                    result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                    if result:
                        return result
                    best_result = (U_sol, V_sol, W_sol)

        if best_result is not None and best_loss < 1e-3:
            U_sol, V_sol, W_sol = best_result
            x0 = np.concatenate([U_sol.flatten(), V_sol.flatten(), W_sol.flatten()])
            try:
                res = minimize(loss, x0, method='L-BFGS-B',
                               options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14})
                if res.fun < 1e-8:
                    size_U = (n * m) * r
                    size_V = (m * p) * r
                    U_sol = res.x[:size_U].reshape((n * m), r)
                    V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                    W_sol = res.x[size_U + size_V:].reshape((p * n), r)

                    result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                    if result:
                        return result
            except Exception:
                pass

        return None

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find lower rank
    # Try ranks from 49 down to 30 with multiple seeds
    ranks_to_try = list(range(49, 29, -1))

    seeds = [42, 123, 999, 111, 222, 333, 444, 555, 666, 777, 888, 1000, 2000, 3000, 4000]

    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        for seed in seeds:
            if time.time() > deadline:
                break
            result = try_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
