import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Precompute support indices
SUPPORT = []
A_TO_BC = {a: [] for a in range(NM)}
B_TO_AC = {b: [] for b in range(MP)}
C_TO_AB = {c: [] for c in range(PN)}

for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            idx = len(SUPPORT)
            SUPPORT.append((a, b, c))
            A_TO_BC[a].append((b, c, idx))
            B_TO_AC[b].append((a, c, idx))
            C_TO_AB[c].append((a, b, idx))

def get_trivial_solution():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# Structured Strassen-style initialization for rank 33 (2x2 blocks with Strassen).
# 2x3x6 = split A into two 2x3 blocks (i=0,1), B into two 3x3 blocks (j pairs), C into two 2x3 blocks.
# Actually split by the i dimension (2) and k dimension (6 into two 3s).
# A is 2x3, B is 3x6, C is 2x6.
# Split A by rows (2 blocks of 1x3), B by cols (2 blocks of 3x3), C by cols (2 blocks of 2x3).
# Then C[i_block, k_block] = A[i_block, :] * B[:, k_block].
# Each block multiply is 1x3 * 3x3 = 1x3, which is just a dot product (rank 3).
# That gives 4 block multiplies * 3 = 12. Not 33.
# Let me reconsider. The Strassen approach for 2x2 blocks:
# Split A into 2x2 blocks? A is 2x3, can't split into 2x2 easily.
# Alternative: the 1.091 score came from rank 33. Let me just use a random structured init.
# For now, use random init with small integers, same as current.

def _structured_init(R, seed=0):
    """Initialize with a mix of structured (schoolbook-like) and random entries."""
    rng = np.random.RandomState(seed)
    U = np.zeros((NM, R), dtype=np.float64)
    V = np.zeros((MP, R), dtype=np.float64)
    W = np.zeros((PN, R), dtype=np.float64)
    
    if R >= 36:
        # Schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W
    else:
        # Random small integers
        U = rng.randint(-2, 3, size=(NM, R)).astype(np.float64)
        V = rng.randint(-2, 3, size=(MP, R)).astype(np.float64)
        W = rng.randint(-2, 3, size=(PN, R)).astype(np.float64)
        return U, V, W

def _compute_P(U, V, W):
    P = np.zeros(len(SUPPORT), dtype=np.float64)
    for idx, (a, b, c) in enumerate(SUPPORT):
        P[idx] = np.dot(U[a], V[b] * W[c])
    return P

def _err(P):
    return np.sum((1.0 - P) ** 2)

def _local_search(U, V, W, R, time_budget, max_iters=50000, max_stall=2000):
    """Fast local search with delta updates."""
    A_TO_BC_l = [A_TO_BC[a] for a in range(NM)]
    B_TO_AC_l = [B_TO_AC[b] for b in range(MP)]
    C_TO_AB_l = [C_TO_AB[c] for c in range(PN)]
    
    P = _compute_P(U, V, W)
    err = _err(P)
    
    improved = True
    stall_count = 0
    iter_count = 0
    
    while improved and stall_count < max_stall and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            mat, dim, name = U, NM, 'U'
        elif mat_choice == 1:
            mat, dim, name = V, MP, 'V'
        else:
            mat, dim, name = W, PN, 'W'
        
        row = np.random.randint(0, dim)
        col = np.random.randint(0, R)
        old_val = mat[row, col]
        
        for delta in [1, -1]:
            new_val = old_val + delta
            delta_err = 0.0
            
            if name == 'U':
                for (b, c, idx) in A_TO_BC_l[row]:
                    D = delta * V[b, col] * W[c, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            elif name == 'V':
                for (a, c, idx) in B_TO_AC_l[row]:
                    D = delta * U[a, col] * W[c, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            else:
                for (a, b, idx) in C_TO_AB_l[row]:
                    D = delta * U[a, col] * V[b, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            
            if delta_err < -1e-10:
                mat[row, col] = new_val
                err += delta_err
                if name == 'U':
                    for (b, c, idx) in A_TO_BC_l[row]:
                        P[idx] += delta * V[b, col] * W[c, col]
                elif name == 'V':
                    for (a, c, idx) in B_TO_AC_l[row]:
                        P[idx] += delta * U[a, col] * W[c, col]
                else:
                    for (a, b, idx) in C_TO_AB_l[row]:
                        P[idx] += delta * U[a, col] * V[b, col]
                improved = True
                stall_count = 0
                break
        
        if not improved:
            stall_count += 1
        
        if err < 1e-9:
            break
        if time.time() > time_budget:
            break
    
    return U, V, W, err

def _polish(U, V, W, R, time_budget):
    """High-precision polish: try small perturbations and accept only if error strictly decreases.
    Use a tighter tolerance and more iterations."""
    A_TO_BC_l = [A_TO_BC[a] for a in range(NM)]
    B_TO_AC_l = [B_TO_AC[b] for b in range(MP)]
    C_TO_AB_l = [C_TO_AB[c] for c in range(PN)]
    
    P = _compute_P(U, V, W)
    err = _err(P)
    
    # Try to find exact solution by checking if all P entries are close to 1
    # and then rounding to nearest half-integer
    if err < 1e-4:
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        P_r = _compute_P(U_r, V_r, W_r)
        err_r = _err(P_r)
        if err_r < 1e-6:
            return U_r, V_r, W_r, err_r
    
    # If not exact, do a few more local search steps with smaller steps
    for _ in range(200):
        if time.time() > time_budget:
            break
        mat_choice = np.random.randint(0, 3)
        if mat_choice == 0:
            mat, dim, name = U, NM, 'U'
        elif mat_choice == 1:
            mat, dim, name = V, MP, 'V'
        else:
            mat, dim, name = W, PN, 'W'
        row = np.random.randint(0, dim)
        col = np.random.randint(0, R)
        old_val = mat[row, col]
        
        for delta in [0.5, -0.5, 1, -1]:
            new_val = old_val + delta
            delta_err = 0.0
            if name == 'U':
                for (b, c, idx) in A_TO_BC_l[row]:
                    D = delta * V[b, col] * W[c, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            elif name == 'V':
                for (a, c, idx) in B_TO_AC_l[row]:
                    D = delta * U[a, col] * W[c, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            else:
                for (a, b, idx) in C_TO_AB_l[row]:
                    D = delta * U[a, col] * V[b, col]
                    P_old = P[idx]
                    delta_err += D * D - 2 * D * (1 - P_old)
            
            if delta_err < -1e-12:
                mat[row, col] = new_val
                err += delta_err
                if name == 'U':
                    for (b, c, idx) in A_TO_BC_l[row]:
                        P[idx] += delta * V[b, col] * W[c, col]
                elif name == 'V':
                    for (a, c, idx) in B_TO_AC_l[row]:
                        P[idx] += delta * U[a, col] * W[c, col]
                else:
                    for (a, b, idx) in C_TO_AB_l[row]:
                        P[idx] += delta * U[a, col] * V[b, col]
                break
    
    return U, V, W, err

def search_rank(R, time_budget):
    best_err = np.inf
    best_U = best_V = best_W = None
    
    num_restarts = 0
    while time.time() < time_budget:
        num_restarts += 1
        seed = num_restarts * 7919
        U, V, W = _structured_init(R, seed)
        
        # Local search
        U, V, W, err = _local_search(U, V, W, R, time_budget)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
        
        # Polish if close
        if best_err < 1e-2 and time.time() < time_budget:
            U, V, W, err = _polish(best_U, best_V, best_W, R, time_budget)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-9:
                    break
    
    if best_err < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Try ranks from low to high, prioritizing lower ranks
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    for R in ranks_to_try:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 5:
            break
        
        # Allocate time: more time for lower ranks
        if R <= 20:
            budget = min(remaining - 5, 20)
        elif R <= 30:
            budget = min(remaining - 5, 10)
        else:
            budget = min(remaining - 5, 5)
        
        if budget < 1:
            continue
        
        result = search_rank(R, time.time() + budget)
        if result is not None:
            return result
    
    return get_trivial_solution()
# EVOLVE-BLOCK-END
