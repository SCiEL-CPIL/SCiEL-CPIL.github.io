import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # 1. Initial Solution
    # Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = calculate_cost(a, b, best_p)
    
    # 2. Local Search Loop
    while time.time() < deadline:
        # Run steepest descent 2-opt
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute C = b[p, :]
            C = b[p, :]
            # Compute X = a @ C.T
            X = a @ C.T
            # Compute S
            D = np.diag(X)
            S = D[:, np.newaxis] - X - X.T + D[np.newaxis, :]
            
            # Compute K (corrections)
            # K[i, j] = (a[i, i] - a[j, i]) * (b[p[i], p[i]] - b[p[j], p[i]]) + (a[i, j] - a[j, j]) * (b[p[i], p[j]] - b[p[j], p[j]])
            # Vectorized:
            # Term 1: (a[i, i] - a[j, i]) * (b[p[i], p[i]] - b[p[j], p[i]])
            # a_ii = np.diag(a)
            # a_ji = a.T (if symmetric a)
            # b_pip = np.diag(C) -> b[p[i], p[i]]
            # b_pjp = C[np.arange(n), p] -> b[p[i], p[j]]? No.
            # C[i, j] = b[p[i], j].
            # We need b[p[i], p[j]]. That is C[i, p[j]].
            # We need b[p[j], p[i]]. That is C[j, p[i]].
            # We need b[p[i], p[i]]. That is C[i, p[i]].
            # We need b[p[j], p[j]]. That is C[j, p[j]].
            
            idx = np.arange(n)
            b_pip = C[idx, p] # b[p[i], p[i]]
            b_pjp = C[idx, p] # Wait. C[i, k] = b[p[i], k]. So C[i, p[j]] = b[p[i], p[j]].
            # We want a matrix M where M[i, j] = b[p[i], p[j]].
            # This is C[idx, p].
            M = C[idx, p] # M[i, j] = b[p[i], p[j]]
            
            a_ii = np.diag(a)
            # a_ji is a.T[j, i] = a[i, j] if symmetric.
            # Let's assume symmetric for tai256c.
            # a[i, i] - a[j, i] -> a_ii[i] - a[j, i]
            # a[i, j] - a[j, j] -> a[i, j] - a_ii[j]
            
            # Term 1 matrix: (a_ii[:, None] - a.T) * (b_pip[:, None] - M.T)
            # M.T[i, j] = M[j, i] = b[p[j], p[i]].
            # We need b[p[j], p[i]] for the first part?
            # Formula: (a[i, i] - a[j, i]) * (b[p[i], p[i]] - b[p[j], p[i]])
            # b[p[j], p[i]] is M[j, i].
            # So (b_pip[i] - M[j, i]).
            # Vectorized: b_pip[:, None] - M.T
            
            T1 = (a_ii[:, None] - a.T) * (b_pip[:, None] - M.T)
            
            # Term 2: (a[i, j] - a[j, j]) * (b[p[i], p[j]] - b[p[j], p[j]])
            # b[p[i], p[j]] is M[i, j].
            # b[p[j], p[j]] is b_pip[j].
            # Vectorized: M - b_pip[None, :]
            
            T2 = (a - a_ii[None, :]) * (M - b_pip[None, :])
            
            K = T1 + T2
            
            Delta = 2.0 * (S - K)
            
            # Mask diagonal (no swap with self)
            np.fill_diagonal(Delta, np.inf)
            
            # Find best swap
            i, j = np.unravel_index(np.argmin(Delta), Delta.shape)
            
            if Delta[i, j] < 0:
                # Apply swap
                p[i], p[j] = p[j], p[i]
                best_cost += Delta[i, j] # Approx, better recalc
                # Recalc cost to be safe? Or trust delta?
                # Delta is exact for the swap.
                improved = True
                # Update best if better
                if best_cost < calculate_cost(a, b, p): # This is O(N^2), expensive.
                    best_p = p.copy()
                # Actually, just track cost.
                # But Delta calculation might have floating point errors.
                # Let's just update best_p if we accept swap, and recalc cost periodically or at end.
                # For now, assume Delta is accurate enough for comparison.
                # To be safe, recalc cost every N iterations or if time allows.
                # Given N=256, cost calc is 256^2 = 65k ops. Very fast.
                current_cost = calculate_cost(a, b, p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
        # If local optimum reached, perturb
        if time.time() < deadline:
            # Random perturbation
            for _ in range(10): # Try 10 random swaps
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                p[i], p[j] = p[j], p[i]
                current_cost = calculate_cost(a, b, p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                # Don't revert immediately, let local search fix it
            # Run local search again
            # The loop above will run again
    
    return best_p.tolist()

def calculate_cost(a, b, p):
    return np.sum(a * b[np.ix_(p, p)])
