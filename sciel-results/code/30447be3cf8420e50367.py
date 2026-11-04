"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares, minimize

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute index triples and their corresponding row indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_idx_list = np.array([i * m + j for (i, j, k) in indices])
    v_idx_list = np.array([j * p + k for (i, j, k) in indices])
    w_idx_list = np.array([k * n + i for (i, j, k) in indices])
    num_triples = len(indices)

    def compute_residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        U_rows = U[u_idx_list, :]  # (num_triples, rank)
        V_rows = V[v_idx_list, :]  # (num_triples, rank)
        W_rows = W[w_idx_list, :]  # (num_triples, rank)

        # Compute the product sum for each triple
        products = U_rows * V_rows * W_rows  # (num_triples, rank)
        res = np.sum(products, axis=1) - 1.0
        return res

    def residual(params, rank):
        """Vectorized residual computation for the tensor decomposition."""
        U_flat = params[:nm * rank].reshape(nm, rank)
        V_flat = params[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W_flat = params[nm * rank + mp * rank:].reshape(pn, rank)
        return compute_residual(U_flat, V_flat, W_flat)

    def check_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def to_list(arr):
        """Convert numpy array to nested list with half-integer rounding."""
        lst = []
        for row in arr:
            row_lst = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_lst.append(rounded)
                else:
                    row_lst.append(float(val))
            lst.append(row_lst)
        return lst

    def polish_solution(U, V, W, rank):
        """High-precision polish pass using tighter tolerance."""
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(
                lambda x: np.sum(np.square(residual(x, rank))),
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 1000}
            )
            if result.success and result.fun < 1e-12:
                U_sol = result.x[:nm * rank].reshape(nm, rank)
                V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                if check_decomposition(U_sol, V_sol, W_sol, rank, tol=1e-10):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    def build_rank20_decomp():
        """
        Construct a known rank-20 decomposition using:
        - Rank 9 for 2x3 x 3x3 (Strassen-like or explicit)
        - Rank 6 for 2x3 x 3x1 (trivial)
        Total: 15 for the 3x4 part split, but we use a different approach.
        
        Actually, let's use a constructive approach:
        For 2x3 x 3x4, we can use the fact that rank(2x3x3) <= 9 and rank(2x3x1) <= 6.
        But a better known bound: rank(2x3x3) = 9 (tight), so total <= 9+6=15? No.
        
        Let me use a simpler constructive approach:
        Split the 4 columns into groups and use known decompositions.
        
        Actually, the simplest known construction:
        Use the identity that C = A * B where A is 2x3, B is 3x4.
        We can write B = [B1, B2] where B1, B2 are 3x2.
        Then C = [A*B1, A*B2].
        If rank(2x3x2) = r, then total rank <= 2r.
        
        Known: rank(2x2x2) = 7 (Strassen), so rank(2x3x2) <= ?
        Actually, for 2x3x2: we can use 2 Strassen-like blocks.
        
        Let me just use a numerical approach starting from a good initial guess.
        """
        pass

    def build_naive_decomp(rank):
        """Build a trivial rank-n*m*p decomposition."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for idx, (i, j, k) in enumerate(indices):
            if idx < rank:
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
        return U, V, W

    def try_rank_optimization(rank, U0, V0, W0, max_time=10.0):
        """Try to optimize a decomposition of given rank."""
        t0 = time.time()
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=1000)
            
            if result.cost < 1e-10 and time.time() - t0 < max_time:
                U_sol = result.x[:nm * rank].reshape(nm, rank)
                V_sol = result.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
                W_sol = result.x[nm * rank + mp * rank:].reshape(pn, rank)
                if check_decomposition(U_sol, V_sol, W_sol, rank):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
        
        return None

    # Strategy 1: Try to find a low-rank decomposition using numerical optimization
    # Start with a structured initial guess based on known algorithms
    
    best_result = None
    
    # Try ranks from 15 to 24
    for rank in range(15, 25):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        max_attempts = max(1, int(remaining / 8))
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Structured initialization based on block decomposition
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                
                # Fill with a pattern that might be close to optimal
                np.random.seed(rank * 100 + attempt)
                for r in range(rank):
                    # Assign each rank component to cover certain (i,j,k) triples
                    # Use a round-robin assignment
                    for idx, (i, j, k) in enumerate(indices):
                        if idx % rank == r:
                            U[i * m + j, r] += 1.0
                            V[j * p + k, r] += 1.0
                            W[k * n + i, r] += 1.0
                
                # Add some randomness to break symmetry
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
                
            elif attempt == 1:
                # Random initialization with moderate scale
                np.random.seed(rank * 1000 + attempt * 13)
                scale = 1.0 / (rank ** 0.4)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
                
            elif attempt == 2:
                # Sparse initialization
                np.random.seed(rank * 2000 + attempt * 17)
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    # Each rank component covers a random subset
                    u_rows = np.random.choice(nm, size=max(2, nm // 2), replace=False)
                    v_rows = np.random.choice(mp, size=max(2, mp // 2), replace=False)
                    w_rows = np.random.choice(pn, size=max(2, pn // 2), replace=False)
                    U[u_rows, r] = np.random.randn(len(u_rows))
                    V[v_rows, r] = np.random.randn(len(v_rows))
                    W[w_rows, r] = np.random.randn(len(w_rows))
                    
            else:
                # Pure random
                np.random.seed(rank * 3000 + attempt * 23)
                scale = 1.0 / np.sqrt(rank)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            
            result = try_rank_optimization(rank, U, V, W, max_time=6.0)
            if result is not None:
                U_sol, V_sol, W_sol = result
                polished = polish_solution(U_sol, V_sol, W_sol, rank)
                if polished is not None:
                    U_sol, V_sol, W_sol = polished
                best_result = (rank, U_sol, V_sol, W_sol)
                break  # Found a valid decomposition at this rank
        
        if best_result is not None:
            break
    
    if best_result is not None:
        rank, U_sol, V_sol, W_sol = best_result
        return {"U": to_list(U_sol), "V": to_list(V_sol), "W": to_list(W_sol)}
    
    # Fallback: naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m * p) + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
