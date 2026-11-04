"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import numpy as np
from itertools import product

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)

    def make_schoolbook():
        rank = nmp
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.all(T == 1)

    best_rank = nmp
    best_U, best_V, best_W = make_schoolbook()

    # Try a few random searches for lower rank using a smarter initialization
    # We use a structured approach: for each output element (i,k), we need to solve
    # for W[k*n+i, :] such that sum_c U[i,j,c]*V[j,k,c]*W[k*n+i,c] = 1 for all j

    # Strategy: Try ranks from 83 down to some lower bound
    # For each rank, try random U, V and solve for W

    def try_solve_for_W(U, V, target_rank):
        """Given U and V of shape (n*m, r) and (m*p, r), solve for W of shape (p*n, r)."""
        W = np.zeros((p * n, target_rank), dtype=np.int64)
        for k in range(p):
            for i in range(n):
                # Build system: for each j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((m, target_rank), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                
                # Solve A @ w = ones(m)
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    return None
                
                # Check if solution is close to integer
                sol_int = np.round(sol).astype(np.int64)
                
                if np.allclose(A @ sol_int, np.ones(m), atol=1e-6):
                    W[k * n + i, :] = sol_int
                else:
                    # Try local search around sol_int
                    found = False
                    for radius in [1, 2, 3]:
                        if found:
                            break
                        # For small target_rank, do exhaustive search
                        if target_rank <= 6:
                            for combo in product(range(-radius, radius+1), repeat=target_rank):
                                test = sol_int + np.array(combo, dtype=np.int64)
                                if np.allclose(A @ test, np.ones(m), atol=1e-6):
                                    W[k * n + i, :] = test
                                    found = True
                                    break
                        else:
                            # Coordinate descent
                            test = sol_int.copy()
                            for _ in range(20):
                                improved = False
                                for c in range(target_rank):
                                    for delta in range(-radius, radius+1):
                                        if delta == 0:
                                            continue
                                        old = test[c]
                                        test[c] = old + delta
                                        if np.allclose(A @ test, np.ones(m), atol=1e-6):
                                            found = True
                                            improved = True
                                            break
                                        test[c] = old
                                    if found:
                                        break
                                if found:
                                    break
                    if not found:
                        return None
        return W

    # Try random searches for lower ranks
    np.random.seed(42)
    
    for target_rank in range(nmp - 1, 70, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random attempts
        max_attempts = 5 if target_rank > 75 else 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            U = np.random.randint(-2, 3, size=(n * m, target_rank)).astype(np.int64)
            V = np.random.randint(-2, 3, size=(m * p, target_rank)).astype(np.int64)
            
            W = try_solve_for_W(U, V, target_rank)
            if W is not None:
                if verify(U, V, W):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break  # Move to next (lower) rank
        
        if time.time() > deadline - 10:
            break

    # Final verification
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = make_schoolbook()
        best_rank = nmp

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
