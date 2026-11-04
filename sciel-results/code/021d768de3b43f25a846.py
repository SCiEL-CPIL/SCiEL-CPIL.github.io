"""Combined program: schoolbook fallback + improved integer-local-search for W."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using einsum over all indices at once."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_np.reshape(n, m, -1)
    V3 = V_np.reshape(m, p, -1)
    W3 = W_np.reshape(p, n, -1)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.all(np.abs(T - 1.0) < tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_rank_densified(rank, n, m, p, deadline, max_attempts, seed_offset=0):
    """
    Fix random U,V; solve linear system for W per (k,i) block.
    Use integer local search to find half-integer W.
    """
    rng = np.random.RandomState(42 + seed_offset)
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Generate random U,V with small integer entries
        # Using {-1, 0, 1} or {-2, -1, 0, 1, 2}
        if attempt % 2 == 0:
            U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
        else:
            U = rng.choice([-2, -1, 0, 1, 2], size=(n*m, rank)).astype(float)
            V = rng.choice([-2, -1, 0, 1, 2], size=(m*p, rank)).astype(float)
            
        W = np.zeros((p*n, rank))
        feasible = True
        for k in range(p):
            for i in range(n):
                # Build m x rank matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                
                # Solve A @ Wc = rhs (least squares)
                try:
                    Wc_real, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                
                # Check if real solution is close to integer/half-integer
                # Scale by 2 to work with integers
                Z_real = 2.0 * Wc_real
                Z_int = np.round(Z_real).astype(int)
                
                # Local search on Z_int to minimize ||A (Z_int/2) - rhs||^2
                # Start with Z_int
                best_Z = Z_int.copy()
                best_err = np.sum((A @ (best_Z / 2.0) - rhs) ** 2)
                
                # Iterative improvement
                improved = True
                max_iter = 50
                for _ in range(max_iter):
                    if time.time() > deadline:
                        return None
                    improved = False
                    # Try flipping each coordinate to nearest integers
                    for c in range(rank):
                        # Current value
                        z_c = best_Z[c]
                        # Candidates: z_c - 1, z_c, z_c + 1 (and maybe further)
                        # Actually, since we want to minimize error, we can try z_c +/- 1
                        for delta in [-1, 1]:
                            new_Z = best_Z.copy()
                            new_Z[c] = z_c + delta
                            new_err = np.sum((A @ (new_Z / 2.0) - rhs) ** 2)
                            if new_err < best_err - 1e-10:
                                best_err = new_err
                                best_Z = new_Z
                                improved = True
                    if not improved:
                        break
                
                # Check if we found an exact solution
                if best_err < 1e-7:
                    W[k*n+i, :] = best_Z / 2.0
                else:
                    # Try a few random restarts of local search if it failed
                    found = False
                    for restart in range(10):
                        if time.time() > deadline:
                            return None
                        Z_restart = np.round(Z_real + rng.randn(rank) * 0.5).astype(int)
                        best_Z_r = Z_restart.copy()
                        best_err_r = np.sum((A @ (best_Z_r / 2.0) - rhs) ** 2)
                        for _ in range(20):
                            if time.time() > deadline:
                                return None
                            imp = False
                            for c in range(rank):
                                z_c = best_Z_r[c]
                                for delta in [-1, 1]:
                                    new_Z = best_Z_r.copy()
                                    new_Z[c] = z_c + delta
                                    new_err = np.sum((A @ (new_Z / 2.0) - rhs) ** 2)
                                    if new_err < best_err_r - 1e-10:
                                        best_err_r = new_err
                                        best_Z_r = new_Z
                                        imp = True
                            if not imp:
                                break
                        if best_err_r < 1e-7:
                            W[k*n+i, :] = best_Z_r / 2.0
                            found = True
                            break
                    if not found:
                        feasible = False
                        break
        
        if feasible:
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135.0  # safety margin

    # Best-so-far: schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    U0_list = [[float(x) for x in row] for row in U0]
    V0_list = [[float(x) for x in row] for row in V0]
    W0_list = [[float(x) for x in row] for row in W0]
    best = {"U": U0_list, "V": V0_list, "W": W0_list}
    best_rank = n * m * p

    # Focus on ranks 110 down to 90. 
    # Ranks > 110 are likely not worth the time if we can't find 110 quickly.
    # Ranks < 90 are very hard.
    for rank in range(110, 89, -1):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - 5.0 - time.time()
        # Allocate attempts based on remaining time
        # Higher ranks are easier, so fewer attempts might suffice? 
        # Lower ranks are harder, need more attempts? 
        # But we run out of time. 
        # Let's use a fixed moderate number of attempts.
        attempts = max(10, int(remaining * 5))
        
        result = _try_rank_densified(rank, n, m, p, deadline - 5.0, attempts, seed_offset=rank)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break

    return best
# EVOLVE-BLOCK-END
