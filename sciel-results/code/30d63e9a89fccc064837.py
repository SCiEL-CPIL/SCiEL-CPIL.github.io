"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 100  # 20s safety margin before 120s kill

    # Schoolbook rank 36
    rank_sb = n * m * p
    U_sb = [[0.0] * rank_sb for _ in range(n * m)]
    V_sb = [[0.0] * rank_sb for _ in range(m * p)]
    W_sb = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1.0
                V_sb[j * p + k][col] = 1.0
                W_sb[k * n + i][col] = 1.0
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Try to find a lower rank using gradient-based optimization
    # The matrix multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # We want to express T as sum of r rank-1 tensors
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]

    def tensor_to_3d(U, V, W, r):
        """Compute the 3D tensor from factor matrices."""
        T = [[0.0] * (p * n) for _ in range(m * p)]
        for a in range(n * m):
            for b in range(m * p):
                row = T[b]
                Ua = U[a]
                for t in range(r):
                    ua = Ua[t]
                    if ua != 0.0:
                        vb = V[b][t]
                        if vb != 0.0:
                            coeff = ua * vb
                            Wt = W
                            for c in range(p * n):
                                row[c] += coeff * Wt[c][t]
        return T

    def compute_error(U, V, W, r):
        """Compute Frobenius norm squared of reconstruction error."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for t in range(r):
                        val += U[a][t] * V[b][t] * W[c][t]
                    diff = val - 1.0
                    err += diff * diff
        return err

    def try_rank(r):
        """Try to find a decomposition of rank r using gradient descent."""
        if time.time() > deadline - 30:
            return None

        import random
        best_err = float('inf')
        best_solution = None

        n_vars = r * (n * m + m * p + p * n)
        
        # Multiple random restarts
        max_restarts = 20
        max_iter = 500
        lr = 0.01

        for restart in range(max_restarts):
            if time.time() > deadline - 30:
                break

            # Initialize with small random values
            U = [[random.uniform(-0.5, 0.5) for _ in range(r)] for _ in range(n * m)]
            V = [[random.uniform(-0.5, 0.5) for _ in range(r)] for _ in range(m * p)]
            W = [[random.uniform(-0.5, 0.5) for _ in range(r)] for _ in range(p * n)]

            for it in range(max_iter):
                if time.time() > deadline - 30:
                    break

                # Compute gradient of error w.r.t. each variable
                # err = sum_{i,j,k} (sum_t U[a,t]*V[b,t]*W[c,t] - 1)^2
                # d_err/d_U[a,t] = 2 * sum_{i,j,k where a=i*m+j} (val-1) * V[b,t] * W[c,t]
                # d_err/d_V[b,t] = 2 * sum_{i,j,k where b=j*p+k} (val-1) * U[a,t] * W[c,t]
                # d_err/d_W[c,t] = 2 * sum_{i,j,k where c=k*n+i} (val-1) * U[a,t] * V[b,t]

                # Build index mappings
                # For each (i,j,k), we have a=i*m+j, b=j*p+k, c=k*n+i
                grads_U = [[0.0] * r for _ in range(n * m)]
                grads_V = [[0.0] * r for _ in range(m * p)]
                grads_W = [[0.0] * r for _ in range(p * n)]

                # First compute all (val-1) terms
                vals = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = 0.0
                            for t in range(r):
                                val += U[a][t] * V[b][t] * W[c][t]
                            vals[(i, j, k)] = val - 1.0

                # Compute gradients
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            diff = vals[(i, j, k)]
                            for t in range(r):
                                grads_U[a][t] += 2.0 * diff * V[b][t] * W[c][t]
                                grads_V[b][t] += 2.0 * diff * U[a][t] * W[c][t]
                                grads_W[c][t] += 2.0 * diff * U[a][t] * V[b][t]

                # Gradient descent step
                for a in range(n * m):
                    for t in range(r):
                        U[a][t] -= lr * grads_U[a][t]
                for b in range(m * p):
                    for t in range(r):
                        V[b][t] -= lr * grads_V[b][t]
                for c in range(p * n):
                    for t in range(r):
                        W[c][t] -= lr * grads_W[c][t]

                # Check convergence
                err = compute_error(U, V, W, r)
                if err < best_err:
                    best_err = err
                    best_solution = (
                        [row[:] for row in U],
                        [row[:] for row in V],
                        [row[:] for row in W]
                    )
                if err < 1e-6:
                    break

            if best_err < 1e-6:
                break

        if best_err < 1e-4:
            # Try to snap to integer/half-integer values
            U_sol, V_sol, W_sol = best_solution
            def snap(val):
                # Snap to nearest half-integer
                return round(val * 2) / 2.0

            U_snapped = [[snap(x) for x in row] for row in U_sol]
            V_snapped = [[snap(x) for x in row] for row in V_sol]
            W_snapped = [[snap(x) for x in row] for row in W_sol]

            # Check if snapped solution is still valid
            err_snapped = compute_error(U_snapped, V_snapped, W_snapped, r)
            if err_snapped < 1e-4:
                return (U_snapped, V_snapped, W_snapped)
            else:
                # Use the unsnapped solution if it's close enough
                return (U_sol, V_sol, W_sol)

        return None

    # Try ranks from 22 up to 35
    for r in range(22, rank_sb):
        if time.time() > deadline - 20:
            break
        result = try_rank(r)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            best_rank = r
            break  # Found a valid lower rank, stop searching

    # Convert to lists (they should already be lists)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
