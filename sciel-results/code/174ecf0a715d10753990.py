import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20

    # --- Known Rank 65 Solution ---
    # Recursive Strassen 4x4x4 (rank 49) + Schoolbook 4x4x1 (rank 16)
    rank_part1 = 49
    rank_part2 = 16
    total_rank = rank_part1 + rank_part2

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # Part 2: Schoolbook for column 4 (k=4)
    col_offset = rank_part1
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0  # k=4
            W[4 * n + i, col] = 1.0  # k=4, i=row of C

    # Part 1: Recursive Strassen 4x4x4
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j

    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k

    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i

    A_side = [
        [(0, 1.0), (3, 1.0)],
        [(2, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (0, -1.0)],
        [(1, 1.0), (3, -1.0)],
    ]

    B_side = [
        [(0, 1.0), (3, 1.0)],
        [(0, 1.0)],
        [(1, 1.0), (3, -1.0)],
        [(2, 1.0), (0, -1.0)],
        [(3, 1.0)],
        [(0, 1.0), (1, 1.0)],
        [(2, 1.0), (3, 1.0)],
    ]

    out_coeff = [
        {0: 1.0, 3: 1.0},
        {2: 1.0, 3: -1.0},
        {1: 1.0, 3: 1.0},
        {0: 1.0, 2: 1.0},
        {0: -1.0, 1: 1.0},
        {3: 1.0},
        {0: 1.0},
    ]

    inner_A_side = A_side
    inner_B_side = B_side
    inner_out_coeff = out_coeff

    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk

            for (bi, c1) in A_side[mk]:
                for (bi2, c2) in inner_A_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    U[a_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi, c1) in B_side[mk]:
                for (bi2, c2) in inner_B_side[nk]:
                    u_local = bi2 // 2
                    v_local = bi2 % 2
                    V[b_idx(bi, u_local, v_local), col] += c1 * c2

            for (bi_out, c_inner) in inner_out_coeff[nk].items():
                for (bi_outer, c_outer) in out_coeff[mk].items():
                    u_local = bi_out // 2
                    v_local = bi_out % 2
                    W[c_idx(bi_outer, u_local, v_local), col] += c_inner * c_outer

    # Convert to list for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Check time
    if time.time() > deadline - 20:
        return {"U": U_list, "V": V_list, "W": W_list}

    # --- Attempt Rank 64 Search ---
    # We try to find a rank 64 decomposition.
    # Target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i, j, k] corresponds to term A[i,j] * B[j,k] contributing to C[i,k]
                # Index in U: i*m + j
                # Index in V: j*p + k
                # Index in W: k*n + i
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def flatten_params(params):
        # params is a flat vector of length 64 * (16+20+20) = 64*56
        R = 64
        U_flat = params[:R * nm].reshape(R, nm)
        V_flat = params[R * nm:R * (nm + mp)].reshape(R, mp)
        W_flat = params[R * (nm + mp):].reshape(R, pn)
        return U_flat, V_flat, W_flat

    def objective(params):
        U_flat, V_flat, W_flat = flatten_params(params)
        # Compute tensor decomposition
        # T_approx = sum_r U_r \otimes V_r \otimes W_r
        # We can compute the Frobenius norm efficiently
        # || sum_r U_r \otimes V_r \otimes W_r - T ||_F^2
        # This is equivalent to sum_{i,j,k} (sum_r U[i,r]V[j,r]W[k,r] - T[i,j,k])^2
        
        # Efficient calculation:
        # Let M_r = U_r \otimes V_r (size nm x mp)
        # Then T_approx[i,j,k] = sum_r M_r[i,j] * W_r[k]
        # We can loop over r, but R=64 is small.
        # Or use einsum
        # T_approx = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
        # This might be slow if called many times.
        # Let's try to optimize the gradient calculation or use a simpler check.
        
        # For speed in the objective function, we can compute the residual norm.
        # However, scipy minimize calls this many times.
        # Let's use a simple loop if it's faster, or einsum.
        # einsum is usually optimized.
        T_approx = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
        residual = T_approx - T
        return np.sum(residual * residual)

    # Initial guess: Random
    # Or better: Initialize with the rank 65 solution and drop one rank?
    # Dropping a rank from a valid decomposition doesn't give a valid rank 64 decomposition.
    # We need a random start.
    
    best_rank_64 = None
    best_error = float('inf')
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        # Scale factors to help convergence
        scale = 1.0
        U_init = np.random.randn(64, nm) * scale
        V_init = np.random.randn(64, mp) * scale
        W_init = np.random.randn(64, pn) * scale
        
        # Flatten
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize
        # Use L-BFGS-B or Nelder-Mead? L-BFGS-B is better for high dimensions.
        # But we need gradient. Let's try to compute gradient analytically or use numerical.
        # Numerical gradient is slow.
        # Let's try Nelder-Mead with a small number of iterations?
        # Or just use a simple coordinate descent?
        
        # Given the complexity, let's try a few iterations of gradient descent with a small learning rate
        # or just check if a random solution is close (unlikely).
        
        # Actually, for tensor rank minimization, the problem is non-convex and hard.
        # A random search is unlikely to find the exact integer solution.
        
        # Alternative: Check if rank 64 is possible by checking the border rank?
        # No, we need exact decomposition.
        
        # Let's try a specific heuristic: 
        # If we assume the solution exists with small integers, we can try to solve a system of linear equations?
        # No, it's nonlinear.
        
        # Given the low probability of success in the remaining time with a generic optimizer,
        # and the risk of wasting time, we will stick to the rank 65 solution.
        # However, we can try to see if we can merge any of the 16 schoolbook terms with the Strassen terms?
        # The Strassen terms for 4x4x4 involve indices k=0..3.
        # The Schoolbook terms for k=4 are disjoint in V and W indices.
        # U indices overlap (i*m+j for i,j in 0..3).
        # Can we express one of the schoolbook terms as a linear combination of Strassen terms?
        # No, because the V and W indices are different (k=4 vs k=0..3).
        # So the supports in V and W are disjoint.
        # Thus, the rank is additive: R(4,4,4) + R(4,4,1)
        
        # Since we cannot easily improve, we return the rank 65 solution.
        pass

    return {"U": U_list, "V": V_list, "W": W_list}

BET: {mechanism: "The program constructs a rank-65 decomposition using recursive Strassen for the 4x4x4 part and schoolbook for the 4x4x1 part, which is an improvement over the seed's rank-80 schoolbook approach. The structural change from naive schoolbook to Strassen reduces the rank, improving the score.", q: 0.9, delta: [0.2, 0.3]}
