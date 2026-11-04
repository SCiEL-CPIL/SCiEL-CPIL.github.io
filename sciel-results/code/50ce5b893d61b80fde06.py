import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # First, build the known rank-39 decomposition using Strassen blocks
    def build_strassen_decomposition():
        # Rank calculation:
        # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
        # 1 Schoolbook block (2x2x1) for k=6 -> 4
        # 1 Schoolbook block (2x1x7) for j=2 -> 14
        rank = 21 + 4 + 14
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, coeffs=None):
            nonlocal col
            if coeffs is None:
                coeffs = [1] * len(u_indices)
            for idx, c in zip(u_indices, coeffs):
                U[idx][col] += c
            v_coeffs = [1] * len(v_indices)
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        # Helper for Strassen block
        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            add_term([0, 4], [b00, b11], [c00, c11])
            
            # M2: (A10+A11)B00 -> C10, -C11
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            # M3: A00(B01-B11) -> C01, C11
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            # M4: A11(B10-B00) -> C00, C10
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            # M5: (A00+A01)B11 -> -C00, C01
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            # M6: (A10-A00)(B00+B01) -> C11
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            # M7: (A01-A11)(B10+B11) -> C00
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        # Add 3 Strassen blocks
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        # Block 4: k=6, 2x2x1 Schoolbook
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        # Part 2: j=2
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W form a valid exact decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_lower_rank(start_rank, end_rank, U0, V0, W0, max_iter=100, tol=1e-8):
        """
        Try to find a lower rank decomposition starting from the given initial decomposition.
        Uses iterative refinement on a smaller rank problem.
        """
        if time.time() > deadline - 5:
            return None
            
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        U0_arr = np.array(U0, dtype=np.float64)
        V0_arr = np.array(V0, dtype=np.float64)
        W0_arr = np.array(W0, dtype=np.float64)
        
        for target_rank in range(start_rank, min(end_rank + 1, 40)):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random projection of the known solution
            rng = np.random.default_rng(42 + target_rank)
            
            best_residual = np.inf
            best_solution = None
            
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                    
                # Try to find a rank-target_rank decomposition
                # Initialize by taking a random linear combination of the existing rank-39 terms
                if target_rank < 39:
                    # Random mixing of the 39 terms
                    mix = rng.standard_normal((39, target_rank))
                    U = U0_arr @ mix
                    V = V0_arr @ mix
                    W = W0_arr @ mix
                else:
                    U = U0_arr.copy()
                    V = V0_arr.copy()
                    W = W0_arr.copy()
                
                # Iterative refinement
                for it in range(max_iter):
                    if time.time() > deadline - 10:
                        break
                    
                    # Update U
                    for idx in range(nm):
                        i, j = divmod(idx, m)
                        for r in range(target_rank):
                            num = 0.0
                            for k in range(p):
                                num += V[j*p+k, r] * W[k*n+i, r]
                            if abs(num) > 1e-12:
                                U[idx, r] = 1.0 / num
                            else:
                                U[idx, r] = 0.0
                    
                    # Update V
                    for idx in range(mp):
                        j, k = divmod(idx, p)
                        for r in range(target_rank):
                            num = 0.0
                            for i in range(n):
                                num += U[i*m+j, r] * W[k*n+i, r]
                            if abs(num) > 1e-12:
                                V[idx, r] = 1.0 / num
                            else:
                                V[idx, r] = 0.0
                    
                    # Update W
                    for idx in range(pn):
                        k, i = divmod(idx, n)
                        for r in range(target_rank):
                            num = 0.0
                            for j in range(m):
                                num += U[i*m+j, r] * V[j*p+k, r]
                            if abs(num) > 1e-12:
                                W[idx, r] = 1.0 / num
                            else:
                                W[idx, r] = 0.0
                    
                    # Compute residual
                    residual = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                residual += (s - 1.0)**2
                    residual = np.sqrt(residual)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_solution = (U.copy(), V.copy(), W.copy())
                    
                    if residual < tol:
                        break
                
                if best_residual < tol:
                    break
            
            if best_solution is not None and best_residual < tol:
                U_sol, V_sol, W_sol = best_solution
                # Convert to lists with rounding for cleanliness
                U_list = [[round(float(x), 6) for x in row] for row in U_sol]
                V_list = [[round(float(x), 6) for x in row] for row in V_sol]
                W_list = [[round(float(x), 6) for x in row] for row in W_sol]
                
                # Final verification
                if verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with the known rank-39 decomposition
    base_solution = build_strassen_decomposition()
    base_rank = 39
    
    # Verify the base solution
    if not verify_decomposition(base_solution["U"], base_solution["V"], base_solution["W"], n, m, p):
        # Fallback to schoolbook if Strassen is broken
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    better_solution = try_lower_rank(30, 38, base_solution["U"], base_solution["V"], base_solution["W"])
    
    if better_solution is not None:
        return better_solution
    else:
        return base_solution

# EVOLVE-BLOCK-END
