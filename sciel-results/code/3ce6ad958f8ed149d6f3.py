import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 115  # Safety margin before 120s limit

    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Rank-50 construction
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
    
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
            
    rank = len(U_cols)
    
    def verify_decomposition(U_cols, V_cols, W_cols):
        r = len(U_cols)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = 0
                    for s in range(r):
                        val += U_cols[s][a] * V_cols[s][b] * W_cols[s][c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True

    def try_als_reduce(target_rank, U_init, V_init, W_init):
        """Try to reduce to target_rank using Alternating Least Squares."""
        if time.time() > deadline - 20:
            return None
        
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        # Precompute target tensor values
        target = np.zeros((n, m, p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    target[i, k, j] = 1.0
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # Equation: sum_s U_s[a] V_s[b] W_s[c] = 1
            # For fixed V, W, this is linear in U.
            # Flatten U to vector u of size target_rank * n*m
            # A_u is (n*m*p) x (target_rank * n*m)
            # A_u[idx, s*n*m + a] = V_s[b] * W_s[c] where idx corresponds to (a,b,c)
            
            A_u = np.zeros((n*m*p, target_rank * n*m))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        idx = i*m*p + k*p + j # Flatten index for target
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        for s in range(target_rank):
                            A_u[idx, s * n*m + a] = V[s, b] * W[s, c]
            
            b_u = np.ones(n*m*p)
            try:
                u_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = u_sol.reshape(target_rank, n*m)
            except:
                pass
                
            # Update V
            A_v = np.zeros((n*m*p, target_rank * m*p))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        idx = i*m*p + k*p + j
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        for s in range(target_rank):
                            A_v[idx, s * m*p + b] = U[s, a] * W[s, c]
            
            b_v = np.ones(n*m*p)
            try:
                v_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = v_sol.reshape(target_rank, m*p)
            except:
                pass
                
            # Update W
            A_w = np.zeros((n*m*p, target_rank * p*n))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        idx = i*m*p + k*p + j
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        for s in range(target_rank):
                            A_w[idx, s * p*n + c] = U[s, a] * V[s, b]
            
            b_w = np.ones(n*m*p)
            try:
                w_sol, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = w_sol.reshape(target_rank, p*n)
            except:
                pass
            
            # Check convergence
            err = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        val = 0
                        for s in range(target_rank):
                            val += U[s, a] * V[s, b] * W[s, c]
                        err += (val - 1.0)**2
            if err < 1e-10:
                # Convert back to list of lists
                new_U_cols = [U[s, :].tolist() for s in range(target_rank)]
                new_V_cols = [V[s, :].tolist() for s in range(target_rank)]
                new_W_cols = [W[s, :].tolist() for s in range(target_rank)]
                return (new_U_cols, new_V_cols, new_W_cols)
                
        return None

    # Try ALS to reduce to 49
    res = try_als_reduce(49, U_cols, V_cols, W_cols)
    if res:
        U_cols, V_cols, W_cols = res
        rank = len(U_cols)
    else:
        # Fallback to term removal search
        def try_remove_terms(indices_to_remove):
            if time.time() > deadline - 3:
                return None
            
            r = len(U_cols)
            remaining = [s for s in range(r) if s not in indices_to_remove]
            num_remaining = len(remaining)
            
            if num_remaining == 0:
                return None
            
            equations_A = []
            equations_b = []
            
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        row = np.zeros(num_remaining)
                        for idx, s in enumerate(remaining):
                            row[idx] = U_cols[s][a] * V_cols[s][b] * W_cols[s][c]
                        equations_A.append(row)
                        equations_b.append(1.0)
            
            A_sys = np.array(equations_A)
            b_sys = np.array(equations_b)
            
            try:
                d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                
                A_sys_d = A_sys @ d
                if not np.allclose(A_sys_d, b_sys, atol=1e-10):
                    return None
                
                new_U_cols = [U_cols[s][:] for s in remaining]
                new_V_cols = [V_cols[s][:] for s in remaining]
                new_W_cols = [W_cols[s][:] for s in remaining]
                
                for idx, s in enumerate(remaining):
                    scale = d[idx]
                    if abs(scale) < 1e-15:
                        continue
                    for i in range(n*m):
                        new_U_cols[idx][i] *= scale
                    for i in range(m*p):
                        new_V_cols[idx][i] *= scale
                    for i in range(p*n):
                        new_W_cols[idx][i] *= scale
                
                if verify_decomposition(new_U_cols, new_V_cols, new_W_cols):
                    return (new_U_cols, new_V_cols, new_W_cols)
            except:
                pass
            
            return None

        # Try removing 1 term
        for r_remove in range(rank):
            if time.time() > deadline - 10:
                break
            res = try_remove_terms([r_remove])
            if res:
                U_cols, V_cols, W_cols = res
                rank = len(U_cols)
                break
        
        # Try removing 2 terms
        if rank > 40 and time.time() < deadline - 15:
            for r1, r2 in itertools.combinations(range(rank), 2):
                if time.time() > deadline - 5:
                    break
                res = try_remove_terms([r1, r2])
                if res:
                    U_cols, V_cols, W_cols = res
                    rank = len(U_cols)
                    break
    
    # Final cleanup
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
