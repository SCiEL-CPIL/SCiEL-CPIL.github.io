import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 115  # seconds, safety margin for 120s limit

    def is_valid(U, V, W, n, m, p):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if val != 1:
                        return False
        return True

    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def search_decomposition(n, m, p, target_rank, time_limit, max_attempts=1000):
        """
        Searches for a decomposition of rank target_rank for n x m x p.
        Returns (U, V, W) as numpy arrays or None.
        """
        start_time = time.time()
        for R in range(target_rank, 20, -1):
            if time.time() - start_time > time_limit:
                break
            
            attempts = 0
            while attempts < max_attempts and (time.time() - start_time) < time_limit:
                attempts += 1
                
                # Generate random U, V with entries in {-1, 0, 1}
                U = np.random.randint(-1, 2, size=(n * m, R))
                V = np.random.randint(-1, 2, size=(m * p, R))
                
                # Construct linear system for W
                num_vars = p * n * R
                A_mat = np.zeros((n * m * p, num_vars))
                b_vec = np.ones(n * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_base = k * n + i
                            for r in range(R):
                                var_idx = w_base * R + r
                                A_mat[eq_idx, var_idx] = U[u_idx, r] * V[v_idx, r]
                            eq_idx += 1
                
                try:
                    W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    if residuals[0] < 1e-9:
                        W_int = np.round(W_flat)
                        if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                            W_sol = W_int.reshape((p * n, R))
                            if is_valid(U, V, W_sol, n, m, p):
                                return U, V, W_sol
                except:
                    pass
        return None

    # Strategy 1: Try to find a rank 23 decomposition for 3x3x3 and extend to 3x3x4 with rank 26
    # This is a specific constructive approach based on Laderman's result.
    if time.time() - start_time < 30: # Time budget for this phase
        U3, V3, W3 = search_decomposition(3, 3, 3, 23, time_limit=25, max_attempts=200)
        if U3 is not None:
            # Construct 3x3x4 decomposition
            # U3: 9x23, V3: 9x23, W3: 9x23
            # We need U: 9x26, V: 12x26, W: 12x26
            
            R_total = 26
            U_new = np.zeros((9, R_total), dtype=int)
            V_new = np.zeros((12, R_total), dtype=int)
            W_new = np.zeros((12, R_total), dtype=int)
            
            # Copy Laderman part
            U_new[:, :23] = U3
            # V3 has rows for j=0..2, k=0..2. Index: j*3 + k
            # V_new has rows for j=0..2, k=0..3. Index: j*4 + k
            for j in range(3):
                for k in range(3):
                    V_new[j * 4 + k, :23] = V3[j * 3 + k, :23]
            # V_new rows for k=3 are currently 0.
            
            # W3 has rows for k=0..2, i=0..2. Index: k*3 + i
            # W_new has rows for k=0..3, i=0..2. Index: k*4 + i? No, W is (p*n, R) -> (12, R).
            # Index in W is k*n + i = k*3 + i.
            for k in range(3):
                for i in range(3):
                    W_new[k * 3 + i, :23] = W3[k * 3 + i, :23]
            # W_new rows for k=3 are currently 0.
            
            # Add the 3 terms for k=3
            # U_new[:, 23:26]
            # V_new[j*4 + 3, 23:26]
            # W_new[3*3 + i, 23:26]
            
            # We need sum_s U_new[i,j, 23+s] * V_new[j,3, 23+s] * W_new[3,i, 23+s] = 1
            # Using U_{ij,s} = delta_{is}, V_{j3,s} = 1, W_{3i,s} = delta_{is}
            for s in range(3):
                # s corresponds to i = s
                for i in range(3):
                    if i == s:
                        U_new[i * 3 + j, 23 + s] = 1 # Wait, U depends on i,j.
                        # U_{ij, 23+s} = 1 if i==s.
                        # So for fixed s, U_{ij, 23+s} = 1 for all j if i==s.
                        pass
                
                # Let's just fill it directly
                for i in range(3):
                    for j in range(3):
                        if i == s:
                            U_new[i * 3 + j, 23 + s] = 1
                        else:
                            U_new[i * 3 + j, 23 + s] = 0
                        
                        # V_{j3, 23+s} = 1
                        V_new[j * 4 + 3, 23 + s] = 1
                        
                        # W_{3i, 23+s} = 1 if i==s
                        if i == s:
                            W_new[3 * 3 + i, 23 + s] = 1
                        else:
                            W_new[3 * 3 + i, 23 + s] = 0
            
            # Verify
            if is_valid(U_new, V_new, W_new, 3, 3, 4):
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }

    # Strategy 2: General search for 3x3x4 with R < 36
    # Search from 35 down to 21
    for R in range(35, 20, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a solution for this specific R
        # We can call the search function, but it loops R down.
        # Let's just run the loop logic directly for efficiency
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            num_vars = p * n * R
            A_mat = np.zeros((n * m * p, num_vars))
            b_vec = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_base = k * n + i
                        for r in range(R):
                            var_idx = w_base * R + r
                            A_mat[eq_idx, var_idx] = U[u_idx, r] * V[v_idx, r]
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if residuals[0] < 1e-9:
                    W_int = np.round(W_flat)
                    if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                        W_sol = W_int.reshape((p * n, R))
                        if is_valid(U, V, W_sol, n, m, p):
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W_sol.tolist()
                            }
            except:
                pass

    # Fallback
    return get_schoolbook(n, m, p)

# EVOLVE-BLOCK-START
# (No changes to interface)
# EVOLVE-BLOCK-END
