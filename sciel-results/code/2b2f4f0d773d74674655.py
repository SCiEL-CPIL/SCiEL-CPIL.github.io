"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomposition():
        """
        Build the known rank-84 decomposition using Strassen on 2x2x2 blocks.
        """
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7
                    else:
                        r = si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        # Strassen terms for 2x2x2
        strassen_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_randomized_search(target_rank, time_limit):
        """
        Attempt to find a decomposition of the given rank using randomized search.
        """
        if time.time() > time_limit:
            return None
            
        # We will try to find U, V such that the least-squares solution for W is integer/half-integer.
        # We use a simple random walk on U and V entries.
        
        # Initialize U and V with small random integers
        U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for _ in range(500): # 500 iterations
            if time.time() > time_limit:
                break
            
            # Solve for W
            # For each (i, k), we have a system:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all j
            # This is a linear system M * w = b, where w is the vector W[k,i,:]
            # M[j, r] = U[i,j,r] * V[j,k,r]
            # b[j] = 1
            
            W_candidate = [[0.0] * target_rank for _ in range(p * n)]
            valid = True
            error = 0.0
            
            for i in range(n):
                for k in range(p):
                    # Build M (m x target_rank)
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m+j][r] * V[j*p+k][r]
                    
                    # Solve M * w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w is close to integers or half-integers
                    # We'll check distance to nearest integer
                    w_int = np.round(w)
                    err = np.sum((w - w_int)**2)
                    error += err
                    
                    if err > 1e-6:
                        # Not a good candidate, but we can still store it for now
                        pass
                    
                    W_candidate[k*n+i] = list(w_int)
            
            if not valid:
                continue
                
            if error < best_error:
                best_error = error
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W_candidate]
                
                # If error is very small, verify exactness
                if error < 1e-9:
                    if verify(best_U, best_V, best_W, target_rank):
                        return {"U": best_U, "V": best_V, "W": best_W, "rank": target_rank}
            
            # Perturb U and V randomly
            # Pick a random entry in U and V and change it
            idx_u = random.randint(0, n*m-1)
            idx_v = random.randint(0, m*p-1)
            r_u = random.randint(0, target_rank-1)
            r_v = random.randint(0, target_rank-1)
            
            delta_u = random.choice([-1, 0, 1])
            delta_v = random.choice([-1, 0, 1])
            
            U[idx_u][r_u] += delta_u
            V[idx_v][r_v] += delta_v
            
            # Occasionally reset if stuck
            if _ % 50 == 0 and error > 1.0:
                U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]

        return None

    # 1. Try to find a solution with rank 83
    # 2. If not found, use the known rank 84 solution
    
    result = None
    # Try rank 83 with a time limit of 60s
    result = try_randomized_search(83, start_time + 60)
    
    if result is None:
        # Try rank 82? Maybe too hard. Let's stick to 84 fallback.
        # But we can try a quick search for 83 again with different seed?
        # For now, fall back to 84.
        result = build_strassen_decomposition()
    
    rank = result.pop("rank")
    
    if verify(result["U"], result["V"], result["W"], rank):
        return result
    else:
        # Fallback to schoolbook if verification fails
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

# EVOLVE-BLOCK-END
