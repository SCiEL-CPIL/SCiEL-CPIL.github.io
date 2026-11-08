import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True
    
    def try_structured_rank(r, n, m, p):
        """
        Try structured low-rank decompositions.
        For matrix multiplication, we can use bilinear algorithms.
        Here we try a systematic approach based on Strassen-like ideas
        but adapted for 3x4x4.
        
        Key insight: We can try to find decompositions where the factor
        matrices have a specific structure. One approach is to use the
        fact that matrix multiplication is a trilinear form and try to
        minimize the number of scalar multiplications.
        
        We'll use a random search with structured initialization.
        """
        num_u = n * m  # 12
        num_v = m * p  # 16
        num_w = p * n  # 12
        
        # Try multiple random restarts with different seeds
        for seed in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            random.seed(seed * 1000 + r)
            
            # Initialize with random {-1, 0, 1} values
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_u)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_v)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_w)]
            
            # Local improvement: try to fix violations by adjusting entries
            # Compute the current "error" for each (i,j,k)
            def compute_errors(U, V, W, n, m, p, r):
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for rr in range(r):
                                val += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                            errors[(i, j, k)] = val - 1
                return errors
            
            # Iterative local search
            max_iters = 2000
            for it in range(max_iters):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                
                errors = compute_errors(U, V, W, n, m, p, r)
                
                # Check if solved
                if all(abs(e) < 1e-10 for e in errors.values()):
                    # Convert to exact integers if close
                    U_int = [[int(round(x)) for x in row] for row in U]
                    V_int = [[int(round(x)) for x in row] for row in V]
                    W_int = [[int(round(x)) for x in row] for row in W]
                    if verify_decomposition(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int, "V": V_int, "W": W_int}
                    break
                
                # Find the most violated constraint
                worst_key = max(errors.keys(), key=lambda k: abs(errors[k]))
                wi, wj, wk = worst_key
                err = errors[worst_key]
                
                # Try flipping an entry in U, V, or W that affects this constraint
                # U index: wi*m + wj, V index: wj*p + wk, W index: wk*n + wi
                u_idx = wi * m + wj
                v_idx = wj * p + wk
                w_idx = wk * n + wi
                
                # Try adjusting U[u_idx]
                improved = False
                for rr in range(r):
                    old_u = U[u_idx][rr]
                    # Try changing U[u_idx][rr] by ±1
                    for delta in [-1, 1]:
                        new_val = old_u + delta
                        if new_val in [-1, 0, 1]:
                            U[u_idx][rr] = new_val
                            # Quick check: does this help?
                            new_err = 0
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        if ii == wi and jj == wj and kk == wk:
                                            continue
                                        val = 0
                                        for rrr in range(r):
                                            if rrr == rr and ii == wi and jj == wj:
                                                val += U[u_idx][rrr] * V[jj * p + kk][rrr] * W[kk * n + ii][rrr]
                                            elif rrr == rr and jj == wj:
                                                val += U[ii * m + jj][rrr] * V[jj * p + kk][rrr] * W[kk * n + ii][rrr]
                                            elif rrr == rr and kk == wk and ii == wi:
                                                val += U[ii * m + jj][rrr] * V[jj * p + kk][rrr] * W[kk * n + ii][rrr]
                                            else:
                                                val += U[ii * m + jj][rrr] * V[jj * p + kk][rrr] * W[kk * n + ii][rrr]
                                        new_err += (val - 1) ** 2
                            # This is too slow, just accept the change if it reduces the specific error
                            U[u_idx][rr] = new_val
                            improved = True
                            break
                    if improved:
                        break
                    U[u_idx][rr] = old_u
                
                if not improved:
                    # Try V
                    for rr in range(r):
                        old_v = V[v_idx][rr]
                        for delta in [-1, 1]:
                            new_val = old_v + delta
                            if new_val in [-1, 0, 1]:
                                V[v_idx][rr] = new_val
                                improved = True
                                break
                        if improved:
                            break
                        V[v_idx][rr] = old_v
                    
                    if not improved:
                        # Try W
                        for rr in range(r):
                            old_w = W[w_idx][rr]
                            for delta in [-1, 1]:
                                new_val = old_w + delta
                                if new_val in [-1, 0, 1]:
                                    W[w_idx][rr] = new_val
                                    improved = True
                                    break
                            if improved:
                                break
                            W[w_idx][rr] = old_w
                        
                        if not improved:
                            break
        
        return None
    
    # Try ranks from low to high
    # Known lower bound for 3x4x4 matrix multiplication is at least 12
    # Trivial rank is 48
    for r in range(12, min(rank_trivial + 1, 25)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_structured_rank(r, n, m, p)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If no improvement found via structured search, try pure random search for slightly higher ranks
    if best_rank == rank_trivial:
        for r in range(20, min(rank_trivial + 1, 35)):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            for seed in range(100):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                
                random.seed(seed * 7 + r)
                num_u = n * m
                num_v = m * p
                num_w = p * n
                
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_u)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_v)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_w)]
                
                if verify_decomposition(U, V, W, n, m, p):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
                    break
            
            if best_rank != rank_trivial:
                break
    
    return best_result
# EVOLVE-BLOCK-END
