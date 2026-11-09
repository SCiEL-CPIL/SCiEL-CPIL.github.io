import random
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        """Check exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Fallback: schoolbook rank 27
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb
    
    # Search for rank 23 decomposition
    # Use a structured approach: the tensor entries are all 1, so we need
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # Strategy: try rank 23 with random {-1,0,1} matrices, using many restarts
    # and a local search that tries to fix violations
    
    target_ranks = [23, 24, 25, 26]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 0
        max_restarts = 200000
        
        while not found and num_restarts < max_restarts and time.time() < deadline - 5:
            num_restarts += 1
            
            # Initialize with random sparse {-1, 0, 1} matrices
            density = 0.25
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            for r in range(n*m):
                for c in range(target_rank):
                    if random.random() < density:
                        U[r][c] = 1 if random.random() < 0.5 else -1
            for r in range(m*p):
                for c in range(target_rank):
                    if random.random() < density:
                        V[r][c] = 1 if random.random() < 0.5 else -1
            for r in range(p*n):
                for c in range(target_rank):
                    if random.random() < density:
                        W[r][c] = 1 if random.random() < 0.5 else -1
            
            if verify(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
            
            # Periodically do local search
            if num_restarts % 50 == 0:
                # Compute current errors
                errors = {}
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            errors[(i,j,k)] = s - 1
                
                # Try to fix the worst error by flipping entries
                if errors:
                    worst_key = max(errors, key=lambda x: abs(errors[x]))
                    wi, wj, wk = worst_key
                    # Try flipping entries in U[wi*m+wj, :], V[wj*p+wk, :], W[wk*n+wi, :]
                    for c in range(target_rank):
                        # Flip U
                        old_u = U[wi*m+wj][c]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_u:
                                continue
                            U[wi*m+wj][c] = new_val
                            # Quick check: only affected entries are those with i=wi, j=wj
                            ok = True
                            for k2 in range(p):
                                s = 0
                                for c2 in range(target_rank):
                                    s += U[wi*m+wj][c2] * V[wj*p+k2][c2] * W[k2*n+wi][c2]
                                if s != 1:
                                    ok = False
                                    break
                            if ok:
                                # Check all others still fine (they should be since we only changed U[wi,wj,:])
                                # Actually changing U[wi*m+wj,:] only affects entries with i=wi,j=wj
                                # So we just need those to be 1
                                U[wi*m+wj][c] = old_u
                                # Actually we want to keep the new value if it works
                                U[wi*m+wj][c] = new_val
                                # Verify full solution
                                if verify(U, V, W, target_rank):
                                    best_result = {"U": U, "V": V, "W": W}
                                    best_rank = target_rank
                                    found = True
                                    break
                                else:
                                    U[wi*m+wj][c] = old_u
                            else:
                                U[wi*m+wj][c] = old_u
                        if found:
                            break
                        
                        # Flip V
                        old_v = V[wj*p+wk][c]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_v:
                                continue
                            V[wj*p+wk][c] = new_val
                            ok = True
                            for i2 in range(n):
                                s = 0
                                for c2 in range(target_rank):
                                    s += U[i2*m+wj][c2] * V[wj*p+wk][c2] * W[wk*n+i2][c2]
                                if s != 1:
                                    ok = False
                                    break
                            if ok:
                                V[wj*p+wk][c] = new_val
                                if verify(U, V, W, target_rank):
                                    best_result = {"U": U, "V": V, "W": W}
                                    best_rank = target_rank
                                    found = True
                                    break
                                else:
                                    V[wj*p+wk][c] = old_v
                            else:
                                V[wj*p+wk][c] = old_v
                        if found:
                            break
                        
                        # Flip W
                        old_w = W[wk*n+wi][c]
                        for new_val in [-1, 0, 1]:
                            if new_val == old_w:
                                continue
                            W[wk*n+wi][c] = new_val
                            ok = True
                            for j2 in range(m):
                                s = 0
                                for c2 in range(target_rank):
                                    s += U[wi*m+j2][c2] * V[wj*p+j2][c2] * W[wk*n+wi][c2]
                                if s != 1:
                                    ok = False
                                    break
                            if ok:
                                W[wk*n+wi][c] = new_val
                                if verify(U, V, W, target_rank):
                                    best_result = {"U": U, "V": V, "W": W}
                                    best_rank = target_rank
                                    found = True
                                    break
                                else:
                                    W[wk*n+wi][c] = old_w
                            else:
                                W[wk*n+wi][c] = old_w
                        if found:
                            break
                    if found:
                        break
    
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
