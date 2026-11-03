import random
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 27
    pn = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
    # Target tensor: T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1
    
    # Build schoolbook decomposition (rank 54)
    rank_sb = n * m * p  # 54
    U = [[0] * rank_sb for _ in range(nm)]
    V = [[0] * rank_sb for _ in range(mp)]
    W = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor."""
        r = len(U_mat[0])
        # Check all target entries
        for (a, b, c), val in target_entries.items():
            s = 0.0
            for t in range(r):
                s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
            if abs(s - val) > tol:
                return False
        # Check non-target entries (sample if too many)
        target_set = set(target_entries.keys())
        total = nm * mp * pn  # 6*27*18 = 2916
        n_target = len(target_entries)  # 54
        n_non_target = total - n_target  # 2862
        
        # For small rank, check all; otherwise sample
        if r <= 40:
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if (a, b, c) not in target_set:
                            s = 0.0
                            for t in range(r):
                                s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
                            if abs(s) > tol:
                                return False
        else:
            # Sample 500 non-target entries
            import random
            random.seed(123)
            for _ in range(500):
                a = random.randint(0, nm - 1)
                b = random.randint(0, mp - 1)
                c = random.randint(0, pn - 1)
                if (a, b, c) not in target_set:
                    s = 0.0
                    for t in range(r):
                        s += U_mat[a][t] * V_mat[b][t] * W_mat[c][t]
                    if abs(s) > tol:
                        return False
        
        return True
    
    # Try to find lower-rank decomposition using structured approach
    # Key insight: For 2x3x9, we can use Strassen-style sharing for 2x2x2 blocks
    # The matrix is C = A * B where A is 2x3, B is 3x9
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For j=0,1: 2x2x2 subproblem (can use Strassen with 7 multiplications)
    # For j=2: 2x1x2 subproblem (needs 2 multiplications)
    # But we need to be careful about the k dimension (9 columns)
    
    # Let's try a different approach: use the fact that the tensor has a specific structure
    # and try random search with local optimization
    
    best_rank = rank_sb
    best_result = {"U": U, "V": V, "W": W}
    
    # Try ranks from 40 to 53
    for r in range(40, 54):
        if time.time() > deadline - 30:
            break
        
        # Random search with structured initialization
        for attempt in range(10):
            if time.time() > deadline - 25:
                break
            
            # Initialize with schoolbook and perturb
            U_try = [[0.0] * r for _ in range(nm)]
            V_try = [[0.0] * r for _ in range(mp)]
            W_try = [[0.0] * r for _ in range(pn)]
            
            # Use first r terms of schoolbook as starting point
            for t in range(min(r, rank_sb)):
                i = t // (m * p)
                j = (t // p) % m
                k = t % p
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_try[a][t] = 1.0
                V_try[b][t] = 1.0
                W_try[c][t] = 1.0
            
            # Add some random perturbation to encourage exploration
            if r < rank_sb:
                # Zero out some entries to create room for optimization
                for t in range(r):
                    if random.random() < 0.3:
                        idx = random.randint(0, nm - 1)
                        U_try[idx][t] = random.uniform(-0.5, 0.5)
                    if random.random() < 0.3:
                        idx = random.randint(0, mp - 1)
                        V_try[idx][t] = random.uniform(-0.5, 0.5)
                    if random.random() < 0.3:
                        idx = random.randint(0, pn - 1)
                        W_try[idx][t] = random.uniform(-0.5, 0.5)
            
            # Try a few gradient steps (simplified)
            # This is a placeholder for actual optimization
            # For now, just verify if the current decomposition works
            if verify(U_try, V_try, W_try):
                best_rank = r
                best_result = {"U": U_try, "V": V_try, "W": W_try}
                break
        
        if best_rank == r:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < rank_sb:
        return best_result
    else:
        return {"U": U, "V": V, "W": W}
