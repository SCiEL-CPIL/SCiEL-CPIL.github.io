import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Precompute the positions of non-zero entries for faster cost computation
    nz_positions = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_positions.append((i*m + j, j*p + k, k*n + i))
    
    def check_decomposition(U, V, W, tol=1e-9):
        rank = U.shape[1]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.linalg.norm(T - T_recon) < tol
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = get_schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    def compute_cost_fast(U, V, W):
        """Compute cost only at non-zero positions for speed"""
        cost = 0.0
        for (a, b, c) in nz_positions:
            recon = 0.0
            for r in range(U.shape[1]):
                recon += U[a, r] * V[b, r] * W[c, r]
            diff = 1.0 - recon
            cost += diff * diff
        return cost
    
    def local_search(U, V, W, max_iter=500):
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_cost = compute_cost_fast(current_U, current_V, current_W)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = current_U
            elif mat_idx == 1:
                mat = current_V
            else:
                mat = current_W
            
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, mat.shape[1] - 1)
            
            old_val = mat[row, col]
            delta = random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            
            if mat_idx == 0:
                current_U[row, col] = new_val
            elif mat_idx == 1:
                current_V[row, col] = new_val
            else:
                current_W[row, col] = new_val
            
            new_cost = compute_cost_fast(current_U, current_V, current_W)
            
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                if mat_idx == 0:
                    current_U[row, col] = old_val
                elif mat_idx == 1:
                    current_V[row, col] = old_val
                else:
                    current_W[row, col] = old_val
        
        return current_U, current_V, current_W, current_cost
    
    def structured_init(target_rank):
        """Initialize by grouping schoolbook terms and combining them"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Assign each schoolbook term to one of target_rank groups
        # Use a deterministic assignment based on modular arithmetic
        for idx in range(n * m * p):
            i = idx // (m * p)
            j = (idx // p) % m
            k = idx % p
            r = idx % target_rank
            U[i*m + j, r] += 1.0
            V[j*p + k, r] += 1.0
            W[k*n + i, r] += 1.0
        
        # Add small random perturbations to break symmetry
        for r in range(target_rank):
            if random.random() < 0.3:
                U[random.randint(0, n*m-1), r] += random.choice([-0.5, 0.5])
                V[random.randint(0, m*p-1), r] += random.choice([-0.5, 0.5])
                W[random.randint(0, p*n-1), r] += random.choice([-0.5, 0.5])
        
        return U, V, W
    
    # Try decreasing ranks from 41 down to 35
    for target_rank in range(best_rank - 1, max(30, best_rank - 8), -1):
        if time.time() > deadline - 10:
            break
        
        found_better = False
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            if attempts % 10 == 0:
                # Use structured initialization every 10th attempt
                U, V, W = structured_init(target_rank)
            else:
                U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, target_rank))
                V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, target_rank))
                W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, target_rank))
            
            U, V, W, cost = local_search(U, V, W, max_iter=300)
            
            if cost < 1e-9:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found_better = True
                break
            
            # Also try a quick verification with full einsum for safety
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.linalg.norm(T - T_recon) < 1e-9:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found_better = True
                break
        
        if found_better:
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
