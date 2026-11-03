# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0

    # Precompute useful structures
    a_rows = a
    a_cols = a.T
    b_matrix = b
    
    # Compute initial cost efficiently
    def compute_cost(perm):
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    # Fast delta computation for swapping i and j
    def compute_delta(i, j, perm):
        li = perm[i]
        lj = perm[j]
        if li == lj:
            return 0

        # Delta = sum over all k of a[k,i]*(b[perm[k],lj]-b[perm[k],li]) + a[k,j]*(b[perm[k],li]-b[perm[k],lj])
        #       + sum over all k of a[i,k]*(b[li,perm[k]]-b[lj,perm[k]]) + a[j,k]*(b[lj,perm[k]]-b[li,perm[k]])
        #       + cross terms
        
        # Let's use a more efficient formulation
        # When we swap positions i and j (facilities at locations li and lj),
        # the change in cost comes from:
        # 1. Terms involving facility i: a[i,*] and a[*,i]
        # 2. Terms involving facility j: a[j,*] and a[*,j]
        
        # For row i (facility i to others):
        # Old: sum_k a[i,k] * b[li, perm[k]]  (k != i,j) + a[i,j]*b[li,lj] + a[i,i]*b[li,li]
        # New: sum_k a[i,k] * b[lj, perm[k]]  (k != i,j) + a[i,j]*b[lj,li] + a[i,i]*b[lj,li]
        
        # For column i (others to facility i):
        # Old: sum_k a[k,i] * b[perm[k], li]  (k != i,j) + a[j,i]*b[lj,li] + a[i,i]*b[li,li]
        # New: sum_k a[k,i] * b[perm[k], lj]  (k != i,j) + a[j,i]*b[li,lj] + a[i,i]*b[lj,li]
        
        # Similarly for j...
        
        # Let's compute using vectorized operations
        diff_row_i = b_matrix[lj, :] - b_matrix[li, :]
        diff_col_i = b_matrix[:, lj] - b_matrix[:, li]
        diff_row_j = b_matrix[li, :] - b_matrix[lj, :]
        diff_col_j = b_matrix[:, li] - b_matrix[:, lj]
        
        # Sum over all k except i and j
        s = np.zeros(1)
        
        # Row i contribution: a[i,:] dot diff_row_i, but exclude i and j
        s[0] += np.dot(a_rows[i, :], diff_row_i[perm])
        s[0] -= a_rows[i, i] * diff_row_i[li]
        s[0] -= a_rows[i, j] * diff_row_i[lj]
        
        # Col i contribution: a[:,i] dot diff_col_i, but exclude i and j
        s[0] += np.dot(a_cols[i, :], diff_col_i[perm])
        s[0] -= a_cols[i, i] * diff_col_i[li]
        s[0] -= a_cols[i, j] * diff_col_i[lj]
        
        # Row j contribution: a[j,:] dot diff_row_j, but exclude i and j
        s[0] += np.dot(a_rows[j, :], diff_row_j[perm])
        s[0] -= a_rows[j, i] * diff_row_j[li]
        s[0] -= a_rows[j, j] * diff_row_j[lj]
        
        # Col j contribution: a[:,j] dot diff_col_j, but exclude i and j
        s[0] += np.dot(a_cols[j, :], diff_col_j[perm])
        s[0] -= a_cols[j, i] * diff_col_j[li]
        s[0] -= a_cols[j, j] * diff_col_j[lj]
        
        # Cross term: a[i,j]*b[li,lj] -> a[i,j]*b[lj,li], and a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
        # Change: a[i,j]*(b[lj,li]-b[li,lj]) + a[j,i]*(b[li,lj]-b[lj,li])
        cross = (a_rows[j, i] - a_rows[i, j]) * (b_matrix[li, lj] - b_matrix[lj, li])
        s[0] += cross
        
        return s[0]

    # Simulated Annealing parameters
    T_initial = 5000.0
    T_final = 1.0
    cooling_rate = 0.9995
    max_iter = 500000
    
    # Generate initial solution using greedy + randomization
    def generate_initial_solution():
        # Use rank-based initialization with some randomization
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        
        # Mix deterministic and random
        for idx in range(n):
            if np.random.random() < 0.3:
                p[heavy[idx]] = np.random.randint(n)
            else:
                p[heavy[idx]] = central[idx]
        
        # Ensure it's a valid permutation
        used = set(p.tolist())
        remaining = set(range(n)) - used
        for i in range(n):
            if p[i] in used and i not in heavy[:n]:
                pass
        # Fix any duplicates by assigning unused locations
        assigned = set(p.tolist())
        unassigned = set(range(n)) - assigned
        for i in range(n):
            if p[i] in assigned and sum(1 for x in p if x == p[i]) > 1:
                p[i] = unassigned.pop()
        
        return p

    # Run multiple SA restarts
    best_cost = float('inf')
    best_p = None
    
    num_restarts = 8
    time_per_restart = (time_budget - 3.0) / num_restarts
    
    for restart in range(num_restarts):
        # Generate initial solution
        if restart == 0:
            # First restart: use the good initial from before
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            p = generate_initial_solution()
        
        current_cost = compute_cost(p)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        T = T_initial
        no_improve = 0
        
        while T > T_final and time.time() - start_time < time_budget - 2.0:
            # Choose random pair to swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            
            if i == j:
                continue
            
            delta = compute_delta(i, j, p)
            
            # Metropolis criterion
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept move
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            
            # Cool down
            T *= cooling_rate
            
            # Adaptive cooling: if stuck, cool faster
            if no_improve > 500:
                T *= 0.99
                no_improve = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
