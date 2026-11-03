import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute for fast cost evaluation
    # For swap(i, j): delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]]) 
    #                + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    #                + a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    def swap_delta(p, i, j):
        """Compute cost delta for swapping positions i and j."""
        u, v = p[i], p[j]
        if u == v:
            return 0
        # Vectorized delta computation
        diff_a_row = a[i, :] - a[j, :]
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[v, :] - b[u, :]
        diff_b_col = b[:, v] - b[:, u]
        
        s1 = np.dot(diff_a_row, diff_b_row[p])
        s2 = np.dot(diff_a_col, diff_b_col[p])
        
        # Subtract self-interaction terms
        s1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        s1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        s2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        s2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        return s1 + s2 + cross
    
    # Simulated Annealing with multiple neighborhoods
    rng = np.random.RandomState(42)
    
    # Initial solution: greedy by row sums
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Temperature schedule
    T_start = max(1.0, abs(current_cost) * 0.01)
    T_end = max(0.1, T_start * 0.001)
    alpha = 0.995  # cooling rate
    
    T = T_start
    step_count = 0
    
    while time.time() < deadline:
        # Choose move type: 60% swap, 20% insert, 20% block
        move_type = rng.random()
        
        if move_type < 0.6:
            # Swap two random positions
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            delta = swap_delta(p, i, j)
            
            # Acceptance criterion
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                step_count += 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        elif move_type < 0.8:
            # Insert: move element at position i to position j
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            # Compute delta for moving p[i] to position j
            # This is more complex; use full recomputation for simplicity
            old_p = p.copy()
            # Remove from i and insert at j
            if i < j:
                p[i:j] = p[i+1:j+1]
                p[j] = old_p[i]
            else:
                p[i+1:j+1] = p[i:j]
                p[i] = old_p[i]
            
            new_cost = compute_cost(p)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                step_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                p = old_p  # revert
        else:
            # Block move: reverse a segment
            i, j = sorted(rng.choice(n, size=2, replace=False))
            if i == j:
                continue
            old_p = p.copy()
            p[i:j+1] = p[i:j+1][::-1]
            
            new_cost = compute_cost(p)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                step_count += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            else:
                p = old_p  # revert
        
        # Cool down
        T = max(T_end, T * alpha)
        
        # Periodic local search refinement
        if step_count % 1000 == 0 and time.time() < deadline - 10:
            # Do a few passes of best-improvement 2-swap
            for _ in range(3):
                improved = False
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i + 1, n):
                        delta = swap_delta(p, i, j)
                        if delta < 0:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            improved = True
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = p.copy()
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    return best_p.tolist()
