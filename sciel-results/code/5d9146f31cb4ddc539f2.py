import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
    
    def delta_cost_vectorized(i, j, p):
        """Vectorized delta cost calculation for swapping facilities i and j"""
        li = p[i]
        lj = p[j]
        
        # Precompute differences
        b_diff = b[lj] - b[li]  # b[lj, :] - b[li, :]
        b_diff_T = b[:, lj] - b[:, li]  # b[:, lj] - b[:, li]
        
        # Vectorized computation over all k
        # a[i, :] * (b[lj, p] - b[li, p]) = a[i, :] * b_diff[p]
        term1 = np.dot(a[i, :], b_diff[p])
        # (a[:, i]) * (b[p, lj] - b[p, li]) = a[:, i] * b_diff_T[p]
        term2 = np.dot(a[:, i], b_diff_T[p])
        # a[j, :] * (b[li, p] - b[lj, p]) = -a[j, :] * b_diff[p]
        term3 = -np.dot(a[j, :], b_diff[p])
        # (a[:, j]) * (b[p, li] - b[p, lj]) = -a[:, j] * b_diff_T[p]
        term4 = -np.dot(a[:, j], b_diff_T[p])
        
        delta = term1 + term2 + term3 + term4
        
        # Subtract contributions from k=i and k=j (they were included in the sums above)
        # For k=i:
        # old: a[i,i]*b[li,li] + a[i,i]*b[li,li] + a[j,i]*b[lj,li] + a[i,j]*b[li,lj]
        # But wait, we need to be more careful. Let me recalculate.
        
        # Actually, let's use a cleaner approach. The vectorized formula above includes k=i and k=j.
        # Let's manually correct for those terms.
        
        # For k=i:
        # In term1: a[i,i] * (b[lj,li] - b[li,li])
        # In term2: a[i,i] * (b[li,lj] - b[li,li])  [since p[i]=li, so b[p[i],lj]-b[p[i],li] = b[li,lj]-b[li,li]]
        # In term3: -a[j,i] * (b[lj,li] - b[li,li])
        # In term4: -a[i,j] * (b[li,lj] - b[li,li])
        
        # For k=j:
        # In term1: a[i,j] * (b[lj,lj] - b[li,lj])
        # In term2: a[j,i] * (b[lj,lj] - b[li,lj])  [since p[j]=lj, so b[p[j],lj]-b[p[j],li] = b[lj,lj]-b[lj,li]]
        # Wait, p[j]=lj, so b[p[j],lj] = b[lj,lj], b[p[j],li] = b[lj,li]
        # In term3: -a[j,j] * (b[lj,lj] - b[li,lj])
        # In term4: -a[j,j] * (b[lj,lj] - b[lj,li])
        
        # This is getting complex. Let me just use the O(n) loop which is fast enough for n=100.
        
        delta = 0.0
        # Direct interaction between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Vectorized for all k != i, j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        lks = p[ks]
        
        if len(ks) > 0:
            # a[i, ks] * (b[lj, lks] - b[li, lks])
            term1 = np.dot(a[i, ks], b[lj, lks] - b[li, lks])
            # a[ks, i] * (b[lks, lj] - b[lks, li])
            term2 = np.dot(a[ks, i], b[lks, lj] - b[lks, li])
            # a[j, ks] * (b[li, lks] - b[lj, lks])
            term3 = np.dot(a[j, ks], b[li, lks] - b[lj, lks])
            # a[ks, j] * (b[lks, li] - b[lks, lj])
            term4 = np.dot(a[ks, j], b[lks, li] - b[lks, lj])
            
            delta += term1 + term2 + term3 + term4
            
        return int(delta)
    
    def initial_solution():
        """Generate initial solution using rank matching"""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    def simulated_annealing(p, start_time, time_limit):
        """Run SA from initial permutation p"""
        current_cost = calc_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        # Adaptive SA parameters
        T = max(100.0, current_cost * 0.001)  # Scale with problem
        cooling_rate = 0.9995
        min_T = 0.5
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            dc = delta_cost_vectorized(i, j, p)
            
            if dc < 0:
                p[i], p[j] = p[j], p[i]
                current_cost += dc
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if T > min_T:
                    prob = np.exp(-dc / T)
                    if random.random() < prob:
                        p[i], p[j] = p[j], p[i]
                        current_cost += dc
                no_improve_count += 1
            
            T *= cooling_rate
            
            # Early stop if stuck
            if no_improve_count > max_no_improve and T < 10.0:
                break
        
        return best_p, best_cost
    
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-start SA
    best_overall_p = None
    best_overall_cost = float('inf')
    
    # Number of restarts based on time
    # Each SA run should take ~10-20 seconds
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Generate diverse initial solutions
        if restart == 0:
            p = initial_solution()
        else:
            # Perturb the best solution found so far
            if best_overall_p is not None:
                p = best_overall_p.copy()
                # Apply random swaps
                num_swaps = random.randint(5, 20)
                for _ in range(num_swaps):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
            else:
                p = initial_solution()
        
        # Time budget for this restart
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5.0:
            break
        
        # Allocate time: first restart gets more time
        if restart == 0:
            sa_time_limit = min(remaining_time * 0.5, 60.0)
        else:
            sa_time_limit = min(remaining_time * 0.8, 30.0)
        
        p_best, cost_best = simulated_annealing(p, time.time(), time.time() + sa_time_limit)
        
        if cost_best < best_overall_cost:
            best_overall_cost = cost_best
            best_overall_p = p_best
    
    if best_overall_p is None:
        best_overall_p = initial_solution()
    
    return best_overall_p.tolist()
