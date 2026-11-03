"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p):
        """Compute cost delta when swapping positions of facilities i and j."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        delta = 0
        
        # Diagonal terms
        delta += a[i,i] * (b[lj,lj] - b[li,li])
        delta += a[j,j] * (b[li,li] - b[lj,lj])
        
        # Cross terms between i and j
        delta += a[i,j] * (b[lj,li] - b[li,lj])
        delta += a[j,i] * (b[li,lj] - b[lj,li])
        
        # Terms with other facilities k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            
            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))
        
        return delta

    def simulated_annealing(p, best_cost, end_time):
        """Perform Simulated Annealing."""
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        # Estimate initial temperature
        # Sample a few random deltas to gauge scale
        rng = np.random.RandomState(42)
        deltas = []
        for _ in range(10):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 10.0
        cooling_rate = 0.999
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. Hungarian Initialization
    # Formulate as Minimum Weight Perfect Matching (Assignment Problem)
    # Cost matrix C derived from linearized objective
    C = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    best_p = p_hung.copy()
    best_cost = compute_cost_fast(p_hung)
    
    # 2. Simulated Annealing from Hungarian solution
    end_time = deadline - 5.0
    p_sa, cost_sa = simulated_annealing(best_p, best_cost, end_time)
    
    if cost_sa < best_cost:
        best_p = p_sa
        best_cost = cost_sa
        
    # 3. If time remains, try another initialization with different cost matrix
    # e.g. using row sums interaction
    if time.time() < deadline - 10:
        # Alternative linearization: C_alt = (a.sum(axis=1)[:, None] * b.sum(axis=1)[None, :])
        # This is a rank-1 matrix, might be weak, but let's try a different one:
        # C_alt = a @ b (if b is symmetric, same as b.T)
        # Let's try C_alt = a.T @ b
        C_alt = a.T @ b
        row_ind_alt, col_ind_alt = linear_sum_assignment(-C_alt)
        p_alt = np.empty(n, dtype=np.int64)
        p_alt[row_ind_alt] = col_ind_alt
        
        cost_alt = compute_cost_fast(p_alt)
        
        if cost_alt < best_cost:
            best_p = p_alt.copy()
            best_cost = cost_alt
        
        # SA from this new start
        end_time_alt = deadline - 2.0
        p_sa_alt, cost_sa_alt = simulated_annealing(best_p, best_cost, end_time_alt)
        
        if cost_sa_alt < best_cost:
            best_p = p_sa_alt
            best_cost = cost_sa_alt
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
