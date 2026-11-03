"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Hungarian Algorithm on Linear Relaxation
    # We use a strong linear relaxation to seed the replicas.
    try:
        c_mat = -(a @ b.T)
        row_ind, col_ind = linear_sum_assignment(c_mat)
        p_init = np.empty(n, dtype=np.int64)
        p_init[row_ind] = col_ind
    except Exception:
        # Fallback to random
        p_init = np.random.permutation(n)
    
    # 2. Parallel Tempering Setup
    # We maintain N_REPLICAS solutions (replicas) at different temperatures.
    # Lower T replicas are "cold" and accept only improvements (or very slight worsenings).
    # Higher T replicas are "hot" and accept worsenings more readily, allowing exploration.
    
    N_REPLICAS = 10
    # Temperatures should span from very cold (near 0) to very hot.
    # The energy scale of QAP can be large. We need to estimate a typical step size.
    # Let's compute the cost of the initial solution to gauge scale.
    
    def calc_cost(p):
        return int(np.sum(a * b[p][:, p]))
    
    init_cost = calc_cost(p_init)
    
    # Estimate typical delta magnitude.
    # A random swap changes cost by roughly O(n * max_flow * max_dist).
    # Let's sample a few random swaps to estimate the variance.
    deltas = []
    p_temp = p_init.copy()
    for _ in range(50):
        i, j = np.random.choice(n, 2, replace=False)
        p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
        c1 = calc_cost(p_temp)
        # Swap back
        p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
        deltas.append(abs(c1 - init_cost))
    
    typical_delta = np.mean(deltas) if len(deltas) > 0 else 1.0
    if typical_delta == 0:
        typical_delta = 1.0
        
    # Define temperature ladder.
    # T_min should be low enough to mostly accept improvements.
    # T_max should be high enough to accept most moves.
    # Let's use a geometric progression.
    T_min = typical_delta * 0.1
    T_max = typical_delta * 10.0
    temperatures = np.geomspace(T_min, T_max, N_REPLICAS)
    
    # Initialize replicas with the same initial solution (or slightly perturbed)
    replicas = [p_init.copy() for _ in range(N_REPLICAS)]
    costs = [init_cost] * N_REPLICAS
    
    # Track global best
    best_cost = init_cost
    best_p = p_init.copy()
    
    # Helper to update best
    def update_best(rep_idx):
        nonlocal best_cost, best_p
        if costs[rep_idx] < best_cost:
            best_cost = costs[rep_idx]
            best_p = replicas[rep_idx].copy()
            
    # Main Loop
    # Each iteration:
    # 1. For each replica, propose a move (swap two facilities).
    # 2. Accept/reject based on Metropolis criterion at that replica's temperature.
    # 3. After all replicas have moved, attempt exchanges between adjacent replicas.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # 1. Move Phase
        for r in range(N_REPLICAS):
            p = replicas[r]
            c_old = costs[r]
            T = temperatures[r]
            
            # Propose swap
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            c_new = calc_cost(p)
            delta = c_new - c_old
            
            # Metropolis Acceptance
            if delta < 0:
                # Improvement, always accept
                accept = True
            else:
                # Deterioration, accept with probability exp(-delta / T)
                # To avoid underflow for large delta/T, use log
                if delta / T > 50: # exp(-50) is tiny
                    accept = False
                else:
                    accept = (rng.random() < np.exp(-delta / T))
                    
            if accept:
                costs[r] = c_new
                update_best(r)
            else:
                # Revert swap
                p[i], p[j] = p[j], p[i]
                
        # 2. Exchange Phase
        # Attempt to swap states between adjacent replicas (r, r+1).
        # We iterate through adjacent pairs.
        # Standard Parallel Tempering: try to swap if it preserves detailed balance.
        # Probability of swap: min(1, exp(-(E_high - E_low) * (1/T_low - 1/T_high)))
        # Where E_high is energy of replica at higher T (r+1), E_low at lower T (r).
        # Note: Lower index r has lower T.
        
        # We can try exchanges in a random order or sequential. Sequential is fine.
        # To be efficient, we might not do all exchanges every iteration, but let's do a few.
        # For simplicity and robustness, let's try all adjacent pairs.
        
        for r in range(N_REPLICAS - 1):
            T1 = temperatures[r]
            T2 = temperatures[r+1]
            E1 = costs[r]
            E2 = costs[r+1]
            
            # We want to swap if it helps the lower energy state go to higher T and vice versa?
            # Actually, the standard rule is:
            # Swap (E1, T1) and (E2, T2) with prob min(1, exp(-(E2-E1)*(1/T1 - 1/T2)))
            # Wait, the derivation is:
            # P(swap) = min(1, exp( - (E2 - E1) * (1/T1 - 1/T2) ))
            # Let's check limits.
            # If E2 > E1 (higher energy at higher T index? No, T2 > T1).
            # If E2 > E1, then (E2-E1) > 0. (1/T1 - 1/T2) > 0 because T1 < T2.
            # So exponent is negative. Prob < 1.
            # This means we are less likely to swap if the "hot" replica has higher energy than the "cold" one?
            # Wait.
            # Replica r (cold, T1) has E1. Replica r+1 (hot, T2) has E2.
            # We want the cold replica to have low energy.
            # If E1 < E2, it's "good" that cold has low energy. Swapping would put low energy in hot, high in cold. This is bad for convergence?
            # Actually, Parallel Tempering works by allowing the high energy state to move to the low temperature (where it would be stuck) and the low energy state to move to the high temperature (where it can explore).
            # So we WANT to swap if E1 < E2?
            # Let's re-derive or check standard formula.
            # Standard: Swap configurations x1 (at T1) and x2 (at T2).
            # Acceptance prob: min(1, exp( -(E(x2)-E(x1)) * (1/T1 - 1/T2) ))
            # If E(x2) > E(x1) and T2 > T1:
            # 1/T1 > 1/T2, so (1/T1 - 1/T2) > 0.
            # Exponent is negative. Prob < 1.
            # This implies we are LESS likely to swap if the hotter replica has higher energy.
            # This seems counter-intuitive for "escaping local minima".
            # Let's think.
            # If Cold has Low E, Hot has High E. This is the ideal state. Swapping ruins it. So we should NOT swap. Correct.
            # If Cold has High E, Hot has Low E. This is bad. Cold is stuck in high E. Hot has found a low E but can't refine it?
            # We WANT to swap so Cold gets Low E and Hot gets High E (to explore).
            # In this case E1 (Cold) > E2 (Hot).
            # E2 - E1 < 0.
            # 1/T1 - 1/T2 > 0.
            # Product is negative.
            # Exponent is positive.
            # Prob > 1? No, min(1, ...) so Prob = 1.
            # So we ALWAYS swap if Cold has Higher E than Hot.
            # This makes sense! We move the good solution to the cold chain to refine, and the bad solution to the hot chain to explore.
            
            diff_E = E2 - E1
            diff_inv_T = (1.0 / T1) - (1.0 / T2)
            
            if diff_E * diff_inv_T < 0:
                # Exponent is positive, prob > 1, so always accept
                accept = True
            else:
                # Exponent is negative (or zero)
                # If diff_E * diff_inv_T is large positive, exp is small.
                val = - (diff_E * diff_inv_T)
                if val < -50:
                    accept = False
                else:
                    accept = (rng.random() < np.exp(val))
                    
            if accept:
                # Swap states
                replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                costs[r], costs[r+1] = costs[r+1], costs[r]
                update_best(r)
                update_best(r+1)

    return best_p.tolist()
# EVOLVE-BLOCK-END
