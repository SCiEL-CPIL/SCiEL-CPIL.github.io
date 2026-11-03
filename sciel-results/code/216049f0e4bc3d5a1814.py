"""Vectorized Parallel Simulated Annealing for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_vectorized(a_sym, b, p, i_indices, j_indices):
    """
    Compute cost deltas for a batch of swaps (i, j) simultaneously.
    a_sym = a + a.T
    delta = (a_sym[i,:] - a_sym[j,:]) . (b[p[j],:] - b[p[i],:]) 
            - (a_sym[i,i] - a_sym[j,j])*(b[p[j],p[j]] - b[p[i],p[i]]) 
            - (a_sym[i,j] - a_sym[j,i])*(b[p[j],p[i]] - b[p[i],p[j]])
            
    Wait, the standard delta formula for swapping positions i and j in permutation p:
    Let x = p[i], y = p[j].
    Cost change = sum_{k != i, j} (a[i,k] + a[j,k] - a[i,j] - a[j,i]) * (b[x, p[k]] + b[y, p[k]] - b[x, y] - b[y, x]) ... this is messy.
    
    Simpler delta formula using a_sym (symmetric flow):
    Let A = a_sym.
    Delta = (A[i, :] - A[j, :]) @ (b[y, :] - b[x, :]) 
            - A[i, i] * (b[y, y] - b[x, x]) 
            - A[j, j] * (b[y, x] - b[x, y])
            - A[i, j] * (b[y, x] - b[x, y])
            - A[j, i] * (b[y, y] - b[x, x])
            
    Actually, let's derive it carefully or use a known vectorized form.
    Let v_i = A[i, :] - A[j, :]
    Let w_j = b[y, :] - b[x, :]
    
    The interaction term for k != i, j is v_i[k] * w_j[p[k]].
    For k = i: v_i[i] * w_j[y] = (A[i,i] - A[j,i]) * (b[y,y] - b[x,y])
    For k = j: v_i[j] * w_j[x] = (A[i,j] - A[j,j]) * (b[y,x] - b[x,x])
    
    Sum = v_i @ w_j[p] - v_i[i]*w_j[y] - v_i[j]*w_j[x]
    Note: w_j[p] is the vector w_j indexed by the permutation p.
    
    Let's verify indices.
    p is the permutation of facilities to locations.
    Cost = sum_{i,j} A[i,j] * b[p[i], p[j]]
    
    Swap i and j in p.
    New p': p'[i]=y, p'[j]=x, p'[k]=p[k] for k!=i,j.
    
    Change in cost:
    Terms involving i or j in the first index of A:
    Sum_{k} (A[i,k] + A[j,k] - A[i,j] - A[j,i]) * (b[p'[i], p'[k]] + b[p'[j], p'[k]] - b[p[i], p[k]] - b[p[j], p[k]])
    
    This is complex. Let's use the efficient delta formula:
    delta = (A[i, :] - A[j, :]) @ (b[y, :] - b[x, :]) 
            - (A[i, i] - A[j, j]) * (b[y, y] - b[x, x]) 
            - (A[i, j] - A[j, i]) * (b[y, x] - b[x, y])
    
    Wait, A is symmetric, so A[i,j]=A[j,i].
    delta = (A[i, :] - A[j, :]) @ (b[y, :] - b[x, :]) 
            - (A[i, i] - A[j, j]) * (b[y, y] - b[x, x]) 
            - 2 * A[i, j] * (b[y, x] - b[x, y])
    
    Let's check this.
    Let u = A[i, :] - A[j, :]
    Let v = b[y, :] - b[x, :]
    
    u @ v = sum_k u[k] v[k]
    = sum_k (A[i,k] - A[j,k]) (b[y,k] - b[x,k])
    
    This sums over all k.
    The actual change in cost only involves k where p[k] changes or the indices i,j change.
    
    Standard efficient delta:
    delta = (A[i, :] - A[j, :]) @ (b[y, :] - b[x, :]) 
            - (A[i, i] - A[j, j]) * (b[y, y] - b[x, x])
            - (A[i, j] - A[j, i]) * (b[y, x] - b[x, y])
    
    Since A is symmetric:
    delta = (A[i, :] - A[j, :]) @ (b[y, :] - b[x, :]) 
            - (A[i, i] - A[j, j]) * (b[y, y] - b[x, x])
            - 2 * A[i, j] * (b[y, x] - b[x, y])
    
    This formula is O(n) per swap if computed naively, but we can vectorize the dot product part.
    However, b[y, :] - b[x, :] is a vector of size n.
    A[i, :] - A[j, :] is a vector of size n.
    The dot product is O(n).
    If we do this for M swaps, it is O(M*n).
    With M=1000 and n=256, that's 256k ops. Very fast.
    
    Let's implement this vectorized delta.
    """
    x = p[i_indices]
    y = p[j_indices]
    
    # u = A[i, :] - A[j, :]  (M, n)
    # v = b[y, :] - b[x, :]  (M, n)
    # dot = u @ v.T? No, u is (M, n), v is (M, n). We want sum over k for each row.
    # np.sum(u * v, axis=1)
    
    u = a_sym[i_indices, :] - a_sym[j_indices, :]
    v = b[y, :] - b[x, :]
    
    dot_products = np.sum(u * v, axis=1)
    
    # Correction terms
    # term1 = (A[i, i] - A[j, j]) * (b[y, y] - b[x, x])
    a_ii = a_sym[i_indices, i_indices]
    a_jj = a_sym[j_indices, j_indices]
    b_yy = b[y, y]
    b_xx = b[x, x]
    term1 = (a_ii - a_jj) * (b_yy - b_xx)
    
    # term2 = 2 * A[i, j] * (b[y, x] - b[x, y])
    a_ij = a_sym[i_indices, j_indices]
    b_yx = b[y, x]
    b_xy = b[x, y]
    term2 = 2 * a_ij * (b_yx - b_xy)
    
    deltas = dot_products - term1 - term2
    
    return deltas

def _sa_trajectory(a, b, p_init, time_limit, start_time, seed):
    """Run a single Simulated Annealing trajectory."""
    n = len(a)
    rng = random.Random(seed)
    a_sym = a + a.T
    
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # SA Parameters
    # Start temperature high enough to accept bad moves, cool down slowly
    # Heuristic: T0 ~ 10 * max_flow * max_dist / n ?
    # Let's estimate typical delta magnitude.
    # max_a = max(a), max_b = max(b)
    # delta ~ n * max_a * max_b
    # Let's just use a fixed schedule or adaptive.
    
    T0 = 1000.0
    Tend = 1.0
    # Number of steps to complete cooling
    # We don't know how many steps we can take, so use time-based cooling.
    # Or simply use a fixed number of steps if we can estimate speed.
    # Let's say we can do ~5000 steps in 10s? 
    # Vectorized delta for 1000 swaps takes ~1ms. 1000 swaps per step.
    # So ~1000 steps per second.
    # Let's target ~5000 steps total.
    
    total_steps = 5000
    # Geometric cooling
    # T_k = T0 * (Tend/T0)^(k/total_steps)
    # To avoid log(0), use linear or geometric with min.
    
    for step in range(total_steps):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Generate M random swaps
        M = 500
        i_idx = rng.choices(range(n), k=M)
        j_idx = rng.choices(range(n), k=M)
        
        # Ensure i != j
        mask = np.array([i_idx[k] != j_idx[k] for k in range(M)])
        if not np.any(mask):
            continue
        i_idx = np.array(i_idx)[mask]
        j_idx = np.array(j_idx)[mask]
        
        if len(i_idx) == 0:
            continue
            
        deltas = _compute_delta_vectorized(a_sym, b, p, i_idx, j_idx)
        
        # Temperature for this step
        progress = step / total_steps
        T = T0 * (Tend / T0) ** progress
        
        # Acceptance criteria
        # Accept if delta < 0 or exp(-delta/T) > random
        # Vectorized acceptance
        # For delta < 0, always accept.
        # For delta >= 0, accept with prob exp(-delta/T)
        
        # To avoid calculating exp for all, we can just do it.
        # exp is fast in numpy.
        
        # However, we need to apply the swaps.
        # We should apply them one by one or in a way that doesn't conflict?
        # If we apply all M swaps simultaneously, they might interfere if they share indices.
        # Simplest: apply them sequentially but use the pre-computed deltas?
        # No, deltas are valid only for the current p. If we apply one, p changes, deltas for others are invalid.
        
        # Strategy: Pick the BEST swap from the M candidates.
        # This is "Greedy SA" or "Best-of-M".
        # It reduces the number of state changes to 1 per step, but uses M evaluations.
        # This is much safer and often effective.
        
        min_delta_idx = np.argmin(deltas)
        min_delta = deltas[min_delta_idx]
        i_move = i_idx[min_delta_idx]
        j_move = j_idx[min_delta_idx]
        
        # Acceptance
        if min_delta < 0:
            accept = True
        else:
            # Prob = exp(-min_delta / T)
            # If T is very small, prob is 0.
            # Use log for stability?
            # log(prob) = -min_delta / T
            # Compare with log(random)
            if T > 1e-9:
                log_prob = -min_delta / T
                log_rand = np.log(rng.random())
                accept = log_prob > log_rand
            else:
                accept = False
                
        if accept:
            p[i_move], p[j_move] = p[j_move], p[i_move]
            current_cost += min_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p, best_cost

def _greedy_init(a, b, seed=42):
    """Greedy rank-matching initialization."""
    n = len(a)
    rng = random.Random(seed)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.copy()

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Run multiple SA trajectories in parallel (sequentially in time, but independent)
    # We can run 5 trajectories.
    K = 5
    time_per_traj = time_limit / K
    
    for k in range(K):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        # Initialization
        if k == 0:
            p_init = _greedy_init(a, b, seed=42 + k)
        else:
            # Perturbed greedy
            p_init = _greedy_init(a, b, seed=42 + k)
            rng = random.Random(999 + k)
            for _ in range(n // 4):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
                    
        # Run SA
        p_best, cost_best = _sa_trajectory(a, b, p_init.copy(), time_per_traj, start_time, seed=1000 + k)
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
            
    if best_p is None:
        best_p = _greedy_init(a, b, seed=0)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
