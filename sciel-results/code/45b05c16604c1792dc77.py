"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))

def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    Optimized for small n (100).
    """
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    # Create mask excluding i and j
    # Using np.ones and indexing is standard. 
    # For n=100, this is fast enough.
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # Vectorized delta calculation
    # Terms involving k not in {i, j}
    # Sum over k != i, j of a[i,k]*(b[pj, p[k]] - b[pi, p[k]]) + a[j,k]*(b[pi, p[k]] - b[pj, p[k]])
    # + a[k,i]*(b[p[k], pj] - b[p[k], pi]) + a[k,j]*(b[p[k], pi] - b[p[k], pj])
    
    # Let's group terms to minimize numpy operations
    # Row i contribution: a[i, mask] * (b[pj, p[mask]] - b[pi, p[mask]])
    # Row j contribution: a[j, mask] * (b[pi, p[mask]] - b[pj, p[mask]])
    # Col i contribution: a[mask, i] * (b[p[mask], pj] - b[p[mask], pi])
    # Col j contribution: a[mask, j] * (b[p[mask], pi] - b[p[mask], pj])
    
    # Note: b is symmetric in QAP usually? 
    # The task says "a and b are this task's QAPLIB matrices". 
    # QAPLIB instances typically have symmetric A and B. 
    # If B is symmetric, b[pj, p[k]] == b[p[k], pj].
    # Let's assume symmetry for potential optimization, but keep general form to be safe.
    # However, wil100 has symmetric matrices.
    
    # If B is symmetric:
    # b[pj, p[mask]] - b[pi, p[mask]] is the same as b[p[mask], pj] - b[p[mask], pi]
    # So we can combine row and col terms if A is symmetric? 
    # A is also symmetric in QAP.
    # If A and B are symmetric, delta = sum_{k != i,j} (a[i,k] + a[k,i]) * (b[pj, p[k]] - b[pi, p[k]]) ... 
    # Wait, a[i,k] = a[k,i]. So (a[i,k] + a[k,i]) = 2*a[i,k].
    # And b[pj, p[k]] - b[pi, p[k]] is the distance difference.
    # So delta_inter = 2 * sum_{k != i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]]) 
    #                  + 2 * sum_{k != i,j} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
    #                  = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    
    # Let's verify this simplification.
    # Original:
    # sum_k a[i,k](b[v,p_k]-b[u,p_k]) + a[j,k](b[u,p_k]-b[v,p_k]) 
    # + a[k,i](b[p_k,v]-b[p_k,u]) + a[k,j](b[p_k,u]-b[p_k,v])
    # If A sym: a[i,k]=a[k,i], a[j,k]=a[k,j].
    # If B sym: b[v,p_k]=b[p_k,v], b[u,p_k]=b[p_k,u].
    # Term 1 + Term 3: a[i,k](b[v,p_k]-b[u,p_k]) + a[i,k](b[p_k,v]-b[p_k,u]) 
    #                  = a[i,k](b[v,p_k]-b[u,p_k]) + a[i,k](b[v,p_k]-b[u,p_k]) 
    #                  = 2*a[i,k](b[v,p_k]-b[u,p_k])
    # Term 2 + Term 4: a[j,k](b[u,p_k]-b[v,p_k]) + a[j,k](b[p_k,u]-b[p_k,v])
    #                  = a[j,k](b[u,p_k]-b[v,p_k]) + a[j,k](b[u,p_k]-b[v,p_k])
    #                  = 2*a[j,k](b[u,p_k]-b[v,p_k])
    #                  = -2*a[j,k](b[v,p_k]-b[u,p_k])
    # Sum = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v,p_k] - b[u,p_k])
    
    # This is much faster! One vector subtraction, one vector multiplication, one sum.
    # Plus the self terms:
    # a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])
    # If B sym: b[v,u]=b[u,v], so first two terms are 0.
    # Self terms: a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])
    #            = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
    
    # So if A and B are symmetric:
    # delta = 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v,p_k] - b[u,p_k]) + (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
    
    # Let's check if wil100 has symmetric matrices. 
    # Yes, standard QAPLIB instances have symmetric flow and distance matrices.
    # We will assume symmetry for this optimization, but add a check or fallback if needed.
    # Given the context, it's safe to assume symmetry for wil100.
    
    # Calculate inter-part
    # a_diff = a[i, mask] - a[j, mask]
    # b_diff = b[pj, p[mask]] - b[pi, p[mask]]
    # delta_inter = 2.0 * np.dot(a_diff, b_diff)
    
    # Calculate self-part
    # delta_self = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    
    # Total
    # return int(delta_inter + delta_self)
    
    # Let's implement this optimized version.
    
    a_diff = a[i, mask] - a[j, mask]
    b_diff = b[pj, p[mask]] - b[pi, p[mask]]
    
    delta_inter = 2.0 * np.dot(a_diff, b_diff)
    
    delta_self = (a[i, i] - a[j, j]) * (b[pj, pj] - b[pi, pi])
    
    return int(delta_inter + delta_self)

def _local_search(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    """
    n = len(p)
    cost = _cost(a, b, p)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        if not improved:
            break
    
    return p, cost

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    # Check symmetry to decide on optimized delta
    # For wil100, it is symmetric.
    # We will use the optimized delta function derived above.
    
    # --- Phase 1: Initialization ---
    best_p = None
    best_cost = np.inf
    
    # Heuristic 1: Rank-matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Heuristic 2: Reverse
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # --- Phase 2: Hybrid Simulated Annealing + Local Search ---
    # This phase explores the landscape more broadly than simple ILS.
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    rng = np.random.default_rng(42)
    
    # SA Parameters
    T = 1000.0  # Initial temperature
    T_min = 1.0
    cooling_rate = 0.995
    sa_deadline = deadline - 10.0  # Reserve time for final polish
    
    # We will interleave SA steps with periodic local searches.
    # Every `ls_interval` SA steps, run a local search.
    # This keeps the solution "close" to a local minimum while allowing exploration.
    
    sa_steps_since_ls = 0
    ls_interval = 50  # Run LS every 50 SA steps
    
    while time.time() < sa_deadline:
        # Pick a random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        
        if i == j:
            continue
            
        delta = _delta_swap(a, b, current_p, i, j)
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
            sa_steps_since_ls += 1
            if sa_steps_since_ls >= ls_interval:
                # Run local search to polish
                current_p, current_cost = _local_search(a, b, current_p, sa_deadline)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                sa_steps_since_ls = 0
        else:
            # Accept with probability exp(-delta/T)
            if rng.random() < math.exp(-delta / T):
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost += delta
                sa_steps_since_ls += 1
                if sa_steps_since_ls >= ls_interval:
                    current_p, current_cost = _local_search(a, b, current_p, sa_deadline)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                    sa_steps_since_ls = 0
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If T is very low, we are effectively doing random restarts or ILS.
        # We can increase cooling rate if stuck, but fixed is okay for now.
        
    # --- Phase 3: Final Polish ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        # Run a few more local search passes
        polish_p, polish_cost = _local_search(a, b, polish_p, polish_deadline)
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
