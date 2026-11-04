"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    t0 = time.time()
    time_limit = 115.0
    
    # 1. Greedy Construction
    # Sort facilities by total flow (in + out) descending
    flow_sums = a.sum(axis=1) + a.sum(axis=0)
    fac_order = np.argsort(-flow_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Precompute location distance sums for initial attraction
    dist_sums = b.sum(axis=1)
    
    # We will place facilities one by one.
    # To make this faster and better, we evaluate the incremental cost of placing facility f at location l
    # given current placement p.
    # inc_cost(f, l) = sum_{g already placed} a[f, g] * b[l, p[g]] + a[g, f] * b[p[g], l]
    
    # Maintain row/col sums of the "interaction" to speed up? 
    # For n=100, O(n^2) per placement is 100*100*100 = 1e6 ops, which is fast enough in Python/NumPy if vectorized.
    
    for idx in range(n):
        f = fac_order[idx]
        if idx == 0:
            # Place first facility at location with min distance sum? Or just 0?
            # Better: place at location that minimizes potential future conflict?
            # Simple heuristic: place at location with smallest distance sum (central)
            loc = np.argmin(dist_sums)
        else:
            # Compute incremental cost for all unused locations
            # placed_facilities = fac_order[:idx]
            # placed_locs = p[fac_order[:idx]]
            # We need sum over k in placed: a[f, k] * b[l, p[k]] + a[k, f] * b[p[k], l]
            
            # Let's vectorize over l
            # b_l_k = b[:, p[placed]] -> shape (n, idx)
            # a_f_k = a[f, fac_order[:idx]] -> shape (idx,)
            # term1 = a_f_k @ b[:, p[placed]] -> shape (n,)
            # term2 = (a[fac_order[:idx], f] @ b[p[placed], :]) -> shape (n,)
            
            placed_facs = fac_order[:idx]
            placed_locs = p[placed_facs]
            
            # b[:, placed_locs] is (n, idx)
            # a[f, placed_facs] is (idx,)
            cost1 = a[f, placed_facs] @ b[:, placed_locs]
            
            # b[placed_locs, :] is (idx, n)
            # a[placed_facs, f] is (idx,)
            cost2 = a[placed_facs, f] @ b[placed_locs, :]
            
            total_inc = cost1 + cost2
            
            # Mask used locations
            total_inc[used] = np.iinfo(np.int64).max
            loc = np.argmin(total_inc)
            
        p[f] = loc
        used[loc] = True

    # 2. Local Search Initialization (2-opt)
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute delta components for 2-opt
    # For swap i, j:
    # delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]]) + (a[k,i]-a[k,j])*(b[p[k],p[i]]-b[p[k],p[j]])
    #       + (a[i,j]-a[j,i])*(b[p[i],p[j]]-b[p[j],p[i]])
    
    # To speed up, we can precompute row/col sums of a and b interactions?
    # Actually, for n=100, direct calculation is ~100 ops per swap.
    # 100*100/2 = 5000 swaps. 5000 * 100 = 500k ops. Very fast.
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # diff_a_row = a[i, :] - a[j, :]
        # diff_b_col = b[u, :] - b[v, :]  # Note: b[u, p[k]] - b[v, p[k]]
        # term1 = sum(diff_a_row * (b[u, p] - b[v, p]))
        
        # diff_a_col = a[:, i] - a[:, j]
        # diff_b_row = b[:, u] - b[:, v] # Note: b[p[k], u] - b[p[k], v]
        # term2 = sum(diff_a_col * (b[p, u] - b[p, v]))
        
        # term3 = (a[i, j] - a[j, i]) * (b[u, v] - b[v, u])
        
        # Subtract self terms i and j
        
        diff_a_row = a[i, :] - a[j, :]
        b_u_p = b[u, p]
        b_v_p = b[v, p]
        term1 = np.dot(diff_a_row, b_u_p - b_v_p)
        
        diff_a_col = a[:, i] - a[:, j]
        b_p_u = b[p, u]
        b_p_v = b[p, v]
        term2 = np.dot(diff_a_col, b_p_u - b_p_v)
        
        term3 = (a[i, j] - a[j, i]) * (b[u, v] - b[v, u])
        
        # Remove k=i and k=j contributions
        # k=i: p[i]=u
        # term1_i = diff_a_row[i] * (b[u, u] - b[v, u])
        # term2_i = diff_a_col[i] * (b[u, u] - b[u, v])
        
        # k=j: p[j]=v
        # term1_j = diff_a_row[j] * (b[u, v] - b[v, v])
        # term2_j = diff_a_col[j] * (b[v, u] - b[v, v])
        
        term1 -= diff_a_row[i] * (b[u, u] - b[v, u])
        term1 -= diff_a_row[j] * (b[u, v] - b[v, v])
        
        term2 -= diff_a_col[i] * (b[u, u] - b[u, v])
        term2 -= diff_a_col[j] * (b[v, u] - b[v, v])
        
        return term1 + term2 + term3

    # Tabu Search
    tabu_list = {} # (i, j) -> iteration until which it is tabu
    tabu_tenure = 10
    max_tabu_tenure = 20
    no_improve = 0
    iter_count = 0
    
    rng = np.random.default_rng(42)
    
    while time.time() - t0 < time_limit:
        # Generate all 2-opt moves
        best_delta = np.iinfo(np.int64).max
        best_move = None
        best_is_tabu = True
        
        # Sample a subset of moves to save time? Or all?
        # n=100, 4950 moves. 4950 * delta_calc(~100 ops) = ~500k ops.
        # In Python loop, this might be slow. Let's vectorize the move generation?
        # Hard to vectorize delta for all pairs at once without huge memory.
        # Let's just do a random sample of moves if time is tight, or all if fast enough.
        # Let's try all first. If too slow, switch to random sampling.
        
        # Optimization: Precompute b[u, p] and b[p, u] for all u?
        # b is (n,n), p is (n,). b[:, p] is (n,n).
        # Let's precompute Bp = b[:, p] (shape n,n) and BpT = b[p, :] (shape n,n)
        # Then b[u, p] is Bp[u, :], b[p, u] is BpT[:, u]
        
        Bp = b[:, p]
        BpT = b[p, :]
        
        # We will iterate i, j
        # To speed up, we can precompute a[i, :] - a[j, :] ? No, depends on i,j.
        
        # Let's just do a random subset of 500 moves per iteration to keep it fast.
        # This makes it a stochastic local search with tabu.
        
        moves_sampled = 1000
        for _ in range(moves_sampled):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            
            delta = get_delta(p, i, j)
            
            is_tabu = False
            if (i, j) in tabu_list:
                if tabu_list[(i, j)] > iter_count:
                    is_tabu = True
            if (j, i) in tabu_list:
                if tabu_list[(j, i)] > iter_count:
                    is_tabu = True
            
            # Aspiration criterion: accept if it improves global best
            if delta < best_cost or not is_tabu:
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    best_is_tabu = is_tabu
        
        if best_move is None:
            # No valid moves found in sample? Unlikely.
            break
        
        i, j = best_move
        u, v = p[i], p[j]
        
        # Perform move
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update Tabu List
        tabu_list[(i, j)] = iter_count + tabu_tenure
        tabu_list[(j, i)] = iter_count + tabu_tenure
        
        # Clean up old tabu entries
        if len(tabu_list) > 5000:
            keys_to_del = [k for k, v in tabu_list.items() if v <= iter_count]
            for k in keys_to_del:
                del tabu_list[k]
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
            # Increase tabu tenure slightly on improvement?
            tabu_tenure = min(max_tabu_tenure, tabu_tenure + 1)
        else:
            no_improve += 1
            # Decrease tabu tenure to allow more flexibility
            tabu_tenure = max(5, tabu_tenure - 1)
            
            # Diversification
            if no_improve > 1000:
                # Random restart or shake
                # Shake: perform 10 random swaps
                for _ in range(10):
                    i = rng.integers(n)
                    j = rng.integers(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                no_improve = 0
                tabu_list.clear()
                
        iter_count += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
