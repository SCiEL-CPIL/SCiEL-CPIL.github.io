"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Precompute row sums for initial heuristic
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Helper to calculate cost
    def calc_cost(p):
        return int(np.sum(a * b[p]))

    # Calculate initial cost
    cost = calc_cost(p)
    best_cost = cost
    best_p = p.copy()

    # Helper to get delta for swapping i and j
    # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + 2*a[i,j]*b[u,v]
    # where u = p[i], v = p[j]
    # Note: The term for k=i and k=j in the sum above is 0 in the "sum over all k" formulation 
    # if we use the full dot product, but we must be careful.
    # Let's derive:
    # C_new - C_old = sum_{k,l} a[k,l] b[p_new[k], p_new[l]] - sum_{k,l} a[k,l] b[p_old[k], p_old[l]]
    # Only terms involving i or j change.
    # Terms:
    # 1. k=i, l=i: a[i,i]b[p[i],p[i]] -> a[i,i]b[p[j],p[j]] (no change in b index for self? No, p[i] becomes p[j])
    #    Wait, p is the assignment. p[i] is the location of facility i.
    #    If we swap locations of i and j:
    #    p_new[i] = p_old[j] = v
    #    p_new[j] = p_old[i] = u
    #    p_new[k] = p_old[k] for k != i,j
    
    # Cost contribution of i in old: sum_l a[i,l] b[u, p[l]]
    # Cost contribution of i in new: sum_l a[i,l] b[v, p_new[l]]
    #   = a[i,i]b[v,v] + a[i,j]b[v,u] + sum_{l!=i,j} a[i,l]b[v, p[l]]
    
    # Cost contribution of j in old: sum_l a[j,l] b[v, p[l]]
    # Cost contribution of j in new: sum_l a[j,l] b[u, p_new[l]]
    #   = a[j,i]b[u,v] + a[j,j]b[u,u] + sum_{l!=i,j} a[j,l]b[u, p[l]]
    
    # Contributions of other k to i and j:
    # Old: sum_{k!=i,j} a[k,i]b[p[k],u] + a[k,j]b[p[k],v]
    # New: sum_{k!=i,j} a[k,i]b[p[k],v] + a[k,j]b[p[k],u]
    
    # Total Delta = (New_i + New_j + New_others) - (Old_i + Old_j + Old_others)
    
    # Let's group terms:
    # Terms involving b[., u] and b[., v]
    
    # From i's row:
    # a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + sum_{l!=i,j} a[i,l](b[v,p[l]]-b[u,p[l]])
    
    # From j's row:
    # a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v]) + sum_{l!=i,j} a[j,l](b[u,p[l]]-b[v,p[l]])
    
    # From others' columns:
    # sum_{k!=i,j} a[k,i](b[p[k],v]-b[p[k],u]) + a[k,j](b[p[k],u]-b[p[k],v])
    # = sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],v]-b[p[k],u])
    
    # Combine sums over l!=i,j:
    # sum_{l!=i,j} (a[i,l]-a[j,l]) * (b[v,p[l]]-b[u,p[l]])
    
    # So Delta = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[j,j](b[u,u]-b[v,v])
    #           + sum_{l!=i,j} (a[i,l]-a[j,l]) * (b[v,p[l]]-b[u,p[l]])
    
    # Note that a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) = (a[i,j]-a[j,i])(b[v,u]-b[u,v])
    # And a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]) = (a[i,i]-a[j,j])(b[v,v]-b[u,u])
    
    # The sum term can be extended to all l if we handle l=i and l=j carefully or just compute full dot product
    # and subtract the i,j terms if the formula holds for all k.
    # Let's check if (a[i,l]-a[j,l]) * (b[v,p[l]]-b[u,p[l]]) works for l=i and l=j.
    # For l=i: p[i]=u. Term is (a[i,i]-a[j,i]) * (b[v,u]-b[u,u]).
    # Our derived term for l=i in sum was NOT included in sum_{l!=i,j}.
    # The explicit terms were:
    # a[i,i](b[v,v]-b[u,u]) ...
    
    # It is safer to use the full dot product formula which is often cited:
    # delta = (a[i] - a[j]) @ (b[v, p] - b[u, p]) + 2 * a[i, j] * b[u, v]
    # Let's verify this.
    # (a[i] - a[j]) @ (b[v, p] - b[u, p]) = sum_k (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]])
    # = sum_{k!=i,j} ... + (a[i,i]-a[j,i])(b[v,u]-b[u,u]) + (a[i,j]-a[j,j])(b[v,v]-b[u,v])
    
    # We want to match:
    # sum_{k!=i,j} (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]]) + (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + (a[i,j]-a[j,i])(b[v,u]-b[u,v])
    
    # Let's expand the dot product term for k=i and k=j:
    # k=i: (a[i,i]-a[j,i])(b[v,u]-b[u,u]) = a[i,i]b[v,u] - a[i,i]b[u,u] - a[j,i]b[v,u] + a[j,i]b[u,u]
    # k=j: (a[i,j]-a[j,j])(b[v,v]-b[u,v]) = a[i,j]b[v,v] - a[i,j]b[u,v] - a[j,j]b[v,v] + a[j,j]b[u,v]
    
    # Sum of these two:
    # a[i,i]b[v,u] - a[i,i]b[u,u] - a[j,i]b[v,u] + a[j,i]b[u,u] + a[i,j]b[v,v] - a[i,j]b[u,v] - a[j,j]b[v,v] + a[j,j]b[u,v]
    
    # Now look at the target terms:
    # (a[i,i]-a[j,j])(b[v,v]-b[u,u]) = a[i,i]b[v,v] - a[i,i]b[u,u] - a[j,j]b[v,v] + a[j,j]b[u,u]
    # (a[i,j]-a[j,i])(b[v,u]-b[u,v]) = a[i,j]b[v,u] - a[i,j]b[u,v] - a[j,i]b[v,u] + a[j,i]b[u,v]
    
    # Sum of target terms:
    # a[i,i]b[v,v] - a[i,i]b[u,u] - a[j,j]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] - a[i,j]b[u,v] - a[j,i]b[v,u] + a[j,i]b[u,v]
    
    # Comparing Dot(i,j) vs Target(i,j):
    # Dot has: a[i,i]b[v,u] ... Target has: ... a[i,j]b[v,u]
    # They are different.
    
    # However, there is a known identity for QAP swap delta:
    # delta = 2 * a[i,j] * b[u,v] + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # This is equivalent to:
    # delta = (a[i] - a[j]) @ (b[v, p] - b[u, p]) + 2 * a[i, j] * b[u, v]
    # IF the matrices are symmetric? Or is it always true?
    # Let's check the difference.
    # Dot(i,j) - Target(i,j) = 
    # (a[i,i]b[v,u] - a[j,i]b[v,u] + a[j,i]b[u,u] - a[j,j]b[u,u] + a[i,j]b[v,v] - a[j,j]b[v,v] + a[j,j]b[u,v] - a[i,j]b[u,v])
    # - (a[i,j]b[v,u] - a[j,i]b[v,u] + a[j,i]b[u,v] - a[i,j]b[u,v] + a[i,i]b[v,v] - a[i,i]b[u,u] - a[j,j]b[v,v] + a[j,j]b[u,u])
    # Wait, I'm getting confused. Let's just use the explicit sum for k != i, j.
    # For n=150, summing 148 terms is fast.
    
    def get_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        
        # diff_a = a[i] - a[j]
        # diff_b = b[v, p] - b[u, p]
        # We need sum_{k != i, j} diff_a[k] * diff_b[k]
        
        # Vectorized approach:
        # Create a mask or just exclude i and j.
        # Since n is small, we can do:
        # total_dot = np.dot(a[i] - a[j], b[v, p] - b[u, p])
        # Subtract terms for k=i and k=j.
        
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        
        total = np.dot(diff_a, diff_b)
        
        # Term k=i:
        # diff_a[i] * diff_b[i] = (a[i,i]-a[j,i]) * (b[v,u]-b[u,u])
        # Term k=j:
        # diff_a[j] * diff_b[j] = (a[i,j]-a[j,j]) * (b[v,v]-b[u,v])
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k_neq = total - term_i - term_j
        
        # Add the cross term 2 * a[i,j] * b[u,v]
        # Wait, is it 2*a[i,j]*b[u,v] or something else?
        # Let's re-verify the standard formula.
        # Standard formula: delta = sum_{k,l} (a[i,k]b[v,p[l]] + a[k,i]b[p[k],v] - a[i,k]b[u,p[l]] - a[k,i]b[p[k],u]) ... no.
        
        # Let's trust the derivation:
        # Delta = sum_{k!=i,j} (a[i,k]-a[j,k])(b[v,p[k]]-b[u,p[k]]) + (a[i,i]-a[j,j])(b[v,v]-b[u,u]) + (a[i,j]-a[j,i])(b[v,u]-b[u,v])
        
        # Let's compute this directly to be safe.
        # It's only 3 extra operations.
        
        part_self = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
        part_cross = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        delta = sum_k_neq + part_self + part_cross
        
        return delta

    # Main Loop: Tabu Search with 2-Opt Swaps
    tabu_size = 30
    tabu_list = {}  # (i, j) -> expiry step
    current_step = 0
    no_improve_count = 0
    max_no_improve = 2000
    
    # Precompute Bp = b[:, p] to speed up delta calculation?
    # b[v, p] is a row of b indexed by p.
    # If we maintain Bp where Bp[k, l] = b[p[k], l]? No.
    # b[v, p] is vector.
    # Let's just use b[v, p] directly. It's a gather operation.
    
    while time.time() < deadline:
        # Check for restart
        if no_improve_count > max_no_improve:
            # Perturb: swap 10 random pairs
            for _ in range(10):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            cost = calc_cost(p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_list.clear()
            continue

        # Find best non-tabu move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                delta = get_delta(i, j, p, a, b)
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            # No non-tabu moves or all bad?
            # If no non-tabu moves, clear tabu
            if tabu_list:
                tabu_list.clear()
                continue
            else:
                # Local optimum
                break

        # Perform swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += int(best_delta)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update tabu
        tabu_list[(best_i, best_j)] = current_step + tabu_size
        current_step += 1
        
        # Periodic cleanup
        if current_step % 100 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()

# EVOLVE-BLOCK-END
