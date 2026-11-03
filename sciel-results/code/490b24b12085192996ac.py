"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 118.0  # Safety margin before 120s kill

    def calc_cost(p):
        # p is permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        """
        u = p[i]
        v = p[j]
        
        # Vectorized calculation
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        # Total sum over all k
        sum_all = np.sum(diff_a * diff_b)
        
        # Subtract terms for k=i and k=j
        # Term k=i: diff_a[i] * diff_b[i]
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Term k=j: diff_a[j] * diff_b[j]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost):
        # First Improvement Local Search with vectorized pair evaluation
        # We evaluate all pairs for a given i, find best improving move, apply it, repeat.
        
        while True:
            if time.time() - start_time > timeout:
                break
            
            moved = False
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                
                # Compute deltas for swapping i with all j > i
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    # Swap i and best_j
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    moved = True
                    break # Restart scan from i=0
            
            if not moved:
                break
        
        return p, current_cost

    def lap_perturb(p, current_cost, noise_scale=0.0):
        """
        Generate a new solution by solving the Linear Assignment Problem
        on the linearized cost matrix, possibly with noise.
        """
        # Linearization of QAP cost:
        # C[i, j] = sum_k a[i, k] * b[j, p[k]] + sum_k a[k, i] * b[p[k], j]
        # This is expensive to compute exactly for the current p if we want to perturb away from p.
        # Instead, we can use the standard linear surrogate:
        # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        # This assumes p is the identity or is a global approximation.
        # A better approach for perturbation:
        # Calculate the "linear cost" of assigning facility i to location j given the current assignment p of others.
        # However, that's O(n^3) per LAP solve which might be too slow.
        
        # Let's use the global linear surrogate which is O(n^3) to build but solved in O(n^3) or less.
        # C[i, j] = (A * B)[i, j] where A and B are the matrices.
        # Actually, the standard linear relaxation for QAP is:
        # min sum_{i,j} C_{ij} x_{ij}
        # C_{ij} = sum_k a_{ik} b_{kj} + sum_k a_{ki} b_{jk}
        
        # Pre-compute C once? No, C is constant for the instance.
        # Let's compute C.
        # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        # This can be computed as:
        # term1 = a @ b.T  (sum_k a[i, k] * b.T[j, k] = sum_k a[i, k] * b[k, j]) -> Wait.
        # b.T[j, k] is b[k, j].
        # So a @ b.T gives sum_k a[i, k] * b[k, j].
        
        # Let's check indices:
        # Cost = sum_{i, j} a[i, j] b[p[i], p[j]]
        # Linear surrogate usually defined as:
        # C[i, j] = sum_k a[i, k] b[j, k] + sum_k a[k, i] b[k, j]
        
        # Let's compute this matrix C.
        # C1 = a @ b.T  -> C1[i, j] = sum_k a[i, k] b[k, j]
        # C2 = a.T @ b  -> C2[i, j] = sum_k a[k, i] b[k, j] ... wait.
        # a.T @ b: (a.T)[i, k] * b[k, j] = a[k, i] * b[k, j]. Sum over k.
        # So C = C1 + C2?
        # Let's verify.
        # If we assign i to j, the cost contribution in the linear model is:
        # 1. Flow from i to k, distance from j to k? No, distance from p[i] to p[k].
        # The standard linearization (e.g. in QAPLIB papers) is:
        # C_{ij} = \sum_k a_{ik} b_{jk} + \sum_k a_{ki} b_{kj}
        
        # Let's compute C = a @ b + a.T @ b.T ?
        # (a @ b)[i, j] = sum_k a[i, k] b[k, j].
        # (a.T @ b.T)[i, j] = sum_k a.T[i, k] b.T[j, k] = sum_k a[k, i] b[k, j].
        
        # So C = a @ b + a.T @ b.T is NOT quite right because b is not symmetric.
        # Let's stick to the definition:
        # C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
        
        # Term 1: sum_k a[i, k] * b[j, k]
        # This is (a @ b.T)[i, j]
        
        # Term 2: sum_k a[k, i] * b[k, j]
        # This is (a.T @ b)[i, j]
        
        C = (a @ b.T) + (a.T @ b)
        
        # Add noise to diversify
        if noise_scale > 0:
            C_noisy = C + np.random.normal(0, noise_scale, (n, n))
        else:
            C_noisy = C
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(C_noisy)
        p_new = np.zeros(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        return p_new

    # 1. Initial Solution: LAP on linear surrogate
    # Use the same C as above
    C_init = (a @ b.T) + (a.T @ b)
    row_ind, col_ind = linear_sum_assignment(C_init)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop with LAP-based perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    noise_scale = 0.0
    
    while time.time() - start_time < timeout:
        # If we haven't improved in a while, increase noise
        if no_improve_count == 0:
            noise_scale = 0.0
        elif no_improve_count < 3:
            noise_scale = 100.0
        elif no_improve_count < 10:
            noise_scale = 1000.0
        else:
            noise_scale = 10000.0
            # If very stuck, maybe restart from scratch?
            # For now, keep perturbing.
            
        # Perturbation: LAP with noise
        # To avoid O(n^3) overhead being too high, we only do this if we are stuck?
        # Actually, LAP is quite fast for n=100.
        # But we need to be careful with time.
        
        # Alternative: If no_improve_count is small, do small random swaps.
        # If large, do LAP.
        
        if no_improve_count < 2:
             # Small random perturbation
             num_swaps = 3
             for _ in range(num_swaps):
                 i = np.random.randint(n)
                 j = np.random.randint(n)
                 if i == j:
                     continue
                 delta = get_delta(p, i, j)
                 p[i], p[j] = p[j], p[i]
                 current_cost += delta
        else:
             # LAP perturbation
             p_new = lap_perturb(p, current_cost, noise_scale)
             # Check if p_new is significantly different
             p = p_new
             current_cost = calc_cost(p)
        
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset p to best_p to stay in the neighborhood of the best solution
            # But if we used LAP, p is already far.
            # If we didn't improve, reset to best_p for next random perturbation?
            # Or keep the perturbed p?
            # Keeping best_p is safer for "small steps" from best.
            p = best_p.copy()
            current_cost = best_cost
            
        # Check timeout before next iteration
        if time.time() - start_time > timeout - 5.0:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
