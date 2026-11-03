import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost_full(p):
        return np.sum(a * b[p][:, p])
    
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Hungarian-based reassignment: compute dynamic cost matrix
    # C[i, j] = marginal cost of assigning facility i to location j
    # given current assignment of all other facilities
    
    def compute_dynamic_cost(p):
        """
        Compute the dynamic cost matrix where C[i,j] represents the
        marginal cost of placing facility i at location j, given that
        all other facilities k (k != i) remain at their current locations p[k].
        
        The total cost is sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        For facility i at location j, the contribution is:
        sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        
        But we need to be careful: when we assign i to j, we're changing
        p[i] from its old value to j. The terms involving i are:
        - a[i,k] * b[j, p[k]] for all k != i
        - a[k,i] * b[p[k], j] for all k != i
        - a[i,i] * b[j, j] (if k=i, but we're reassigning i so this is separate)
        
        Actually, let's think more carefully. The cost is:
        sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        If we change p[i] from u to v, the change in cost is:
        - Remove terms where i is the first index: sum_j a[i,j] * b[u, p[j]]
        - Remove terms where i is the second index: sum_j a[j,i] * b[p[j], u]
        - Add terms where i is the first index: sum_j a[i,j] * b[v, p[j]]
        - Add terms where i is the second index: sum_j a[j,i] * b[p[j], v]
        
        For the dynamic cost matrix approach, we want C[i,j] such that
        the Hungarian algorithm finds a good reassignment.
        
        Let's define for each facility i and location j:
        C[i,j] = sum_{k != i} a[i,k] * b[j, p[k]] + sum_{k != i} a[k,i] * b[p[k], j]
        
        This is the cost contribution of facility i at location j, excluding
        the self-interaction term a[i,i]*b[j,j] which is constant for the
        Hungarian problem (it doesn't depend on which facility gets which location
        in a way that matters for the assignment).
        
        Actually, a[i,i]*b[j,j] does depend on j, so we should include it:
        C[i,j] = sum_{k != i} a[i,k] * b[j, p[k]] + sum_{k != i} a[k,i] * b[p[k], j] + a[i,i]*b[j,j]
        
        But wait, when we do the Hungarian reassignment, we're reassigning ALL
        facilities simultaneously. So the dynamic cost matrix should capture
        the marginal contribution of each facility-location pair given the
        current assignment of others.
        
        Let's use:
        C[i,j] = sum_{k} a[i,k] * b[j, p[k]] + sum_{k} a[k,i] * b[p[k], j]
        
        This includes the k=i term which gives a[i,i]*b[j,j].
        """
        # For each facility i, compute the vector of costs for all locations j
        # C[i,j] = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
        
        # First term: sum_k a[i,k] * b[j, p[k]]
        # = sum_k a[i,k] * b[j, p[k]]
        # For fixed i, this is: a[i,:] dot b[j, p]
        # b[j, p] is the row j of b, indexed by p
        
        # Second term: sum_k a[k,i] * b[p[k], j]
        # = sum_k a[:,i] * b[p, j]
        # b[p, j] is the column j of b, indexed by p
        
        # Let's compute this efficiently using matrix operations
        # b_p = b[:, p] gives us b[*, p[0]], b[*, p[1]], ...
        # Actually, we need b[j, p[k]] for all j,k and b[p[k], j] for all k,j
        
        # b_jp = b[:, p] has shape (n, n) where b_jp[j, k] = b[j, p[k]]
        # b_pj = b[p, :] has shape (n, n) where b_pj[k, j] = b[p[k], j]
        
        b_jp = b[:, p]  # b_jp[j, k] = b[j, p[k]]
        b_pj = b[p, :]  # b_pj[k, j] = b[p[k], j]
        
        # First term for facility i: sum_k a[i,k] * b[j, p[k]] = a[i,:] @ b_jp[j,:]
        # So C_first[i, j] = (a @ b_jp.T)[i, j]
        # Wait: sum_k a[i,k] * b_jp[j,k] = row i of a dotted with row j of b_jp
        # = (a @ b_jp.T)[i, j]
        
        C_first = a @ b_jp.T  # C_first[i, j] = sum_k a[i,k] * b[j, p[k]]
        
        # Second term for facility i: sum_k a[k,i] * b[p[k], j] = a[:,i] @ b_pj[:,j]
        # = (a.T @ b_pj)[i, j]
        C_second = a.T @ b_pj  # C_second[i, j] = sum_k a[k,i] * b[p[k], j]
        
        return C_first + C_second
    
    # Main loop: alternate between Hungarian reassignment and local swaps
    iteration = 0
    T = 1000.0  # Initial temperature
    min_T = 1.0
    
    while time.time() < deadline:
        # Compute dynamic cost matrix
        C = compute_dynamic_cost(p)
        
        # Run Hungarian algorithm to find optimal reassignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        # Acceptance criterion:
        # 1. If new_cost < current_cost, accept.
        # 2. If new_cost >= current_cost, accept with probability exp(-delta / T).
        delta = new_cost - current_cost
        if delta < 0 or (T > min_T and np.random.rand() < np.exp(-delta / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            
            p = new_p.copy()
            current_cost = new_cost
            # Cool down
            T = max(min_T, T * 0.95)
        else:
            # If Hungarian step was rejected, try a random 2-swap to perturb
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
                swap_cost = calc_cost_full(p)
                swap_delta = swap_cost - current_cost
                if swap_delta < 0 or (T > min_T and np.random.rand() < np.exp(-swap_delta / T)):
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_p = p.copy()
                    current_cost = swap_cost
                    T = max(min_T, T * 0.95)
                else:
                    p[i], p[j] = p[j], p[i]
                    # Keep T constant or cool slightly if no move accepted
                    T = max(min_T, T * 0.99)
        
        iteration += 1
    
    return best_p.tolist()
