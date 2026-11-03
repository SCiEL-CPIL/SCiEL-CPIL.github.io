# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Ensure matrices are int64 for exact arithmetic if needed, though float64 is usually fine for deltas
    # The problem statement says a and b are int64.
    
    # 1. Initial Heuristic Solution
    # Rank matching: high flow sums to high distance sums (or vice versa? 
    # QAP minimization: high flow * low distance is good.
    # So we want to match high flow row sums with low distance row sums?
    # Actually, the standard heuristic is often:
    # Sort facilities by row sum of A (descending)
    # Sort locations by row sum of B (ascending) -> Match heavy flow to central (low distance) locations.
    
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Heuristic: High flow facilities should go to locations with low "centrality" (low sum of distances)
    # Wait, b_row_sums is sum of distances from that location to all others.
    # If facility i has high flow to everyone, it should be at a location j that has low sum of distances to everyone.
    # So: Sort a_row_sums descending. Sort b_row_sums ascending.
    
    heavy_facilities = np.argsort(-a_row_sums, kind="stable")
    central_locations = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    def compute_cost(p):
        """Compute exact QAP cost."""
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Using matrix multiplication for speed
        # b_perm = b[np.ix_(p, p)]
        # return np.sum(a * b_perm)
        # More efficient:
        # sum_i sum_j a[i,j] b[p[i], p[j]]
        # Let B_p be the permuted matrix.
        # This is trace(A^T B_p) ? No.
        # It is element-wise product sum.
        return np.sum(a * b[np.ix_(p, p)])

    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Simulated Annealing
    # Precompute data for fast delta calculation
    # Delta for swapping facilities i and j (locations li, lj):
    # D(i,j) = sum_{k != i,j} (a[i,k] + a[j,k] + a[k,i] + a[k,j]) * (b[li,k] - b[lj,k]) ... wait, this is getting complicated.
    # Standard 2-swap delta:
    # Let L_i = p[i], L_j = p[j].
    # Swap means i goes to L_j, j goes to L_i.
    # Change in cost = 
    #   - (terms involving i at L_i and j at L_j)
    #   + (terms involving i at L_j and j at L_i)
    #   + (terms involving k at L_k with i and j changed)
    
    # A faster way to compute delta for swap (i, j):
    # Let d_k = b[L_i, k] - b[L_j, k] for k != i, j.
    # Let f_k = a[i, k] + a[k, i] + a[j, k] + a[k, j] for k != i, j.
    # Delta = sum_{k != i, j} f_k * d_k 
    #         + (a[i, j] + a[j, i]) * (b[L_i, L_j] + b[L_j, L_i] - 2*b[L_i, L_j]?) No.
    # Let's derive carefully.
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    # Terms affected by swapping p[i] and p[j]:
    # 1. (i,i), (j,j), (i,j), (j,i)
    # 2. (i,k), (k,i) for k != i,j
    # 3. (j,k), (k,j) for k != i,j
    
    # Old contribution:
    # a[i,i]b[Li,Li] + a[j,j]b[Lj,Lj] + a[i,j]b[Li,Lj] + a[j,i]b[Lj,Li]
    # + sum_{k!=i,j} (a[i,k]b[Li,pk] + a[k,i]b[pk,Li] + a[j,k]b[Lj,pk] + a[k,j]b[pk,Lj])
    
    # New contribution:
    # a[i,i]b[Lj,Lj] + a[j,j]b[Li,Li] + a[i,j]b[Lj,Li] + a[j,i]b[Li,Lj]
    # + sum_{k!=i,j} (a[i,k]b[Lj,pk] + a[k,i]b[pk,Lj] + a[j,k]b[Li,pk] + a[k,j]b[pk,Li])
    
    # Difference (New - Old):
    # Diagonal: a[i,i](b[Lj,Lj]-b[Li,Li]) + a[j,j](b[Li,Li]-b[Lj,Lj]) = (a[i,i]-a[j,j])(b[Lj,Lj]-b[Li,Li])
    # Off-diagonal i,j: a[i,j](b[Lj,Li]-b[Li,Lj]) + a[j,i](b[Li,Lj]-b[Lj,Li]) = (a[j,i]-a[i,j])(b[Li,Lj]-b[Lj,Li])
    
    # Cross terms with k:
    # sum_{k!=i,j} [ a[i,k](b[Lj,pk]-b[Li,pk]) + a[k,i](b[pk,Lj]-b[pk,Li]) + a[j,k](b[Li,pk]-b[Lj,pk]) + a[k,j](b[pk,Li]-b[pk,Lj]) ]
    # = sum_{k!=i,j} [ (a[i,k]-a[j,k])(b[Lj,pk]-b[Li,pk]) + (a[k,i]-a[k,j])(b[pk,Lj]-b[pk,Li]) ]
    
    # Note: b is symmetric? QAPLIB instances usually have symmetric B (distance matrix).
    # If B is symmetric, b[u,v] = b[v,u].
    # Then b[Lj,pk]-b[Li,pk] = -(b[Li,pk]-b[Lj,pk]).
    # And b[pk,Lj]-b[pk,Li] = b[Lj,pk]-b[Li,pk].
    # So the k-sum becomes:
    # sum_{k!=i,j} (a[i,k]-a[j,k] - a[k,i] + a[k,j]) * (b[Lj,pk]-b[Li,pk])
    # Let A_diff[k] = a[i,k] - a[j,k] - a[k,i] + a[k,j]
    # Let B_diff[k] = b[Lj,pk] - b[Li,pk]
    # Sum = dot(A_diff, B_diff) for k != i,j.
    
    # If A is symmetric (flow matrix), a[i,k]=a[k,i].
    # Then A_diff[k] = 2*(a[i,k] - a[j,k]).
    # B_diff[k] = b[Lj,pk] - b[Li,pk].
    
    # Let's assume symmetry for speed, but verify or handle generally.
    # QAPLIB tai256c: A and B are symmetric.
    
    def two_opt_delta(p, i, j):
        li = p[i]
        lj = p[j]
        
        # Vectorized computation of the sum over k
        # We need to exclude i and j from the sum.
        # Create a mask or just compute full sum and subtract i,j terms.
        # Full sum over all k:
        # A_diff_full[k] = a[i,k] - a[j,k] - a[k,i] + a[k,j]
        # B_diff_full[k] = b[lj, p[k]] - b[li, p[k]]
        
        # Since A and B are symmetric:
        # A_diff_full[k] = 2 * (a[i,k] - a[j,k])
        # B_diff_full[k] = b[lj, p[k]] - b[li, p[k]]
        
        # We can compute this efficiently using numpy.
        # But doing this inside a loop for every pair (i,j) is O(n) per pair, O(n^3) per pass.
        # For n=256, n^3 = 16M operations. In Python/NumPy, this might be slow if done naively in a loop.
        # However, we can vectorize the inner product.
        
        # Let's try to optimize the delta calculation.
        # Instead of computing delta for every (i,j) pair, we can iterate i and find best j?
        # Or just do random swaps in SA.
        
        # For SA, we pick random i, j.
        # Delta calculation O(n) is acceptable if we do ~100k-1M swaps.
        # 1M swaps * 256 ops = 256M ops. NumPy can do this in seconds.
        
        # Precompute row vectors for a and b?
        # a[i, :] and a[j, :] are rows.
        # b[lj, p] and b[li, p] are permuted rows.
        
        # A_diff = 2 * (a[i, :] - a[j, :])
        # B_diff = b[lj, p] - b[li, p]
        
        # Sum = dot(A_diff, B_diff)
        
        # Subtract the i and j terms from the sum because we included them in the full vectors.
        # Term for k=i:
        # A_diff[i] = 2*(a[i,i] - a[j,i])
        # B_diff[i] = b[lj, p[i]] - b[li, p[i]] = b[lj, li] - b[li, li]
        # Term for k=j:
        # A_diff[j] = 2*(a[i,j] - a[j,j])
        # B_diff[j] = b[lj, p[j]] - b[li, p[j]] = b[lj, lj] - b[li, lj]
        
        # The formula derived earlier for k != i,j is valid.
        # So we compute full dot product and subtract the i and j components.
        
        a_diff = 2.0 * (a[i, :] - a[j, :])
        b_diff = b[lj, p] - b[li, p]
        
        total_dot = np.dot(a_diff, b_diff)
        
        # Subtract k=i term
        # Note: p[i] is li, p[j] is lj.
        # B_diff[i] = b[lj, li] - b[li, li]
        # A_diff[i] = 2 * (a[i, i] - a[j, i])
        term_i = a_diff[i] * b_diff[i]
        
        # Subtract k=j term
        # B_diff[j] = b[lj, lj] - b[li, lj]
        # A_diff[j] = 2 * (a[i, j] - a[j, j])
        term_j = a_diff[j] * b_diff[j]
        
        sum_k = total_dot - term_i - term_j
        
        # Diagonal and Off-diagonal i,j terms:
        # (a[i,i]-a[j,j]) * (b[lj,lj] - b[li,li])
        # + (a[j,i]-a[i,j]) * (b[li,lj] - b[lj,li])
        
        # Since A and B symmetric:
        # a[j,i] = a[i,j] -> second term is 0.
        # b[li,lj] = b[lj,li] -> second term is 0.
        
        # So only diagonal term remains?
        # Wait, let's re-verify the "Off-diagonal i,j" part.
        # Old: a[i,j]b[li,lj] + a[j,i]b[lj,li]
        # New: a[i,j]b[lj,li] + a[j,i]b[li,lj]
        # Diff: a[i,j](b[lj,li]-b[li,lj]) + a[j,i](b[li,lj]-b[lj,li])
        # If symmetric: b[lj,li]=b[li,lj] and a[i,j]=a[j,i].
        # Diff: a[i,j](D - D) + a[i,j](D - D) = 0.
        # Yes, off-diagonal i,j swap cost is 0 if A and B are symmetric.
        
        # So Delta = sum_k + (a[i,i]-a[j,j]) * (b[lj,lj] - b[li,li])
        
        diag_diff = (a[i, i] - a[j, j]) * (b[lj, lj] - b[li, li])
        
        return sum_k + diag_diff

    # SA Parameters
    T = 1000.0  # Initial temperature
    T_min = 1.0
    alpha = 0.9999  # Cooling factor
    
    # Estimate number of steps we can afford
    # Each step involves delta calc (O(n)) and potentially a cost update.
    # Let's run until time limit.
    
    while time.time() < deadline - 5.0:
        # Pick random i, j
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
            
        delta = two_opt_delta(p, i, j)
        
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Optional: If we are stuck at T_min, we could restart with higher T?
        # Or just keep going.
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
