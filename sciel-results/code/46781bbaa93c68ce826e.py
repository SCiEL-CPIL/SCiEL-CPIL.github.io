"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    weights = list(range(1, N + 1))  # weight[i] = i+1 for vertex i
    
    # Build adjacency: edge between i and j if gcd(i+1, j+1) > 1
    # We want maximum weight independent set in the non-coprime graph
    # i.e., select vertices such that no two have gcd > 1
    
    n = N
    # Adjacency as bitsets
    adj = [0] * n
    for i in range(n):
        vi = i + 1
        for j in range(i + 1, n):
            vj = j + 1
            if gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Degree for ordering
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree (higher degree first) to improve pruning
    order = sorted(range(n), key=lambda x: -degrees[x])
    pos = {v: i for i, v in enumerate(order)}
    
    # Reorder adj in new order
    new_adj = [0] * n
    for i in range(n):
        old_i = order[i]
        mask = adj[old_i]
        for j in range(n):
            if mask & (1 << order[j]):
                new_adj[i] |= (1 << j)
    
    weights_order = [weights[order[i]] for i in range(n)]
    
    # Precompute for each vertex: neighbors bitmask in new order
    # Upper bound: sum of weights of all vertices in candidate set
    
    best_weight = [0] * n  # best_weight[i] = sum of weights of vertices i..n-1
    total = sum(weights_order)
    best_weight[n-1] = weights_order[n-1]
    for i in range(n-2, -1, -1):
        best_weight[i] = best_weight[i+1] + weights_order[i]
    
    best_sum = [0]
    best_mask = [0]
    
    def solve():
        """Branch and bound for maximum weight independent set."""
        stack = [(0, n, 0)]  # (start, end, current_sum)
        # Use iterative approach with explicit bounds
        
        # Actually, let's use recursive with memoization-like approach
        # But for 256 vertices, we need a good branching strategy
        
        def bbs(start, end, current_sum):
            if time.time() > deadline:
                return
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
            
            if start >= end:
                return
            
            # Upper bound: current_sum + sum of all remaining weights
            if current_sum + best_weight[start] <= best_sum[0]:
                return
            
            # Select vertex with max degree in [start, end)
            # Try both including and excluding
            best_v = start
            best_deg = -1
            for v in range(start, end):
                # Count neighbors in [start, end)
                mask = new_adj[v]
                # Count bits in mask & ((1 << (end - start)) - 1) << start
                sub_mask = mask & (((1 << (end - start)) - 1) << start)
                deg = bin(sub_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            wv = weights_order[v]
            
            # Option 1: include v
            # Remove v and all its neighbors from consideration
            new_end_candidates = list(range(start, end))
            # Actually, let's be smarter: process vertices one by one
            
            # Option 2: exclude v
            bbs(v + 1, end, current_sum)
            
            # Option 1: include v
            # The next vertices to consider are those in [start, v) and [v+1, end) 
            # that are not neighbors of v
            # For simplicity in this implementation, let's use a different approach:
            # Process vertices in order, at each step decide include/exclude
            
            # Let me restructure: use a cleaner recursive approach
            pass
        
        # Cleaner approach: process vertices in a fixed order
        def bbs2(idx, current_sum, available_mask):
            if time.time() > deadline:
                return
            if idx == n:
                if current_sum > best_sum[0]:
                    best_sum[0] = current_sum
                    best_mask[0] = available_mask  # not exactly, but we track selection
                return
            
            # Upper bound
            if current_sum + best_weight[idx] <= best_sum[0]:
                return
            
            # Option: exclude vertex idx
            bbs2(idx + 1, current_sum, available_mask)
            
            # Option: include vertex idx (if available)
            if available_mask & (1 << idx):
                # Remove idx and its neighbors
                new_mask = available_mask & ~((1 << idx) | new_adj[idx])
                bbs2(idx + 1, current_sum + weights_order[idx], new_mask)
        
        bbs2(0, 0, (1 << n) - 1)
    
    solve()
    
    # Reconstruct the solution
    # We need to track which vertices were selected
    # Let me redo with proper tracking
    
    best_sum2 = [0]
    selected = [0]
    
    def solve2():
        def bbs(idx, current_sum, selected_mask):
            if time.time() > deadline:
                return
            if idx == n:
                if current_sum > best_sum2[0]:
                    best_sum2[0] = current_sum
                    selected[0] = selected_mask
                return
            
            if current_sum + best_weight[idx] <= best_sum2[0]:
                return
            
            # Exclude idx
            bbs(idx + 1, current_sum, selected_mask)
            
            # Include idx
            if selected_mask & (1 << idx) == 0:
                # Check if idx is compatible with all selected
                conflict = False
                sm = selected_mask
                while sm:
                    lsb = sm & (-sm)
                    bit_pos = lsb.bit_length() - 1
                    if new_adj[idx] & (1 << bit_pos):
                        conflict = True
                        break
                    sm ^= lsb
                if not conflict:
                    new_mask = selected_mask | (1 << idx)
                    bbs(idx + 1, current_sum + weights_order[idx], new_mask)
        
        bbs(0, 0, 0)
    
    solve2()
    
    # Convert back to original indices
    result = []
    for i in range(n):
        if selected[0] & (1 << i):
            result.append(order[i] + 1)
    
    result.sort()
    
    # Validate
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to primes
                from math import isqrt
                return [1] + [a for a in range(2, N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    return result

# EVOLVE-BLOCK-END
