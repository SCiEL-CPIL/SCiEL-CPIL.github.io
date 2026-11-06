import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin for safety
    
    def time_left():
        return deadline - time.time()
        
    # Bitmask representation
    # masks[i] is an integer where bit j is 1 if edge (i, j) exists
    
    def count_k4(masks, n):
        """Count K4s using bit operations."""
        count = 0
        for i in range(n):
            mi = masks[i]
            # Iterate over neighbors j > i
            # We only need to count each K4 once, so we restrict to i < j < k < l
            # But simple counting is fine for SA cost function
            for j in range(i + 1, n):
                if (mi >> j) & 1:
                    # Common neighbors of i and j
                    common = mi & masks[j]
                    # Remove i and j from common
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Check if any edge exists within common
                    # An edge (k, l) with k, l in common exists if 
                    # (masks[k] & common) has any bit set for l > k
                    # To speed up, we can just check if the subgraph induced by common has any edge
                    # This is O(deg^2) in worst case, but degrees are small
                    # Optimization: iterate bits of common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if k is connected to any other node in common
                        # neighbors of k in common
                        neighbors_in_common = masks[k] & common
                        # We must exclude k itself (not in common usually) and ensure we don't count twice
                        # Actually, if neighbors_in_common has any bit set, there is an edge.
                        # But we must ensure the other endpoint is also in common.
                        # masks[k] & common gives all neighbors of k that are in common.
                        # If this is non-zero, there is at least one edge (k, x) where x in common.
                        # This forms a K4 (i, j, k, x).
                        if neighbors_in_common:
                            count += 1
        return count

    def has_independent_6(masks, n):
        """Check if there is an independent set of size 6 using bitset branch and bound."""
        # Independent set in G is a clique in G_bar.
        # Let's work with the complement masks.
        # comp_mask[i] has bits set for all j != i such that NO edge (i, j) exists.
        
        # To speed up, we can precompute comp masks if we call this often, 
        # but for SA we call it on every step, so we need it fast.
        # Actually, checking for existence of K6 in complement is hard.
        # Let's use a recursive search on the original graph for independent sets.
        
        # State: available vertices (bitmask), current set size
        # We want to find a set of 6 vertices with no edges between them.
        
        # Heuristic: Order vertices by degree? 
        # For bitset, we can just pick the lowest bit to branch on.
        
        def search(available, size):
            if size == 6:
                return True
            if available == 0:
                return False
            # Pruning: if popcount(available) < 6 - size, return False
            if available.bit_count() < 6 - size:
                return False
                
            # Pick a vertex to branch on
            # Heuristic: pick a vertex with high degree in the induced subgraph? 
            # Or just the lowest bit for simplicity/speed.
            v = (available & -available).bit_length() - 1
            available ^= (1 << v)
            
            # Option 1: Include v
            # If we include v, we must remove all its neighbors from available
            # Non-neighbors of v in available:
            # neighbors of v in original graph: masks[v]
            # We want vertices in available that are NOT neighbors of v
            # non_neighbors = available & ~masks[v]
            if search(available & ~masks[v], size + 1):
                return True
                
            # Option 2: Exclude v
            if search(available, size):
                return True
                
            return False
            
        all_vertices = (1 << n) - 1
        return search(all_vertices, 0)

    def get_violations(masks, n):
        k4_count = count_k4(masks, n)
        ind6_exists = has_independent_6(masks, n)
        # For SA, we want a scalar cost. 
        # If ind6 exists, it's a binary constraint. 
        # We can treat it as a large penalty if we just want to minimize violations.
        # However, "count" of independent sets is hard to compute fast.
        # Let's use a hybrid: 
        # Cost = k4_count * 1000 + (1 if ind6_exists else 0) * 5000
        # This prioritizes removing K4s first, then independent sets.
        # Actually, we need BOTH to be zero.
        # If we just minimize k4_count, we might get stuck with ind6.
        # Let's try to minimize k4_count first. If k4_count == 0, then minimize ind6.
        # But SA needs a single landscape.
        
        # Alternative: Use a random sample of independent sets? No, must be exact.
        # Let's stick to: Cost = k4_count * 10 + (1 if ind6_exists else 0) * 100
        # This encourages K4-free first.
        
        cost = k4_count * 10
        if ind6_exists:
            cost += 100
        return cost, k4_count, ind6_exists

    # Try to find a valid graph for n=30, 31, etc.
    # We will iterate n from 30 down to 29? No, we want to MAXIMIZE n.
    # So try n=35, 34... down to 29?
    # Starting with large n is hard to satisfy. 
    # Starting with n=30 is a good bet if 29 is the baseline.
    
    best_result = None
    
    # We know Paley(29) is valid. Let's store it as a fallback.
    q = 29
    n_base = q
    squares = set(x * x % q for x in range(1, q))
    base_masks = [0] * n_base
    base_edges = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (j - i) % q in squares:
                base_masks[i] |= (1 << j)
                base_masks[j] |= (1 << i)
                base_edges.append([i, j])
                
    # Verify base
    _, k4c, ind6 = get_violations(base_masks, n_base)
    if k4c == 0 and not ind6:
        best_result = {'n': n_base, 'edges': base_edges}
    else:
        # Should not happen, but if it does, we are in trouble
        pass

    # Try to find valid graphs for n > 29
    # We'll try n=30 first, then 31, etc., if time permits.
    # But finding a valid graph for n=30 might take longer than n=29.
    # So we should spend most time on n=30.
    
    for target_n in [30, 31, 32]:
        if time_left() < 10.0:
            break
            
        # Initialize graph
        # Strategy: Start with a random graph with edge probability ~0.5
        # Or start with a random regular graph?
        # Let's start with a random graph and anneal.
        
        masks = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
                    
        # Simulated Annealing
        T = 100.0
        T_min = 0.01
        cooling = 0.99
        max_steps = 50000
        
        current_cost, _, _ = get_violations(masks, target_n)
        best_cost = current_cost
        best_masks = masks[:]
        
        found_valid = False
        
        for step in range(max_steps):
            if time_left() < 5.0:
                break
                
            # Pick a random edge to flip
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j:
                continue
                
            # Flip edge (i, j)
            masks[i] ^= (1 << j)
            masks[j] ^= (1 << i)
            
            new_cost, new_k4, new_ind6 = get_violations(masks, target_n)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_masks = masks[:]
                
                if new_k4 == 0 and not new_ind6:
                    found_valid = True
                    # Construct edges from best_masks
                    edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if (best_masks[u] >> v) & 1:
                                edges.append([u, v])
                    best_result = {'n': target_n, 'edges': edges}
                    # We can stop early if we found a valid graph for this n
                    # But maybe we can find a valid graph for n+1?
                    # Let's break out of SA loop and try next n if time allows
                    break
            
            # Acceptance probability
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_cost = new_cost
            else:
                # Revert
                masks[i] ^= (1 << j)
                masks[j] ^= (1 << i)
                
            T *= cooling
            if T < T_min:
                T = T_min
                
        if found_valid:
            # We found a valid graph for target_n. 
            # Since we want to maximize n, we should try higher n if possible.
            # But if we found one for 30, that's better than 29.
            # We will continue to try 31, 32 in the outer loop if time permits.
            pass
            
    # If we didn't find anything better than 29, return 29
    if best_result is None:
        # Fallback to Paley 29
        best_result = {'n': 29, 'edges': base_edges}
        
    return best_result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found graph with n={res['n']}")
