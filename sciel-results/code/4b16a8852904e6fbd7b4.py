import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Helper to check if a subset (mask) induces a regular subgraph
    def is_regular_subset(mask, adj_masks, n):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        if not verts:
            return False
            
        # Calculate degrees
        degs = []
        for v in verts:
            # Degree in induced subgraph is popcount of (adj[v] & mask)
            degs.append((adj_masks[v] & mask).bit_count())
        
        # Check if all degrees are equal
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    # Helper to count regular subsets of size k
    def count_regular_subsets(n, adj_masks, k):
        count = 0
        # Generate all masks of size k
        # Using combinations for simplicity, might be slow for large n, but n<=21 is ok
        # For n=21, k=6: 54264 combos. k=7: 116280. 
        # Total for k>=6 is ~1M. 
        # This function will be called many times in local search.
        # We need it to be fast.
        
        # Optimization: Precompute popcounts? No, mask changes.
        # Use Gosper's hack to iterate masks of fixed weight?
        # Or just use combinations and build mask.
        
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if is_regular_subset(mask, adj_masks, n):
                count += 1
        return count

    def full_validation(n, adj_masks):
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            if count_regular_subsets(n, adj_masks, k) > 0:
                return False
        return True

    def edges_to_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def adj_masks_to_edges(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Search for a valid graph
    best_n = 0
    best_edges = []
    
    # Try n=21 first
    for target_n in [21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph
        # We want to avoid regular subgraphs. Random graphs are often irregular.
        # Start with a random graph with p=0.5
        adj_masks = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Local search to minimize number of regular subsets of size 6
        # We use size 6 as a proxy because it's the smallest forbidden size
        # and likely the most numerous constraint.
        
        current_cost = count_regular_subsets(target_n, adj_masks, 6)
        
        # If we start with 0, check full validation immediately
        if current_cost == 0:
            if full_validation(target_n, adj_masks):
                best_n = target_n
                best_edges = adj_masks_to_edges(target_n, adj_masks)
                break
        
        # Hill Climbing / Simulated Annealing
        # We flip edges to reduce cost
        no_improvement = 0
        while time.time() < deadline - 10:
            # Pick a random edge to flip
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
                
            # Flip edge
            if adj_masks[u] & (1 << v):
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
            else:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
            new_cost = count_regular_subsets(target_n, adj_masks, 6)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improvement = 0
                if current_cost == 0:
                    if full_validation(target_n, adj_masks):
                        best_n = target_n
                        best_edges = adj_masks_to_edges(target_n, adj_masks)
                        break
            else:
                # Revert if no improvement (Hill Climbing)
                # Or accept with some probability (Simulated Annealing)
                # For simplicity and speed, let's stick to Hill Climbing with random restarts
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                no_improvement += 1
                
            if no_improvement > 100:
                # Random restart
                adj_masks = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                current_cost = count_regular_subsets(target_n, adj_masks, 6)
                no_improvement = 0
                if current_cost == 0:
                    if full_validation(target_n, adj_masks):
                        best_n = target_n
                        best_edges = adj_masks_to_edges(target_n, adj_masks)
                        break
                continue

        if best_n == target_n:
            break

    if best_n == 0:
        # Fallback: Return a small valid graph (n=5) to ensure validity
        # The seed program returned n=5.
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

