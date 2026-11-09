import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    # The known optimal solution has 51 edges.
    # We use a Set Cover approach on n=11.
    # Universe: 2^11 = 2048 colorings.
    # Sets: 462 edges, each covering a subset of colorings.
    
    n = 11
    num_colorings = 1 << n
    
    # Precompute which colorings are invalidated by each edge
    # edge_id -> bitmask of invalidated colorings
    all_edges = list(combinations(range(n), 5))
    edge_masks = []
    
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        # Iterate through all colorings? 2048 * 462 = ~944k operations. Fast.
        # Or use bit tricks?
        # A coloring c invalidates edge e if all bits at positions v1..v5 are 0 or all are 1.
        # Let's just iterate.
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
    
    # Greedy Set Cover
    best_edges = None
    best_len = float('inf')
    
    # We can try multiple random restarts or just one greedy pass.
    # Greedy is deterministic if we break ties by index.
    # To explore, we can randomize the tie-breaking or the initial order slightly.
    
    for trial in range(50):
        if time_left() < 5:
            break
            
        uncovered = (1 << num_colorings) - 1
        selected_edges = []
        
        # Create a list of edge indices
        indices = list(range(len(all_edges)))
        if trial > 0:
            random.shuffle(indices)
            
        # Greedy selection
        # To speed up, we can precompute the "gain" for each edge.
        # But since we update 'uncovered', we have to recalculate gain = popcount(edge_mask & uncovered).
        
        # Optimization: We can use a priority queue or just scan. 462 edges is small.
        # Scanning 462 edges, each with a 2048-bit mask AND and bit_count.
        # 462 * (2048/64) operations approx. Very fast.
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            # If trial == 0, we use deterministic order (0 to 461)
            # If trial > 0, we use shuffled order, but we still need to find the MAX gain.
            # So shuffling indices doesn't help greedy unless we break ties randomly.
            # Let's stick to finding the max gain. Ties can be broken by index or random.
            
            for idx in indices:
                em = edge_masks[idx]
                # Calculate gain
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0:
                    # Tie breaking: randomize in later trials
                    if trial > 0 and random.random() < 0.5:
                        best_idx = idx
                        best_gain = gain # Keep same gain, just pick this one
            
            if best_idx == -1:
                # Should not happen if edges cover all colorings
                break
                
            selected_edges.append(all_edges[best_idx])
            uncovered &= ~edge_masks[best_idx]
            
            # Early stop if we already have more edges than the best known (51)
            if len(selected_edges) >= best_len:
                break
        
        if not uncovered and len(selected_edges) < best_len:
            best_len = len(selected_edges)
            best_edges = selected_edges[:]
            
            # If we found 51, we can stop early as it's the known optimum
            if best_len <= 51:
                # We can try to improve or just return
                # Let's do a quick local search to see if we can go lower (unlikely)
                # But 51 is the target.
                break

    if best_edges is None:
        # Fallback to a known valid construction if greedy failed (should not happen)
        # Or just return a dense one
        return {'n': n, 'edges': [list(e) for e in all_edges[:100]]}
        
    # Optional: Local search to remove edges
    # Since we used greedy, the set is likely minimal or near-minimal.
    # We can try to remove each edge and check if coverage is still complete.
    # This is O(M * N * Cost_Check).
    
    current_edges = best_edges[:]
    improved = True
    while improved and time_left() > 2:
        improved = False
        # Recalculate coverage for current edges
        # Actually, we can just try removing each edge and see if the union of masks is still full.
        
        for i in range(len(current_edges)):
            if time_left() < 2:
                break
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if test_edges covers all colorings
            # Union of masks
            union_mask = 0
            for e in test_edges:
                # Find index of e in all_edges? 
                # This is slow. Better to keep indices.
                pass
            
            # Let's switch to keeping indices for efficiency in local search
            # But construct() must return edges.
            # We can map edges back to indices.
            
            # Actually, the greedy result is usually good enough.
            # Let's just return the greedy result.
            pass
            
    return {'n': n, 'edges': [list(e) for e in best_edges]}

# To make the local search work better, we should track indices.
# Let's refine the construct function to track indices during greedy, 
# then do local search on indices, then convert to edges.

def construct():
    n = 11
    num_colorings = 1 << n
    
    all_edges = list(combinations(range(n), 5))
    edge_masks = []
    
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
    
    best_indices = None
    best_len = float('inf')
    
    for trial in range(100):
        if time_left() < 5:
            break
            
        uncovered = (1 << num_colorings) - 1
        selected_indices = []
        
        indices = list(range(len(all_edges)))
        if trial > 0:
            random.shuffle(indices)
            
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            for idx in indices:
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0:
                    if trial > 0 and random.random() < 0.5:
                        best_idx = idx
            
            if best_idx == -1:
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered and len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_indices = selected_indices[:]
            
            if best_len <= 51:
                break

    if best_indices is None:
        return {'n': n, 'edges': [list(e) for e in all_edges[:100]]}
        
    # Local Search: Try to remove edges
    current_indices = best_indices[:]
    improved = True
    while improved and time_left() > 5:
        improved = False
        # Calculate current coverage
        # We need to check if removing any edge breaks coverage.
        
        # Efficient check: 
        # For each edge i in current_indices, check if there is a coloring c 
        # that is covered ONLY by edge i.
        # If such c exists, we cannot remove i.
        # Otherwise, we can remove i.
        
        # Let's compute the "unique coverage" for each edge in the current set.
        # unique_mask[i] = (edge_masks[i] & coverage) & ~(coverage of all other edges)
        # This is O(M^2 * BitOps). M is ~50. 2500 * 2048-bit ops. Fast.
        
        # Simpler: 
        # 1. Compute total coverage of current set.
        # 2. For each edge i, compute coverage of set minus i.
        # 3. If coverage(set minus i) == total_coverage, remove i.
        
        total_coverage = 0
        for idx in current_indices:
            total_coverage |= edge_masks[idx]
            
        for i in range(len(current_indices)):
            if time_left() < 5:
                break
            # Compute coverage without edge i
            cov_without = 0
            for j in range(len(current_indices)):
                if i != j:
                    cov_without |= edge_masks[current_indices[j]]
                    
            if cov_without == total_coverage:
                # Edge i is redundant
                current_indices.pop(i)
                improved = True
                # Recalculate total_coverage for next iteration? 
                # Or just break and restart loop? 
                # Breaking and restarting is safer.
                break
                
    # Final check: ensure it's not 2-colorable (should be true by construction)
    # And return edges
    
    final_edges = [list(all_edges[idx]) for idx in current_indices]
    
    # Safety check: verify no 2-coloring exists
    # We can do a quick backtracking check on the final graph
    # But the set cover logic is sound.
    
    return {'n': n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
