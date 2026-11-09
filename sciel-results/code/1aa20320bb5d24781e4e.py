import time
import random
from itertools import combinations

_WALL_LIMIT = 150.0
_MARGIN = 10.0

def construct():
    start = time.time()
    deadline = start + _WALL_LIMIT - _MARGIN
    
    # We focus on n=11 as it is the standard vertex count for the 51-edge construction
    n = 11
    num_colorings = 1 << n
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute masks for each edge
    # mask[i] is a bitset of length 2^n. Bit c is 1 if coloring c is monochromatic for edge i.
    edge_masks = [0] * num_edges
    
    for i, e in enumerate(all_edges):
        v1, v2, v3, v4, v5 = e
        m = 0
        # Optimization: Iterate through all colorings
        # For n=11, 2048 colorings. 462 edges. Total ~1M iterations.
        for c in range(num_colorings):
            # Check if all 5 vertices have the same color in coloring c
            # Color of vertex v is (c >> v) & 1
            # We can check if the bits at these positions are all 0 or all 1.
            # Let's extract the bits.
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            
            # If all equal, it's monochromatic
            if b1 == b2 == b3 == b4 == b5:
                m |= (1 << c)
        edge_masks[i] = m

    full_mask = (1 << num_colorings) - 1
    
    best_indices = None
    best_len = float('inf')
    
    # Run multiple greedy trials to increase chance of finding the optimal 51-edge cover
    # Deterministic first, then randomized
    for trial in range(20):
        if time.time() > deadline:
            break
            
        uncovered = full_mask
        selected_indices = []
        remaining_indices = set(range(num_edges))
        
        # Create a list for iteration
        # In trial 0, use natural order. In others, shuffle for tie-breaking diversity?
        # Actually, greedy picks max gain. Ties are broken by index in current implementation.
        # To explore, we can shuffle the order of checking ties, or just rely on the fact that 
        # for n=11, the greedy solution is often 51.
        
        indices_list = list(range(num_edges))
        if trial > 0:
            random.shuffle(indices_list)
            
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            # Iterate through all remaining edges to find the one covering the most uncovered colorings
            # Note: We don't remove from remaining_indices until we pick one, 
            # but we should only consider edges not already selected.
            # For efficiency, we can just iterate all and check if not in selected? 
            # No, we should maintain a list of available edges.
            
            # Optimization: Instead of removing from a set (which is O(n) for removal if we use list, 
            # but we are using a list of indices and just skipping selected ones? 
            # The current code uses `remaining.remove(best_idx)` which is O(N).
            # Let's use a boolean array or just iterate all and check membership in a set of selected.
            
            selected_set = set(selected_indices)
            
            for idx in indices_list:
                if idx in selected_set:
                    continue
                    
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking for later trials
                    if trial > 0 and random.random() < 0.5:
                        best_idx = idx
            
            if best_idx == -1:
                # No edge covers any new coloring? Should not happen if full_mask is not covered
                # but if we get stuck, break.
                break
                
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            # Early termination if we already have more edges than the best found so far
            if len(selected_indices) >= best_len:
                break
            
        if uncovered == 0:
            # We found a valid cover
            if len(selected_indices) < best_len:
                best_len = len(selected_indices)
                best_indices = selected_indices[:]
                
                # If we found the known optimum, we can stop searching for smaller covers
                # and move to local search to verify minimality or just return.
                if best_len <= 51:
                    break

    # If no cover found (should not happen for n=11), fallback to a dense hypergraph
    if best_indices is None:
        # Return the first 100 edges as a fallback (valid but not minimal)
        # Actually, for n=11, the complete hypergraph is 2-colorable? No, 
        # Wait, is the complete 5-uniform hypergraph on 11 vertices 2-colorable?
        # If it is, we need a subset.
        # But the greedy should find a cover.
        # If greedy fails, we try a random subset of size 100?
        # Let's just return a large set to be safe, though it will score poorly.
        # Better: return the first 51 edges? No, they might not cover.
        # Let's return all edges? 462 edges.
        return {'n': n, 'edges': [list(e) for e in all_edges]}

    # Local Search: Prune redundant edges
    current_indices = set(best_indices)
    
    # Pruning Phase
    # We try to remove each edge and see if the union of the rest still covers full_mask.
    # To do this efficiently, we can maintain the union mask.
    # Union of current set.
    union_mask = 0
    for idx in current_indices:
        union_mask |= edge_masks[idx]
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle for randomization
        indices_list = list(current_indices)
        random.shuffle(indices_list)
        
        for idx in indices_list:
            if idx not in current_indices:
                continue
            
            # Try removing idx
            # union_without_idx = union_mask & ~edge_masks[idx]  <-- INCORRECT. 
            # Union is OR. Removing an element from an OR union is not simply AND NOT.
            # We must recompute the union of the remaining edges.
            # However, if we know that edge_masks[idx] has bits that are NOT covered by any other edge,
            # then removing it will uncover those bits.
            # So, if (edge_masks[idx] & ~union_of_others) != 0, it's essential.
            # union_of_others = union_mask & ~edge_masks[idx] is NOT union of others.
            
            # Correct way:
            # Compute union of all edges in current_indices except idx.
            # This is O(K) where K is number of edges.
            # Since K ~ 51, this is fast.
            
            union_others = 0
            for jdx in current_indices:
                if jdx == idx:
                    continue
                union_others |= edge_masks[jdx]
            
            if union_others == full_mask:
                # Redundant!
                current_indices.remove(idx)
                # Update global union_mask
                # union_mask = union_others (since we removed idx, and others cover all)
                union_mask = union_others
                improved = True
                break # Restart loop with new set
                
    # Swap Phase (Optional, but helps if pruning gets stuck in a local minimum of size > 51)
    # We can try to swap an edge i with an edge j not in the set, if it maintains coverage and allows future pruning.
    # Given time constraints, pruning is usually sufficient for n=11 to reach 51 from the greedy result.
    
    # Final check
    if time.time() > deadline:
        # Return current best
        pass

    # Convert indices to edges
    final_edges = [list(all_edges[idx]) for idx in current_indices]
    
    return {'n': n, 'edges': final_edges}

if __name__ == '__main__':
    construct()
