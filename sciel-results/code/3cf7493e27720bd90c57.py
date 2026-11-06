import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We target n=12, which is known to support a 51-edge non-2-colorable hypergraph.
    # We will use the bitset set-cover approach which is very fast for n=12.
    
    n = 12
    total_colorings = 1 << n
    all_ones = (1 << total_colorings) - 1
    
    # Precompute M0 and M1
    M0 = [0] * n
    M1 = [0] * n
    
    for c in range(total_colorings):
        for v in range(n):
            if (c >> v) & 1:
                M1[v] |= (1 << c)
            else:
                M0[v] |= (1 << c)
                
    all_edges = list(combinations(range(n), 5))
    
    # Precompute edge masks
    edge_masks = []
    for e in all_edges:
        m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
        m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
        edge_masks.append(m0 | m1)
        
    best_count = float('inf')
    best_edges = None
    
    # Run multiple restarts of greedy + local deletion
    # n=12 is small enough to do many restarts
    restarts = 200 if time.time() < deadline - 5 else 20
    
    for r in range(restarts):
        if time.time() > deadline:
            break
            
        random.seed(42 + r) # Deterministic restarts for reproducibility
        
        covered = 0
        selected_indices = []
        remaining = list(range(len(all_edges)))
        random.shuffle(remaining)
        
        # Greedy Set Cover
        while covered != all_ones and remaining:
            best_gain = -1
            best_idx = -1
            
            # Optimization: If remaining is large, we can sample or just iterate.
            # 792 edges is small, iterate all.
            for idx in remaining:
                gain = (edge_masks[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                break # Failed to cover
                
            selected_indices.append(best_idx)
            covered |= edge_masks[best_idx]
            remaining.remove(best_idx)
            
        if covered == all_ones:
            # Local Search: Delete redundant edges
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing edges in random order to escape local minima?
                # Or just sequential.
                for i in range(len(selected_indices) - 1, -1, -1):
                    e_idx = selected_indices[i]
                    # Check if e_idx is redundant
                    new_covered = 0
                    for j in range(len(selected_indices)):
                        if j != i:
                            new_covered |= edge_masks[selected_indices[j]]
                            
                    if new_covered == all_ones:
                        selected_indices.pop(i)
                        covered = new_covered
                        improved = True
                        break # Restart deletion loop after a removal
                        
            if len(selected_indices) < best_count:
                best_count = len(selected_indices)
                best_edges = [list(all_edges[i]) for i in selected_indices]
                
                # If we hit the known minimum, we can stop early
                if best_count <= 51:
                    # We might try to go lower, but 51 is the known bound.
                    # Let's run a few more iterations to see if we can beat 51.
                    if best_count < 51:
                        break
                    else:
                        # Continue to see if we can find 50
                        pass

    if best_edges is None:
        # Fallback: Return a large hypergraph that is definitely non-2-colorable
        # e.g. all edges on n=9
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)
    else:
        best_n = n

    return {'n': best_n, 'edges': best_edges}
