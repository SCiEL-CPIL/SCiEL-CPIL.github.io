import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # We will try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        # Precompute vertex masks
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        M0 = [0] * n
        M1 = [0] * n
        
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
                    
        # Generate all edges
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        # Greedy Set Cover with multiple strategies
        best_local_count = float('inf')
        best_local_edges = None
        
        # Adaptive restarts based on n and remaining time
        remaining_time = deadline - time.time()
        if n <= 10:
            restarts = min(100, int(remaining_time / 0.5))
        elif n <= 12:
            restarts = min(50, int(remaining_time / 1.0))
        else:
            restarts = min(20, int(remaining_time / 2.0))
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected_indices = []
            # Use a set for O(1) removal
            remaining_set = set(range(num_edges))
            
            # Greedy with tie-breaking by randomization
            while covered != all_ones:
                best_gain = -1
                best_candidates = []
                
                for idx in remaining_set:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [idx]
                    elif gain == best_gain:
                        best_candidates.append(idx)
                            
                if best_gain == 0:
                    break # Failed to cover all
                    
                # Randomly pick from best candidates for diversity
                chosen = random.choice(best_candidates)
                selected_indices.append(chosen)
                covered |= edge_masks[chosen]
                remaining_set.discard(chosen)
                
            if covered == all_ones:
                # Local search: try to swap edges to reduce count
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    
                    # First pass: remove redundant edges
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break
                    
                    # Second pass: try swapping one edge for another that covers more
                    if not improved and len(selected_indices) > 1:
                        for i in range(len(selected_indices)):
                            if time.time() > deadline:
                                break
                            e_idx = selected_indices[i]
                            # Coverage if we remove this edge
                            base_covered = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    base_covered |= edge_masks[selected_indices[j]]
                            
                            # Find an edge that covers the gap
                            gap = all_ones & ~base_covered
                            if gap == 0:
                                # This edge is redundant, should have been caught above
                                selected_indices.pop(i)
                                covered = base_covered
                                improved = True
                                break
                            
                            # Look for a replacement edge that covers the gap and isn't already selected
                            best_replace = -1
                            best_replace_gain = 0
                            for idx in range(num_edges):
                                if idx in selected_indices:
                                    continue
                                gain = (edge_masks[idx] & gap).bit_count()
                                if gain > best_replace_gain:
                                    best_replace_gain = gain
                                    best_replace = idx
                            
                            if best_replace >= 0 and best_replace_gain == gap.bit_count():
                                selected_indices[i] = best_replace
                                covered = all_ones
                                improved = True
                                break
        
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                print(f"Found better: n={n}, edges={best_count}")
                
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
