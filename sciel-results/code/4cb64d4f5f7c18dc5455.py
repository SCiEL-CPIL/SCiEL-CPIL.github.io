import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)
    
    # Try n from 9 to 16
    for n in range(9, 17):
        if time.time() > deadline - 5.0:
            break
            
        # Compute M0, M1
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        M0 = [0] * n
        M1 = [0] * n
        
        # Optimization: Generate M0, M1
        # M1[v] has bit c set if bit v of c is 1.
        # This is a regular pattern.
        # For v=0, bits 1,3,5... are 1.
        # We can generate these patterns more efficiently?
        # Looping is fine for n<=16.
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
        
        # Greedy Set Cover
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts
        # Smaller n -> more restarts
        if n <= 11:
            restarts = 100
        elif n <= 13:
            restarts = 50
        else:
            restarts = 20
            
        num_edges = len(all_edges)
        
        for r in range(restarts):
            if time.time() > deadline - 5.0:
                break
                
            covered = 0
            selected = []
            # Use a set for remaining? Or list?
            # List is faster for iteration, but removal is slow.
            # Since we iterate all remaining to find best gain, we can just
            # keep a list of indices and mark used ones.
            # Or simply:
            remaining_indices = list(range(num_edges))
            random.shuffle(remaining_indices)
            
            while covered != all_ones:
                best_gain = -1
                best_idx_in_remaining = -1
                best_edge_idx = -1
                
                # Iterate through remaining
                # To speed up, we can break early if we find a gain that covers everything?
                # No, we want max gain.
                
                for i, idx in enumerate(remaining_indices):
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_in_remaining = i
                        best_edge_idx = idx
                        
                if best_gain == 0:
                    break # Failed to cover all
                    
                selected.append(best_edge_idx)
                covered |= edge_masks[best_edge_idx]
                remaining_indices.pop(best_idx_in_remaining)
                
            if covered == all_ones:
                # Refinement: Remove redundant edges
                improved = True
                while improved:
                    improved = False
                    # Try removing from the end (arbitrary order)
                    for i in range(len(selected) - 1, -1, -1):
                        # Check if selected[i] is redundant
                        # Calculate OR of all others
                        new_covered = 0
                        for j in range(len(selected)):
                            if j != i:
                                new_covered |= edge_masks[selected[j]]
                        
                        if new_covered == all_ones:
                            selected.pop(i)
                            improved = True
                            break # Restart refinement loop
                    
                if len(selected) < best_local_count:
                    best_local_count = len(selected)
                    best_local_edges = [all_edges[i] for i in selected]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                # print(f"Found better: n={n}, edges={best_count}")
                
    return {'n': best_n, 'edges': best_edges}
