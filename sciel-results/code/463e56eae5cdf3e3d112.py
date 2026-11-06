import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # We will try n from 9 to 16
    # Smaller n has fewer edges total, but might require a denser hypergraph to be non-2-colorable
    # Larger n has more edges to choose from, allowing sparser hypergraphs
    for n in range(9, 17):
        if time.time() > deadline:
            break
            
        # Precompute all edges
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Number of colorings is 2^n
        total_colorings = 1 << n
        all_ones_mask = (1 << total_colorings) - 1
        
        # Precompute edge masks
        # edge_masks[i] is a bitmask of colorings where edge i is monochromatic
        edge_masks = []
        
        # To speed up, we can iterate colorings and update masks?
        # Or iterate edges and compute mask.
        # Computing mask for one edge:
        # A coloring c is monochromatic on edge e if all vertices in e have same color.
        # This happens if (c & mask_e) == 0 or (c & mask_e) == mask_e, where mask_e is the bitmask of vertices in e.
        # We need to build a bitset of size 2^n.
        
        # Efficient way to build the bitset:
        # For a given edge e (set of vertices), the colorings where it is monochromatic are those
        # where all bits in e are 0 OR all bits in e are 1.
        
        # Let's build edge_masks
        for e in all_edges:
            mask_e = 0
            for v in e:
                mask_e |= (1 << v)
            
            # We need a bitset B of length 2^n such that B[c] = 1 if c is monochromatic on e.
            # c is an integer 0..2^n-1.
            # B is an integer with 2^n bits.
            # We can construct B by iterating c? That would be 2^n * num_edges, which is too slow.
            # num_edges for n=16 is C(16,5) = 4368. 2^16 = 65536.
            # 4368 * 65536 is ~ 280 million operations. Too slow in Python.
            
            # Alternative: Use the structure.
            # The set of colorings where e is monochromatic is the union of:
            # 1. Colorings where all v in e are 0.
            # 2. Colorings where all v in e are 1.
            
            # Let's generate the bitset for "all v in e are 0".
            # This is a pattern.
            
            # Actually, we can precompute for each vertex v, the bitset of colorings where v is 0, and where v is 1.
            # Let M0[v] be bitset of colorings where v=0.
            # Let M1[v] be bitset of colorings where v=1.
            # Then mask for edge e = AND_{v in e} M0[v] OR AND_{v in e} M1[v].
            
            # Precomputing M0 and M1 for all v is fast.
            # M0[v] has 1s at positions c where (c >> v) & 1 == 0.
            # This is a regular pattern.
            
            # Let's precompute M0 and M1 once per n.
            if n == 9:
                # Re-initialize if n changes
                pass
                
        # Precompute M0 and M1
        M0 = [0] * n
        M1 = [0] * n
        
        # For each vertex v, M1[v] has bits set where bit v of c is 1.
        # Pattern: For n bits, the bit v of c is 1 for c in [2^v, 2^(v+1)-1], [2^(v+2), 2^(v+3)-1], etc.
        # But we are building a bitset of length 2^n.
        # Let's just iterate c for each v? No, that's n * 2^n.
        # For n=16, 16 * 65536 = 1 million. This is fast enough.
        
        for v in range(n):
            m0 = 0
            m1 = 0
            for c in range(total_colorings):
                if (c >> v) & 1:
                    m1 |= (1 << c)
                else:
                    m0 |= (1 << c)
            M0[v] = m0
            M1[v] = m1
            
        # Now compute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        # Now solve Set Cover
        # Universe is all_ones_mask
        # Sets are edge_masks
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts
        # For small n, we can do more
        restarts = 100 if n <= 12 else 50
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected_indices = []
            remaining = list(range(num_edges))
            random.shuffle(remaining)
            
            # Greedy Set Cover
            while covered != all_ones_mask:
                best_gain = -1
                best_idx = -1
                
                # To speed up, we can sample if remaining is large?
                # But num_edges is at most 4368. Iterating 4368 items is fast.
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if best_gain == 0:
                            # No gain, stuck?
                            # This shouldn't happen if a cover exists.
                            pass
                            
                if best_gain == 0:
                    break # Failed to cover all
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones_mask:
                # Refine: try to remove redundant edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Try removing each edge in random order
                    random.shuffle(selected_indices)
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        # Check if e_idx is redundant
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                                
                        if new_covered == all_ones_mask:
                            selected_indices.pop(i)
                            improved = True
                            break # Restart refinement loop
                    
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
        # Fallback to a known valid construction if nothing found
        # e.g. n=9, all edges
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
