import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)

    # Known best is 51 edges. We aim to find <= 51.
    # Start with a known good solution if we can find one quickly.
    
    for n in range(9, 14):
        if time.time() > deadline - 15.0:
            break
            
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        # Precompute edge masks using NumPy for speed
        all_edges = list(combinations(range(n), 5))
        E = len(all_edges)
        
        # Build edge_masks: for each edge, which colorings make it monochromatic
        # Use NumPy to vectorize
        edge_masks = np.zeros(E, dtype=np.uint64)
        
        # For each edge, compute the mask of colorings where it's monochromatic
        # A coloring c makes edge e monochromatic if all vertices in e have same color
        # i.e., (c & e_mask) == 0 or (c & e_mask) == e_mask
        # where e_mask has bits set for vertices in e
        
        # Precompute vertex bit positions
        for i, e in enumerate(all_edges):
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
            
            # Count how many colorings have all bits in e_mask set to 0
            # and how many have all bits set to 1
            # For a 5-uniform edge, the number of colorings where all 5 vertices are 0:
            # The remaining (n-5) vertices can be anything: 2^(n-5)
            # Similarly for all 1: 2^(n-5)
            # But we need the actual bit positions in the coloring space
            
            # Colorings where edge is all 0: bits 0-4 of e_mask are 0
            # This is a pattern in the n-bit coloring
            # We need to iterate or use a smarter approach
            
            # Since n <= 13, total_colorings <= 8192, we can afford to compute per edge
            # but let's vectorize across edges
            
            m0 = 0
            m1 = 0
            for c in range(total_colorings):
                is_all_0 = True
                is_all_1 = True
                for v in e:
                    if (c >> v) & 1:
                        is_all_0 = False
                    else:
                        is_all_1 = False
                if is_all_0:
                    m0 |= (1 << c)
                if is_all_1:
                    m1 |= (1 << c)
            edge_masks[i] = m0 | m1
        
        # Convert to Python ints for bit operations, but keep NumPy array for vectorized gain
        edge_masks_list = [int(m) for m in edge_masks]
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # More restarts since NumPy speeds up greedy
        restarts = 150 if n <= 10 else (80 if n <= 11 else 40)
        
        for r in range(restarts):
            if time.time() > deadline - 8.0:
                break
                
            # 1. Initial Greedy Set Cover using NumPy vectorization
            covered = 0
            selected_indices = []
            remaining = list(range(E))
            random.shuffle(remaining)
            
            # Precompute edge_masks as numpy array for vectorized gain computation
            em_arr = np.array(edge_masks_list, dtype=np.uint64)
            
            while covered != all_ones:
                # Vectorized gain computation
                # gain[i] = popcount(edge_masks[i] & ~covered)
                # NumPy doesn't have built-in popcount for uint64, but we can use
                # a lookup table or bit_count via Python
                # For E up to ~1287 (n=13), this is manageable
                
                # Use list comprehension with int.bit_count()
                gains = [int(em_arr[idx] & ~covered).bit_count() for idx in remaining]
                
                if not gains or max(gains) == 0:
                    break
                    
                # Find index with max gain
                best_gain_idx = max(range(len(gains)), key=lambda i: gains[i])
                best_gain = gains[best_gain_idx]
                best_idx = remaining[best_gain_idx]
                
                selected_indices.append(best_idx)
                covered |= edge_masks_list[best_idx]
                remaining.pop(best_gain_idx)
                
            if covered != all_ones:
                continue
                
            # 2. Local Search: Remove and Repair
            current_S = selected_indices[:]
            current_size = len(current_S)
            no_improve_count = 0
            
            max_ls_iter = 300 if n <= 10 else 150
            
            for ls_iter in range(max_ls_iter):
                if time.time() > deadline - 3.0:
                    break
                
                if not current_S:
                    break
                    
                # Pick a random edge to try removing
                idx_to_remove = random.choice(current_S)
                
                # Remove it
                temp_S = [x for x in current_S if x != idx_to_remove]
                
                # Calculate uncovered
                covered = 0
                for idx in temp_S:
                    covered |= edge_masks_list[idx]
                
                uncovered = all_ones & ~covered
                
                if uncovered == 0:
                    # Edge was redundant
                    current_S = temp_S
                    current_size -= 1
                    no_improve_count = 0
                    continue
                
                # Repair: Add edges to cover 'uncovered'
                new_S = list(temp_S)
                S_set = set(new_S)
                
                # Greedy repair with vectorized gain
                while uncovered != 0:
                    # Get candidates not in S_set
                    candidates = [idx for idx in range(E) if idx not in S_set]
                    if not candidates:
                        break
                    
                    gains = [int(edge_masks_list[idx] & uncovered).bit_count() for idx in candidates]
                    if max(gains) == 0:
                        break
                        
                    best_gain_idx = max(range(len(gains)), key=lambda i: gains[i])
                    best_idx = candidates[best_gain_idx]
                    
                    new_S.append(best_idx)
                    S_set.add(best_idx)
                    uncovered &= ~edge_masks_list[best_idx]
                    
                if uncovered == 0 and len(new_S) < current_size:
                    current_S = new_S
                    current_size = len(new_S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 25:
                        break
                        
            if current_size < best_local_count:
                best_local_count = current_size
                best_local_edges = [all_edges[i] for i in current_S]
                
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]

    # Fallback
    if best_edges is None or len(best_edges) > 200:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}

