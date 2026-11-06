<<<<<<< SEARCH
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Focus on n from 9 to 13 where search is tractable
    for n in range(9, 14):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        total_colorings = 1 << n
        all_ones_mask = (1 << total_colorings) - 1
        
        # Precompute M0 and M1 efficiently using pattern generation
        M0 = [0] * n
        M1 = [0] * n
        
        for v in range(n):
            half = 1 << v
            period = 1 << (v + 1)
            # In each period of length 2^(v+1), positions half..period-1 have bit v = 1
            pattern = ((1 << half) - 1) << half
            num_periods = total_colorings // period
            m1 = 0
            for p in range(num_periods):
                m1 |= (pattern << (p * period))
            M1[v] = m1
            M0[v] = all_ones_mask ^ m1
        
        # Compute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # More restarts for smaller n
        restarts = 300 if n <= 10 else 150 if n <= 11 else 80
        
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
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if best_gain == 0:
                            break
                            
                if best_gain <= 0:
                    break
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones_mask:
                # Refinement: remove redundant edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    random.shuffle(selected_indices)
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                        if new_covered == all_ones_mask:
                            selected_indices.pop(i)
                            improved = True
                            break
                
                # Local search: try swapping edges to reduce count
                if time.time() < deadline:
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        random.shuffle(selected_indices)
                        improved_swap = False
                        for i in range(len(selected_indices)):
                            if time.time() > deadline:
                                break
                            e_idx = selected_indices[i]
                            without_e = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    without_e |= edge_masks[selected_indices[j]]
                            uncovered = all_ones_mask & ~without_e
                            if uncovered == 0:
                                selected_indices.pop(i)
                                improved_swap = True
                                break
                            # Try to find a replacement
                            for j in range(num_edges):
                                if j in selected_indices:
                                    continue
                                if (edge_masks[j] & uncovered) != 0:
                                    new_covered = without_e | edge_masks[j]
                                    if new_covered == all_ones_mask:
                                        selected_indices[i] = j
                                        improved_swap = True
                                        break
                            if improved_swap:
                                break
                        if not improved_swap:
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
=======
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Focus on n from 9 to 13 where search is tractable
    for n in range(9, 14):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        total_colorings = 1 << n
        num_words = (total_colorings + 63) // 64  # number of 64-bit words
        all_ones_mask = (1 << total_colorings) - 1
        
        # Precompute M0 and M1 as arrays of 64-bit words
        # M1[v] is a bitset of length total_colorings
        M1_words = np.zeros((n, num_words), dtype=np.uint64)
        
        for v in range(n):
            half = 1 << v
            period = 1 << (v + 1)
            # Build the bitmask for vertex v
            m1 = 0
            for pos in range(total_colorings):
                if (pos >> v) & 1:
                    m1 |= (1 << pos)
            # Convert to words
            for w in range(num_words):
                M1_words[v, w] = (m1 >> (64 * w)) & 0xFFFFFFFFFFFFFFFF
        
        # Compute edge masks as 2D array: (num_edges, num_words)
        edge_masks_words = np.zeros((num_edges, num_words), dtype=np.uint64)
        
        for i, e in enumerate(all_edges):
            # m0 = AND of M0 for all 5 vertices
            # m1 = AND of M1 for all 5 vertices
            m0_words = np.full(num_words, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
            m1_words = np.zeros(num_words, dtype=np.uint64)
            for v in e:
                m1_words &= M1_words[v]
                m0_words &= ~M1_words[v]
            # Mask off extra bits in the last word
            extra_bits = 64 * num_words - total_colorings
            if extra_bits > 0:
                last_mask = (1 << (64 - extra_bits)) - 1
                m0_words[-1] &= last_mask
                m1_words[-1] &= last_mask
            edge_masks_words[i] = m0_words | m1_words
        
        # all_ones as word array
        all_ones_words = np.full(num_words, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        extra_bits = 64 * num_words - total_colorings
        if extra_bits > 0:
            all_ones_words[-1] = (1 << (64 - extra_bits)) - 1
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # More restarts for smaller n
        restarts = 500 if n <= 10 else 250 if n <= 11 else 120
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered_words = np.zeros(num_words, dtype=np.uint64)
            selected_indices = []
            remaining = list(range(num_edges))
            random.shuffle(remaining)
            
            # Greedy Set Cover using NumPy
            while True:
                if np.array_equal(covered_words, all_ones_words):
                    break
                
                # Compute gain for each remaining edge: popcount of (edge_mask & ~covered)
                not_covered = ~covered_words
                # For each remaining edge, compute the bitwise AND with not_covered
                gains = np.zeros(len(remaining), dtype=np.int64)
                for k, idx in enumerate(remaining):
                    bm = edge_masks_words[idx] & not_covered
                    # Popcount: count set bits across all words
                    for w in range(num_words):
                        gains[k] += np.uint64(bm[w]).bit_count() if hasattr(np.uint64(bm[w]), 'bit_count') else bin(int(bm[w])).count('1')
                
                if gains.max() <= 0:
                    break
                    
                best_k = int(np.argmax(gains))
                best_idx = remaining[best_k]
                
                selected_indices.append(best_idx)
                covered_words |= edge_masks_words[best_idx]
                remaining.pop(best_k)
                
            if np.array_equal(covered_words, all_ones_words):
                # Refinement: remove redundant edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    random.shuffle(selected_indices)
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        # Check if removing this edge still covers everything
                        new_covered = np.zeros(num_words, dtype=np.uint64)
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks_words[selected_indices[j]]
                        if np.array_equal(new_covered, all_ones_words):
                            selected_indices.pop(i)
                            improved = True
                            break
                
                # Local search: try swapping edges to reduce count
                if time.time() < deadline:
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        random.shuffle(selected_indices)
                        improved_swap = False
                        for i in range(len(selected_indices)):
                            if time.time() > deadline:
                                break
                            e_idx = selected_indices[i]
                            without_e = np.zeros(num_words, dtype=np.uint64)
                            for j in range(len(selected_indices)):
                                if j != i:
                                    without_e |= edge_masks_words[selected_indices[j]]
                            uncovered = all_ones_words & ~without_e
                            if not np.any(uncovered):
                                selected_indices.pop(i)
                                improved_swap = True
                                break
                            # Try to find a replacement
                            found = False
                            for j in range(num_edges):
                                if j in selected_indices:
                                    continue
                                if np.any(edge_masks_words[j] & uncovered):
                                    new_covered = without_e | edge_masks_words[j]
                                    if np.array_equal(new_covered, all_ones_words):
                                        selected_indices[i] = j
                                        improved_swap = True
                                        found = True
                                        break
                            if found:
                                break
                        if not improved_swap:
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
>>>>>>> REPLACE

The key improvement is using NumPy for the bitwise operations in the greedy set cover. However, I realize the popcount computation in the inner loop is still slow. Let me refine this further with a better vectorized approach:

