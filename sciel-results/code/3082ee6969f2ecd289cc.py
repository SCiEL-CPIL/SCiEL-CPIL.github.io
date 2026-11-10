import time
import random
import itertools
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    vertices = list(range(n))
    k4s = list(itertools.combinations(vertices, 4))
    num_k4s = len(k4s)
    
    # Edge list and index
    edge_list = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    num_edges = len(edge_list)
    
    # For each K4, store the list of edge indices
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in itertools.combinations(k4, 2):
            edges.append(edge_index[(a, b)])
        k4_edges.append(edges)
    
    # For each edge, the list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Precompute: for each K4, the 6 edge indices as a numpy array
    k4_edges_np = [np.array(e, dtype=np.int32) for e in k4_edges]
    
    # Helper: compute violation count and per-K4 color frequency
    def compute_violations(colours):
        """Returns (total_violations, per_k4_distinct_count, per_k4_color_freq)"""
        colours = np.array(colours, dtype=np.int32)
        total_viol = 0
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        # We'll store for each K4 the count of each color present
        # But that's expensive. Instead, just track distinct count.
        for k in range(num_k4s):
            edges = k4_edges_np[k]
            cvals = colours[edges]
            distinct = len(np.unique(cvals))
            distinct_counts[k] = distinct
            if distinct < 5:
                total_viol += 1
        return total_viol, distinct_counts
    
    def distinct_count_from_colours(colours, k):
        """Compute distinct color count for K4 k given current colours array."""
        edges = k4_edges_np[k]
        cvals = colours[edges]
        return len(np.unique(cvals))
    
    # Try structured colorings first
    def try_structured():
        """Try various structured colorings and return best valid one."""
        candidates = []
        
        # Sum mod p
        for p in range(13, 20):
            c = [((i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sum_mod_%d' % p))
        
        # Difference mod p
        for p in range(13, 20):
            c = [((i - j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'diff_mod_%d' % p))
        
        # Product mod p
        for p in range(13, 20):
            c = [((i * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'prod_mod_%d' % p))
        
        # Sum of squares mod p
        for p in range(13, 20):
            c = [((i * i + j * j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'sq_sum_mod_%d' % p))
        
        # i*j + i + j mod p
        for p in range(13, 20):
            c = [((i * j + i + j) % p) for i in range(n) for j in range(i + 1, n)]
            candidates.append((c, 'ijij_mod_%d' % p))
        
        # Hadamard-based: 4-bit Hadamard codes for 16 vertices
        # Hadamard matrix H_4 (Sylvester construction)
        # Map vertices to rows of H_4, edge color = Hamming weight of XOR
        H4 = [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ]
        # Convert to binary: 1->0, -1->1
        H4_bin = [[1 - b for b in row] for row in H4]
        # 16 vertices: all combinations of 4 bits
        vertices_codes = []
        for a in range(2):
            for b in range(2):
                for c2 in range(2):
                    for d in range(2):
                        vertices_codes.append([a, b, c2, d])
        
        # Edge color = Hamming weight of XOR of two vertex codes (0..4)
        # This gives colors 0-4, only 5 colors - might not be enough
        # Try: color = (hamming_weight * 3 + some_function) mod C
        for C in [13, 14, 15]:
            c = []
            for i in range(n):
                for j in range(i + 1, n):
                    xor = [vertices_codes[i][b] ^ vertices_codes[j][b] for b in range(4)]
                    hw = sum(xor)
                    # Use a hash-like function
                    val = (hw * 7 + i * 3 + j * 5) % C
                    c.append(val)
            candidates.append((c, 'hadamard_%d' % C))
        
        # Pairwise: use (i mod p, j mod p) mapping
        for p in [13, 14, 15]:
            c = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Map vertex to (i % p, i // p) pair
                    i1, i2 = i % p, i // p
                    j1, j2 = j % p, j // p
                    val = (i1 * j2 + i2 * j1) % p
                    c.append(val)
            candidates.append((c, 'pairwise_%d' % p))
        
        # Check each candidate
        best_valid = None
        best_num_colors = float('inf')
        for c, name in candidates:
            # Count distinct colors used
            num_used = len(set(c))
            if num_used >= best_num_colors:
                continue
            # Check validity
            valid = True
            for k in range(num_k4s):
                edges = k4_edges[k]
                cset = set(c[e] for e in edges)
                if len(cset) < 5:
                    valid = False
                    break
            if valid:
                best_valid = c
                best_num_colors = num_used
                # Can't do better than 13
                if best_num_colors <= 13:
                    return best_valid
        
        return best_valid
    
    # Fast local search with simulated annealing
    def local_search(colours, C, max_time=60.0):
        """
        colours: list of ints (edge colors)
        C: number of colors available
        Returns: (best_colours, best_cost, found_valid)
        """
        colours = list(colours)
        
        # Initial violation count
        total_viol, distinct_counts = compute_violations(colours)
        current_cost = total_viol
        
        best_colours = colours[:]
        best_cost = current_cost
        found_valid = (current_cost == 0)
        
        if found_valid:
            return best_colours, best_cost, True
        
        # Simulated annealing parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        max_iter = 500000
        iter_count = 0
        stall_count = 0
        max_stall = 2000
        
        # Per-K4 color frequency dicts
        freq = [dict() for _ in range(num_k4s)]
        for k in range(num_k4s):
            edges = k4_edges[k]
            for e in edges:
                c = colours[e]
                freq[k][c] = freq[k].get(c, 0) + 1
        
        # Maintain set of violating K4s incrementally
        viol_set = set(k for k in range(num_k4s) if distinct_counts[k] < 5)
        
        def compute_delta(e_idx, old_c, new_c):
            if old_c == new_c:
                return 0
            delta = 0
            for k in edge_to_k4s[e_idx]:
                old_distinct = distinct_counts[k]
                old_count = freq[k].get(old_c, 0)
                new_count = freq[k].get(new_c, 0)
                
                new_distinct = old_distinct
                if old_count == 1:
                    new_distinct -= 1
                if new_count == 0:
                    new_distinct += 1
                
                old_viol = 1 if old_distinct < 5 else 0
                new_viol = 1 if new_distinct < 5 else 0
                delta += new_viol - old_viol
            return delta
        
        def apply_move(e_idx, old_c, new_c):
            nonlocal current_cost
            if old_c == new_c:
                return
            colours[e_idx] = new_c
            for k in edge_to_k4s[e_idx]:
                old_count = freq[k].get(old_c, 0)
                if old_count == 1:
                    del freq[k][old_c]
                    distinct_counts[k] -= 1
                else:
                    freq[k][old_c] = old_count - 1
                
                new_count = freq[k].get(new_c, 0)
                if new_count == 0:
                    freq[k][new_c] = 1
                    distinct_counts[k] += 1
                else:
                    freq[k][new_c] = new_count + 1
                
                # Update viol_set
                if distinct_counts[k] < 5:
                    viol_set.add(k)
                elif k in viol_set:
                    viol_set.discard(k)
        
        def undo_move(e_idx, old_c, new_c):
            nonlocal current_cost
            if old_c == new_c:
                return
            colours[e_idx] = old_c
            for k in edge_to_k4s[e_idx]:
                new_count = freq[k].get(new_c, 0)
                if new_count == 1:
                    del freq[k][new_c]
                    distinct_counts[k] -= 1
                else:
                    freq[k][new_c] = new_count - 1
                
                old_count = freq[k].get(old_c, 0)
                if old_count == 0:
                    freq[k][old_c] = 1
                    distinct_counts[k] += 1
                else:
                    freq[k][old_c] = old_count + 1
                
                if distinct_counts[k] < 5:
                    viol_set.add(k)
                elif k in viol_set:
                    viol_set.discard(k)
        
        while iter_count < max_iter:
            if time.time() > deadline:
                break
            
            if not viol_set:
                found_valid = True
                best_cost = 0
                best_colours = colours[:]
                break
            
            k4_idx = random.choice(list(viol_set))
            
            # Try edges in this K4 with random subset of colors for speed
            best_delta = float('inf')
            best_move = None
            
            # Try all 6 edges, but sample colors to keep it fast
            for e_idx in k4_edges[k4_idx]:
                old_c = colours[e_idx]
                # Try a few random colors plus the old one's neighbors
                candidates = set()
                for _ in range(min(C, 8)):
                    candidates.add(random.randint(0, C - 1))
                # Also try colors 0..min(C,10)-1 for structured exploration
                for nc in range(min(C, 10)):
                    candidates.add(nc)
                
                for new_c in candidates:
                    if new_c == old_c:
                        continue
                    d = compute_delta(e_idx, old_c, new_c)
                    if d < best_delta:
                        best_delta = d
                        best_move = (e_idx, old_c, new_c)
            
            if best_move:
                e_idx, old_c, new_c = best_move
                if best_delta <= 0 or (T > T_min and random.random() < np.exp(-best_delta / T)):
                    apply_move(e_idx, old_c, new_c)
                    current_cost += best_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                e_idx = random.choice(k4_edges[k4_idx])
                old_c = colours[e_idx]
                new_c = random.randint(0, C - 1)
                if new_c != old_c:
                    d = compute_delta(e_idx, old_c, new_c)
                    apply_move(e_idx, old_c, new_c)
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colours = colours[:]
                        if best_cost == 0:
                            found_valid = True
                            break
                    stall_count = 0
            
            iter_count += 1
            T = max(T_min, T * cooling)
            
            if stall_count > max_stall:
                for _ in range(15):
                    e_idx = random.randint(0, num_edges - 1)
                    old_c = colours[e_idx]
                    new_c = random.randint(0, C - 1)
                    if new_c != old_c:
                        d = compute_delta(e_idx, old_c, new_c)
                        apply_move(e_idx, old_c, new_c)
                        current_cost += d
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colours = colours[:]
                            if best_cost == 0:
                                found_valid = True
                                break
                stall_count = 0
                T = 5.0
        
        if not found_valid and best_cost == 0:
            found_valid = True
        
        return best_colours, best_cost, found_valid
    
    def reduce_colors(colours, max_time=15.0):
        """Try to reduce the number of distinct colors in a valid coloring."""
        colours = list(colours)
        used_colors = set(colours)
        num_used = len(used_colors)
        
        if num_used <= 13:
            return colours, num_used
        
        # Build freq structures
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        freq = [dict() for _ in range(num_k4s)]
        for k in range(num_k4s):
            edges = k4_edges[k]
            for e in edges:
                c = colours[e]
                freq[k][c] = freq[k].get(c, 0) + 1
            distinct_counts[k] = len(freq[k])
        
        # Try to eliminate each color one by one
        # Sort by frequency (least frequent first) for better elimination chances
        remaining = sorted(used_colors, key=lambda c: sum(1 for e in range(num_edges) if colours[e] == c))
        
        for target_color in remaining:
            if time.time() > deadline:
                break
            if target_color not in set(colours):
                continue
            
            # Find all edges with this color
            target_edges = [e for e in range(num_edges) if colours[e] == target_color]
            if not target_edges:
                continue
            
            # Try to recolor each target edge to another color that keeps validity
            success = True
            for e_idx in target_edges:
                if time.time() > deadline:
                    break
                old_c = colours[e_idx]
                if old_c != target_color:
                    continue
                
                # Try other used colors
                found = False
                for new_c in used_colors:
                    if new_c == target_color:
                        continue
                    # Check if recoloring e_idx from target_color to new_c keeps all K4s valid
                    valid = True
                    for k in edge_to_k4s[e_idx]:
                        old_distinct = distinct_counts[k]
                        old_count = freq[k].get(target_color, 0)
                        new_count = freq[k].get(new_c, 0)
                        
                        new_distinct = old_distinct
                        if old_count == 1:
                            new_distinct -= 1
                        if new_count == 0:
                            new_distinct += 1
                        
                        if new_distinct < 5:
                            valid = False
                            break
                    
                    if valid:
                        # Apply
                        colours[e_idx] = new_c
                        for k in edge_to_k4s[e_idx]:
                            oc = freq[k].get(target_color, 0)
                            if oc == 1:
                                del freq[k][target_color]
                                distinct_counts[k] -= 1
                            else:
                                freq[k][target_color] = oc - 1
                            nc = freq[k].get(new_c, 0)
                            if nc == 0:
                                freq[k][new_c] = 1
                                distinct_counts[k] += 1
                            else:
                                freq[k][new_c] = nc + 1
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                # Remove target_color from used_colors
                used_colors.discard(target_color)
        
        final_used = len(set(colours))
        return colours, final_used
    
    # Main search
    best_result = None
    best_cost = float('inf')
    
    # Try structured first
    structured = try_structured()
    if structured is not None:
        num_used = len(set(structured))
        if num_used < best_cost:
            best_cost = num_used
            best_result = structured
        if best_cost <= 13:
            return best_result
    
    # Try local search with various C and initializations
    # Start with C=13 (minimum possible), then 14, 15
    for C in [13, 14, 15]:
        if time.time() > deadline:
            break
        
        # Multiple restarts - use more restarts for C=13 since it's the target
        num_restarts = 8 if C == 13 else 5
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            colours = [random.randint(0, C - 1) for _ in range(num_edges)]
            
            # Give more time to C=13
            ls_time = 25.0 if C == 13 else 20.0
            best_colours, best_c, found = local_search(colours, C, max_time=ls_time)
            
            if found:
                # Try to reduce colors
                red_time = 12.0 if C == 13 else 8.0
                reduced, reduced_num = reduce_colors(best_colours, max_time=red_time)
                if reduced_num < best_cost:
                    best_cost = reduced_num
                    best_result = reduced
                if best_cost <= 13:
                    return best_result
            else:
                pass
    
    # If no valid found, try with C=16, 17 as fallback
    for C in [16, 17]:
        if time.time() > deadline:
            break
        colours = [random.randint(0, C - 1) for _ in range(num_edges)]
        best_colours, best_c, found = local_search(colours, C, max_time=15.0)
        if found:
            reduced, reduced_num = reduce_colors(best_colours, max_time=8.0)
            if reduced_num < best_cost:
                best_cost = reduced_num
                best_result = reduced
    
    # Final fallback: seed
    if best_result is None:
        best_result = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify and return
    return best_result
