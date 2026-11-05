import time
import random
import itertools
from collections import Counter

def construct():
    deadline = time.time() + 135  # 15s safety margin before 150s limit
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx

    # Precompute all K4s and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    n_k4s = len(k4s)
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # Helper to check validity
    def is_valid(colors):
        for k4_idx in range(n_k4s):
            edges = k4_edges[k4_idx]
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    # Local search with proper state tracking
    def local_search(num_colors, max_iter=100000):
        colors = [random.randint(0, num_colors - 1) for _ in range(n_edges)]
        
        # For each K4, track the count of each color
        # k4_color_counts[k4_idx] is a Counter of colors in that K4
        k4_color_counts = []
        for k4_idx in range(n_k4s):
            edges = k4_edges[k4_idx]
            counts = Counter(colors[e] for e in edges)
            k4_color_counts.append(counts)
        
        # Track number of violations (K4s with < 5 distinct colors)
        def count_violations():
            v = 0
            for k4_idx in range(n_k4s):
                if len(k4_color_counts[k4_idx]) < 5:
                    v += 1
            return v
        
        violations = count_violations()
        
        if violations == 0:
            return colors[:]
        
        T = 5.0
        cooling = 0.9999
        it = 0
        
        while it < max_iter and time.time() < deadline - 5:
            it += 1
            e_idx = random.randint(0, n_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, num_colors - 1)
            if old_c == new_c:
                continue
            
            # Calculate delta in violations
            delta_v = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Save old states
            old_states = {}
            for k4_idx in affected_k4s:
                old_states[k4_idx] = k4_color_counts[k4_idx].copy()
            
            # Apply change tentatively
            colors[e_idx] = new_c
            
            # Update affected K4s
            for k4_idx in affected_k4s:
                counts = k4_color_counts[k4_idx]
                old_len = len(counts)
                
                # Remove old_c
                if counts[old_c] > 0:
                    counts[old_c] -= 1
                    if counts[old_c] == 0:
                        del counts[old_c]
                
                # Add new_c
                counts[new_c] = counts.get(new_c, 0) + 1
                
                new_len = len(counts)
                
                if old_len < 5 and new_len >= 5:
                    delta_v -= 1
                elif old_len >= 5 and new_len < 5:
                    delta_v += 1
            
            violations += delta_v
            
            if violations == 0:
                return colors[:]
            
            # Acceptance criterion
            if delta_v <= 0:
                pass  # Accept
            else:
                # Metropolis criterion
                if random.random() < pow(2.71828, -delta_v / T):
                    pass  # Accept
                else:
                    # Revert
                    colors[e_idx] = old_c
                    for k4_idx in affected_k4s:
                        k4_color_counts[k4_idx] = old_states[k4_idx]
                    violations -= delta_v
            
            T *= cooling
        
        return None

    # Try to find valid colorings with decreasing number of colors
    best_valid_colours = None
    best_num_colors = 120
    
    # Try 13, 14, 15 colors
    for num_colors in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            res = local_search(num_colors, max_iter=30000)
            if res:
                best_valid_colours = res
                best_num_colors = num_colors
                break
        if best_valid_colours and best_num_colors == 13:
            break  # Can't do better than 13
    
    # If we found a valid coloring, try to reduce colors further
    if best_valid_colours:
        colors = best_valid_colours[:]
        
        def try_merge(colors, c1, c2):
            new_colors = [c1 if c == c2 else c for c in colors]
            if is_valid(new_colors):
                return new_colors
            return None
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_used = sorted(set(colors))
            
            # Try to merge each pair
            for i in range(len(current_used)):
                for j in range(i + 1, len(current_used)):
                    c1 = current_used[i]
                    c2 = current_used[j]
                    new_colors = try_merge(colors, c1, c2)
                    if new_colors:
                        colors = new_colors
                        improved = True
                        best_num_colors = len(set(colors))
                        break
                if improved:
                    break
                    
        if best_num_colors < 15:
            best_valid_colours = colors

    # Fallback: Try 16 colors
    if best_valid_colours is None:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            res = local_search(16, max_iter=20000)
            if res:
                best_valid_colours = res
                best_num_colors = 16
                break

    # Last resort: Return a valid coloring with many colors
    if best_valid_colours is None:
        # Color (i,j) with i*16+j. 120 colors. Valid.
        best_valid_colours = [i * 16 + j for i in range(16) for j in range(i + 1, 16)]
        best_num_colors = 120

    return best_valid_colours
