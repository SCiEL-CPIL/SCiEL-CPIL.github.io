"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    num_edges = len(edges)
    
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        e_list = [edge_idx[(a, b)] for a, b in combinations(quad, 2)]
        k4_edges.append(e_list)
    num_k4s = len(k4_edges)
    
    # Start with base coloring
    colors = [(i + j) % 17 for i, j in edges]
    best_colors = colors[:]
    best_count = len(set(best_colors))
    
    # For each edge, precompute which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k4e in enumerate(k4_edges):
        for e in k4e:
            edge_to_k4s[e].append(k4_idx)
    
    def count_k4_colors(k4e, c):
        return len(set(c[e] for e in k4e))
    
    # Try to reduce color count by remapping
    # First, get the frequency of each color
    for target in range(13, best_count + 1):
        if time.time() > deadline - 10:
            break
        
        # Get unique colors and their frequencies
        freq = Counter(colors)
        unique_colors = sorted(freq.keys(), key=lambda x: -freq[x])
        
        # Map to target colors: keep the most frequent `target` colors, merge rest
        if len(unique_colors) <= target:
            # Already using <= target colors
            color_map = {c: i for i, c in enumerate(unique_colors)}
            remapped = [color_map[c] for c in colors]
            # Check validity
            valid = True
            for k4e in k4_edges:
                if count_k4_colors(k4e, remapped) < 5:
                    valid = False
                    break
            if valid:
                best_colors = remapped[:]
                best_count = len(set(remapped))
                break
            continue
        
        # Keep top `target` colors, merge others into them
        keep = set(unique_colors[:target])
        color_map = {}
        for c in unique_colors:
            if c in keep:
                color_map[c] = c
            else:
                # Merge into the most frequent kept color
                best_keep = max(keep, key=lambda x: freq[x])
                color_map[c] = best_keep
        
        remapped = [color_map[c] for c in colors]
        valid = True
        for k4e in k4_edges:
            if count_k4_colors(k4e, remapped) < 5:
                valid = False
                break
        if valid:
            best_colors = remapped[:]
            best_count = len(set(remapped))
            break
    
    # If still using many colors, try local search to reduce
    if best_count > 13 and time.time() < deadline - 20:
        random.seed(123)
        current = best_colors[:]
        
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            # Try to reduce by one color
            target = best_count - 1
            if target < 13:
                break
            
            # Random remapping
            freq = Counter(current)
            unique_colors = sorted(freq.keys(), key=lambda x: -freq[x])
            if len(unique_colors) <= target:
                continue
            
            keep = set(unique_colors[:target])
            color_map = {}
            for c in unique_colors:
                if c in keep:
                    color_map[c] = c
                else:
                    best_keep = max(keep, key=lambda x: freq[x])
                    color_map[c] = best_keep
            
            remapped = [color_map[c] for c in current]
            
            # Check and fix violations
            violations = []
            for k4_idx, k4e in enumerate(k4_edges):
                if count_k4_colors(k4e, remapped) < 5:
                    violations.append(k4_idx)
            
            if not violations:
                best_colors = remapped[:]
                best_count = len(set(remapped))
                current = remapped[:]
                continue
            
            # Try to fix violations by changing edge colors
            fixed = False
            for v_idx in violations[:10]:  # Limit fixes per trial
                k4e = k4_edges[v_idx]
                for e in k4e:
                    old_c = remapped[e]
                    for new_c in keep:
                        if new_c == old_c:
                            continue
                        remapped[e] = new_c
                        # Quick check
                        ok = True
                        for k4_idx2, k4e2 in enumerate(k4_edges):
                            if count_k4_colors(k4e2, remapped) < 5:
                                ok = False
                                break
                        if ok:
                            fixed = True
                            break
                        remapped[e] = old_c
                    if fixed:
                        break
                if not fixed:
                    break
            
            if fixed:
                # Re-check full validity
                valid = True
                for k4e in k4_edges:
                    if count_k4_colors(k4e, remapped) < 5:
                        valid = False
                        break
                if valid:
                    best_colors = remapped[:]
                    best_count = len(set(remapped))
                    current = remapped[:]
    
    return best_colors
# EVOLVE-BLOCK-END
