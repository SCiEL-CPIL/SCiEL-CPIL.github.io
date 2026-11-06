import numpy as np
import time
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safety margin

    # Precompute all weight-8 and weight-9 words
    weight_8_words = []
    weight_9_words = []
    for mask in range(1 << N):
        w = bin(mask).count('1')
        if w == 8:
            weight_8_words.append(mask)
        elif w == 9:
            weight_9_words.append(mask)
    
    all_vertices = weight_8_words + weight_9_words
    n_vertices = len(all_vertices)
    
    # Build adjacency: two vertices are connected if their Hamming distance < MIN_DIST
    # For efficiency, precompute masks for distances 0 to 5
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    # Build conflict sets: for each vertex, which other vertices conflict with it
    # Use bitsets for efficiency
    conflict_sets = [0] * n_vertices
    
    for i, vi in enumerate(all_vertices):
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                vj = vi ^ mask
                # Find if vj is in all_vertices
                # For efficiency, we could use a lookup table, but let's use a set
                # Actually, let's build a dict from word to index first
                pass
    
    # Build word-to-index mapping
    word_to_idx = {w: i for i, w in enumerate(all_vertices)}
    
    # Rebuild conflict sets using the mapping
    for i, vi in enumerate(all_vertices):
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                vj = vi ^ mask
                if vj in word_to_idx:
                    j = word_to_idx[vj]
                    if j != i:
                        conflict_sets[i] |= (1 << j)
    
    # Branch and bound for maximum independent set
    # Use coloring-based upper bound
    def branch_and_bound(vertices, current_set, best_set):
        if not vertices:
            if len(current_set) > len(best_set):
                best_set.extend(current_set)
            return best_set
        
        # Upper bound: |current_set| + |vertices|
        if len(current_set) + len(vertices) <= len(best_set):
            return best_set
        
        # Choose a vertex with minimum degree in the induced subgraph
        # For simplicity, pick the first vertex
        v = vertices[0]
        remaining = vertices[1:]
        
        # Branch 1: Include v
        new_current = current_set + [v]
        # Remove v and its neighbors from remaining
        neighbors_mask = conflict_sets[v]
        new_remaining = [u for u in remaining if not (neighbors_mask >> u) & 1]
        best_set = branch_and_bound(new_remaining, new_current, best_set)
        
        # Branch 2: Exclude v
        best_set = branch_and_bound(remaining, current_set, best_set)
        
        return best_set
    
    # Start with all vertices
    best_set = []
    branch_and_bound(list(range(n_vertices)), [], best_set)
    
    # Convert indices back to words
    code = [all_vertices[i] for i in best_set]
    
    # Verify and return
    if len(code) > 0:
        # Check that all pairs have distance >= MIN_DIST
        for i in range(len(code)):
            for j in range(i+1, len(code)):
                d = bin(code[i] ^ code[j]).count('1')
                if d < MIN_DIST:
                    # This shouldn't happen, but if it does, fall back to greedy
                    break
            else:
                continue
            break
        else:
            return code
    
    # Fallback: greedy
    blocked = bytearray(1 << N)
    code = []
    for w in range(1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1
    
    return code
