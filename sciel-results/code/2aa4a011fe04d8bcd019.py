import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n = 16
    # Edge index: for (i,j) with i<j, idx = i*n + j
    def eidx(i, j):
        if i > j:
            i, j = j, i
        return i * n + j
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_pos = {e: k for k, e in enumerate(all_edges)}
    
    # Precompute all K4s and their 6 edge positions
    all_k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in all_k4s:
        edges = [edge_pos[(a, b)] for a, b in combinations(k4, 2)]
        k4_edges.append(edges)
    
    # For each edge, list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(len(all_edges))]
    for ki, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(ki)
    
    # Build initial 1-factorization (15 colors)
    colors = np.full(len(all_edges), -1, dtype=np.int32)
    for k in range(15):
        colors[edge_pos[(15, k)]] = k
        for i in range(1, 8):
            a = (k + i) % 15
            b = (k - i) % 15
            if a > b:
                a, b = b, a
            colors[edge_pos[(a, b)]] = k
    
    def check_k4(ki, col):
        """Check if K4 ki has >= 5 distinct colors"""
        edges = k4_edges[ki]
        c0 = col[edges[0]]
        c1 = col[edges[1]]
        c2 = col[edges[2]]
        c3 = col[edges[3]]
        c4 = col[edges[4]]
        c5 = col[edges[5]]
        # Count distinct
        distinct = 1
        if c1 != c0: distinct += 1
        if c2 != c0 and c2 != c1: distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        return distinct >= 5
    
    def is_valid(col):
        for ki in range(len(k4_edges)):
            if not check_k4(ki, col):
                return False
        return True
    
    # Verify initial
    assert is_valid(colors)
    
    best = colors.copy()
    best_used = set(best.tolist())
    best_count = len(best_used)
    
    rng = random.Random(12345)
    
    # Local search: try to reduce color count
    # Strategy: pick an edge, try recoloring to an existing color to reduce total
    current = best.copy()
    current_used = set(current.tolist())
    
    for it in range(500000):
        if time.time() > deadline - 3:
            break
        
        # Pick random edge
        e = rng.randint(0, len(all_edges) - 1)
        old_c = current[e]
        
        # Try colors: prefer ones already in use
        candidates = list(current_used)
        if old_c in candidates:
            candidates.remove(old_c)
        # Add some new colors
        for c in range(16):
            if c not in current_used:
                candidates.append(c)
        rng.shuffle(candidates)
        
        for new_c in candidates:
            if new_c == old_c:
                continue
            
            # Apply
            current[e] = new_c
            
            # Check affected K4s
            valid = True
            for ki in edge_to_k4s[e]:
                if not check_k4(ki, current):
                    valid = False
                    break
            
            if valid:
                new_used = set(current.tolist())
                if len(new_used) <= len(current_used):
                    current_used = new_used
                    if len(current_used) < best_count:
                        best = current.copy()
                        best_count = len(current_used)
                        if best_count <= 13:
                            break
                break
            else:
                current[e] = old_c
                break  # try next edge
    
        if best_count <= 13:
            break
    
    # Final check
    final = best
    for ki in range(len(k4_edges)):
        if not check_k4(ki, final):
            return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return [int(final[edge_pos[(i, j)]]) for i in range(16) for j in range(i + 1, 16)]
