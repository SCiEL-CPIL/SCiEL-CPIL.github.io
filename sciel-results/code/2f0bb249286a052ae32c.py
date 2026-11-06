"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _edge_index(n, i, j):
    """Index for edge (i,j) with i<j in lexicographic order."""
    return i * (2 * n - i - 3) // 2 + (j - i - 1)

def _has_c6_color(colors, n, color):
    """Check if edges of given color contain a 6-cycle."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _edge_index(n, i, j)
            if colors[idx] == color:
                adj[i].add(j)
                adj[j].add(i)
    
    # Try all ordered pairs of non-adjacent vertices as start
    # We need a cycle v0-v1-v2-v3-v4-v5-v0
    for v0 in range(n):
        for v1 in sorted(adj[v0]):
            for v2 in sorted(adj[v1]):
                if v2 == v0:
                    continue
                for v3 in sorted(adj[v2]):
                    if v3 in (v0, v1):
                        continue
                    for v4 in sorted(adj[v3]):
                        if v4 in (v0, v1, v2):
                            continue
                        for v5 in sorted(adj[v4]):
                            if v5 in (v0, v1, v2, v3):
                                continue
                            if v5 != v0 and v0 in adj[v5]:
                                return True
    return False

def _has_c6(colors, n):
    """Check if any color class contains a 6-cycle."""
    for c in range(4):
        if _has_c6_color(colors, n, c):
            return True
    return False

def _count_c6_color(colors, n, color):
    """Count 6-cycles in edges of given color."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _edge_index(n, i, j)
            if colors[idx] == color:
                adj[i].add(j)
                adj[j].add(i)
    
    count = 0
    for v0 in range(n):
        for v1 in sorted(adj[v0]):
            for v2 in sorted(adj[v1]):
                if v2 == v0:
                    continue
                for v3 in sorted(adj[v2]):
                    if v3 in (v0, v1):
                        continue
                    for v4 in sorted(adj[v3]):
                        if v4 in (v0, v1, v2):
                            continue
                        for v5 in sorted(adj[v4]):
                            if v5 in (v0, v1, v2, v3):
                                continue
                            if v5 != v0 and v0 in adj[v5]:
                                count += 1
    return count // 2  # Each cycle counted multiple times

def _count_total_c6(colors, n):
    """Total 6-cycles across all colors."""
    total = 0
    for c in range(4):
        total += _count_c6_color(colors, n, c)
    return total

def _spectral_embed(n, k=2, iterations=50):
    """Get a 2D embedding of K_n using random regular graph adjacency."""
    # Create a random 3-regular graph as a proxy for structure
    adj = [[0]*n for _ in range(n)]
    random.seed(42)
    # Simple random graph with p=0.5
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
    # Compute eigenvectors of the adjacency matrix
    # Use numpy for SVD
    import numpy as np
    A = np.array(adj, dtype=float)
    # Center
    A_centered = A - A.mean(axis=1, keepdims=True)
    _, V, W = np.linalg.svd(A_centered)
    # Use first 2 right singular vectors
    X = W.T[:2, :]  # shape (2, n)
    return X

def _initial_coloring_spectral(n):
    """Color edges based on spectral embedding quadrants."""
    X = _spectral_embed(n)
    # Normalize to [0,1]
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range
    
    # Determine quadrant for each vertex
    # Use median splits
    med_x = np.median(X_norm[0])
    med_y = np.median(X_norm[1])
    
    quad = []
    for i in range(n):
        qx = 0 if X_norm[0][i] < med_x else 1
        qy = 0 if X_norm[1][i] < med_y else 1
        quad.append(qx * 2 + qy)
    
    # Color edges by quadrant difference
    colors = [0] * (n * (n-1) // 2)
    for i in range(n):
        for j in range(i+1, n):
            idx = _edge_index(n, i, j)
            diff = abs(quad[i] - quad[j])
            colors[idx] = diff % 4
    return colors

def _repair_local_search(colors, n, max_iters=20000, deadline=100):
    """Local search to eliminate C6 cycles."""
    deadline_time = time.time() + deadline
    best_colors = colors[:]
    best_c6 = _count_total_c6(colors, n)
    
    for it in range(max_iters):
        if time.time() > deadline_time:
            break
        if best_c6 == 0:
            break
        
        # Find a random C6 and try to fix it
        # Pick a random edge and see if it's in a C6
        fixed_any = False
        attempts = 0
        while attempts < 200 and not fixed_any:
            attempts += 1
            # Pick random edge
            ei = random.randint(0, n*(n-1)//2 - 1)
            # Find which color
            c = colors[ei]
            # Try changing to a different color
            new_c = random.choice([x for x in range(4) if x != c])
            
            # Test if this change reduces C6 count
            old_val = colors[ei]
            colors[ei] = new_c
            new_c6 = _count_total_c6(colors, n)
            if new_c6 < best_c6:
                best_c6 = new_c6
                best_colors = colors[:]
                fixed_any = True
            else:
                colors[ei] = old_val
        
        if not fixed_any:
            # Try a more targeted approach: find an actual C6 and break it
            # Sample random starting points
            found_cycle = False
            for _ in range(50):
                v0 = random.randint(0, n-1)
                # Try to find a 6-cycle through v0
                # Get neighbors by color
                for c in range(4):
                    neighbors = []
                    for j in range(n):
                        if j == v0:
                            continue
                        idx = _edge_index(n, min(v0,j), max(v0,j))
                        if colors[idx] == c:
                            neighbors.append(j)
                    if len(neighbors) < 2:
                        continue
                    # Try pairs
                    for a_idx in range(len(neighbors)):
                        for b_idx in range(a_idx+1, len(neighbors)):
                            v1 = neighbors[a_idx]
                            v2 = neighbors[b_idx]
                            # Need v1-v2 edge
                            idx12 = _edge_index(n, min(v1,v2), max(v1,v2))
                            if colors[idx12] != c:
                                continue
                            # Now need v3 adjacent to v2 with color c
                            for v3 in range(n):
                                if v3 in (v0, v1, v2):
                                    continue
                                idx23 = _edge_index(n, min(v2,v3), max(v2,v3))
                                if colors[idx23] != c:
                                    continue
                                # Need v4 adjacent to v3
                                for v4 in range(n):
                                    if v4 in (v0, v1, v2, v3):
                                        continue
                                    idx34 = _edge_index(n, min(v3,v4), max(v3,v4))
                                    if colors[idx34] != c:
                                        continue
                                    # Need v5 adjacent to v4 and v0
                                    idx45 = _edge_index(n, min(v4,v0+1 if False else v4, v0 if v4>v0 else v0), max(v4, v0)) if v4 != v0 else -1
                                    # Simpler check
                                    for v5 in range(n):
                                        if v5 in (v0, v1, v2, v3, v4):
                                            continue
                                        idx45 = _edge_index(n, min(v4,v5), max(v4,v5))
                                        if colors[idx45] != c:
                                            continue
                                        idx50 = _edge_index(n, min(v5,v0), max(v5,v0))
                                        if colors[idx50] != c:
                                            continue
                                        # Found a C6: v0-v1-v2-v3-v4-v5-v0
                                        # Try to break it by recoloring one edge
                                        cycle_edges = [
                                            _edge_index(n, min(v0,v1), max(v0,v1)),
                                            _edge_index(n, min(v1,v2), max(v1,v2)),
                                            _edge_index(n, min(v2,v3), max(v2,v3)),
                                            _edge_index(n, min(v3,v4), max(v3,v4)),
                                            _edge_index(n, min(v4,v5), max(v4,v5)),
                                            _edge_index(n, min(v5,v0), max(v5,v0)),
                                        ]
                                        # Try recoloring each edge to a different color
                                        best_new_c6 = best_c6
                                        best_change = None
                                        for ce in cycle_edges:
                                            old_col = colors[ce]
                                            for new_col in range(4):
                                                if new_col == old_col:
                                                    continue
                                                colors[ce] = new_col
                                                test_c6 = _count_total_c6(colors, n)
                                                if test_c6 < best_new_c6:
                                                    best_new_c6 = test_c6
                                                    best_change = (ce, new_col)
                                                colors[ce] = old_col
                                        if best_change is not None:
                                            ce, new_col = best_change
                                            colors[ce] = new_col
                                            best_c6 = best_new_c6
                                            best_colors = colors[:]
                                            fixed_any = True
                                            found_cycle = True
                                            break
                                    if found_cycle:
                                        break
                                if found_cycle:
                                    break
                            if found_cycle:
                                break
                        if found_cycle:
                            break
                    if found_cycle:
                        break
                if found_cycle:
                    break
            
    return best_colors, best_c6

def _try_greedy_construct(n, deadline=100):
    """Try to construct a valid coloring for n vertices using greedy + local search."""
    deadline_time = time.time() + deadline
    if time.time() > deadline_time:
        return None, -1
    
    # Start with random coloring
    best_colors = None
    best_c6 = float('inf')
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline_time - 5:
            break
        random.seed(attempt * 12345 + 678)
        colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        
        # Local search
        repaired, c6_count = _repair_local_search(colors, n, max_iters=5000, deadline=max(0, deadline_time - time.time()))
        
        if c6_count < best_c6:
            best_c6 = c6_count
            best_colors = repaired
        
        if best_c6 == 0:
            return best_colors, 0
    
    return best_colors, best_c6

def construct():
    """Construct a 4-coloring of K_n with no monochromatic C6."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_n = 9
    best_result = {'n': best_n, 'colors': [min(j - i, best_n - j + i) - 1 for i in range(best_n) for j in range(i + 1, best_n)]}
    
    # Try to find better solutions
    for n in range(17, 9, -1):  # Try 17, 16, 15, ..., 10
        if time.time() > deadline - 30:
            break
        
        colors, c6_count = _try_greedy_construct(n, deadline=max(0, deadline - time.time() - 5))
        
        if colors is not None and c6_count == 0:
            # Verify
            if not _has_c6(colors, n):
                best_n = n
                best_result = {'n': n, 'colors': colors}
                break  # Found a valid coloring for this n
    
    # If we didn't find anything better, fall back to known good solution
    if best_n <= 9:
        n = 9
        best_result = {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    return best_result

# EVOLVE-BLOCK-END
