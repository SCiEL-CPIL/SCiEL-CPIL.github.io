import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit
    
    # Start with the known n=9 solution
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Try to extend from n=9 up to n=19
    for target_n in range(n0 + 1, 20):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Multiple random restarts for each target_n
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize new colors: construct list in lexicographic order
            # For pairs (i,j) where j < n0, use the color from colors0.
            # Otherwise, use a random color.
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if j < n0:
                        # Calculate index of (i,j) in the lexicographic list for K_n0
                        idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                        colors.append(colors0[idx0])
                    else:
                        colors.append(random.randint(0, 3))
            
            # Run local search with simulated annealing
            result = _simulated_annealing(colors, target_n, deadline=deadline - 10)
            
            if _has_c6(result, target_n):
                continue
            
            # Found a valid coloring
            best_n = target_n
            best_colors = result
            found = True
            break
        
        if not found:
            # If we can't find a valid coloring for this n, try a few more approaches
            # before giving up on this n
            found = False
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Try a different initialization strategy: use the best coloring for n-1
                # and add one vertex with random colors
                if best_n >= target_n - 1:
                    colors = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if j < best_n and i < best_n:
                                # Use color from best_colors
                                idx = _pair_index(i, j, best_n)
                                colors.append(best_colors[idx])
                            else:
                                colors.append(random.randint(0, 3))
                
                result = _simulated_annealing(colors, target_n, deadline=deadline - 5)
                
                if not _has_c6(result, target_n):
                    best_n = target_n
                    best_colors = result
                    found = True
                    break
            
            if not found:
                break
    
    return {'n': best_n, 'colors': best_colors}

def _pair_index(i, j, n):
    """Calculate the index of pair (i,j) in the lexicographic list for K_n."""
    # Number of pairs (a, b) where a < i and b > a, plus pairs (i, b) where i < b < j
    count = 0
    for a in range(i):
        count += n - 1 - a
    count += j - i - 1
    return count

def _build_adjacency(colors, n):
    """Build adjacency matrices for each color class."""
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = True
            adj[c][j, i] = True
            idx += 1
    return adj

def _has_c6(colors, n):
    """Check if any color class contains a C6 using fast numpy-based counting."""
    adj = _build_adjacency(colors, n)
    
    for c in range(4):
        if _count_c6_in_adj(adj[c], n) > 0:
            return True
    return False

def _count_c6_in_adj(adj, n):
    """Count C6s in a single adjacency matrix using optimized DFS."""
    count = 0
    for s in range(n):
        # Only count cycles where s is the smallest vertex
        neighbors = [v for v in range(s + 1, n) if adj[s, v]]
        
        if len(neighbors) < 2:
            continue
        
        # DFS to find paths of length 5 from s back to s
        # Use iterative approach for speed
        def dfs_count(current, path_len, path_set, path):
            if path_len == 5:
                if adj[s, current]:
                    return 1
                return 0
            
            total = 0
            for nxt in range(s + 1, n):
                if nxt not in path_set and adj[current, nxt]:
                    path_set.add(nxt)
                    path.append(nxt)
                    total += dfs_count(nxt, path_len + 1, path_set, path)
                    path.pop()
                    path_set.remove(nxt)
            return total
        
        for start_neighbor in neighbors:
            path_set = {s, start_neighbor}
            path = [start_neighbor]
            count += dfs_count(start_neighbor, 1, path_set, path)
    
    return count

def _simulated_annealing(colors, n, deadline=None, max_steps=20000):
    """Perform simulated annealing to minimize C6 count."""
    colors = list(colors)
    best_colors = list(colors)
    best_c6_count = _count_total_c6s(colors, n)
    
    if best_c6_count == 0:
        return colors
    
    current_temp = 1.0
    cooling_rate = 0.9995
    step = 0
    
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        colors[idx] = new_color
        
        new_c6_count = _count_total_c6s(colors, n)
        delta = new_c6_count - best_c6_count
        
        if delta <= 0 or random.random() < np.exp(-delta / current_temp):
            best_c6_count = new_c6_count
            best_colors = list(colors)
            if best_c6_count == 0:
                return colors
        else:
            # Revert
            colors[idx] = old_color
        
        current_temp *= cooling_rate
        step += 1
    
    return best_colors

def _count_total_c6s(colors, n):
    """Count total number of C6s across all color classes."""
    adj = _build_adjacency(colors, n)
    total = 0
    for c in range(4):
        total += _count_c6_in_adj(adj[c], n)
    return total
