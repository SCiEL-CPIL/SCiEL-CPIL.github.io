"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using C6-minimizing local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def edge_index(i, j, n):
    """Get the index in the colors array for edge (i,j) with i<j in lexicographic order."""
    # Number of edges before vertex i: i*(i-1)/2
    # Within vertex i's edges, j is the (j-i-1)th edge
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_adj_per_color(colors, n):
    """Build adjacency sets for each color class."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    return adj

def count_c6s(adj_set, n):
    """Count the number of 6-cycles in a graph given as a set of edges."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for (u, v) in adj_set:
        adj[u].add(v)
        adj[v].add(u)
    
    count = 0
    # For each 6-cycle v0-v1-v2-v3-v4-v5-v0, count it once
    # Fix v0 < v1 to avoid counting the same cycle multiple times
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1):
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2, v3):
                            continue
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if v0 in adj[v5]:
                                count += 1
    return count

def total_c6_count(colors, n):
    """Count total number of monochromatic C6s across all 4 color classes."""
    adj_sets = build_adj_per_color(colors, n)
    total = 0
    for c in range(4):
        total += count_c6s(adj_sets[c], n)
    return total

def construct():
    start_time = time.time()
    deadline = 135  # Leave 15s margin for safety
    
    # Known valid coloring for n=9: circular distance coloring
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find valid colorings for increasing n
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        num_edges = n * (n - 1) // 2
        found = False
        
        # Try multiple random restarts with local search
        max_restarts = 500
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Local search: minimize total C6 count
            current_count = total_c6_count(colors, n)
            
            # If already valid, we're done
            if current_count == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            # Local search iterations
            no_improvement = 0
            max_no_improve = 200
            while no_improvement < max_no_improve and time.time() < deadline:
                # Pick a random edge to change
                ei = random.randint(0, num_edges - 1)
                old_color = colors[ei]
                
                # Find which vertices this edge connects
                # Reverse the edge_index mapping
                # For edge index ei, find i and j
                # i is such that i*(n-1) - i*(i-1)//2 <= ei < (i+1)*(n-1) - (i+1)*i//2
                i = 0
                while True:
                    start = i * (n - 1) - i * (i - 1) // 2
                    end = (i + 1) * (n - 1) - (i + 1) * i // 2
                    if start <= ei < end:
                        break
                    i += 1
                j = i + 1 + (ei - start)
                
                # Try changing to each other color
                best_new_color = old_color
                best_new_count = current_count
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    colors[ei] = new_color
                    new_count = total_c6_count(colors, n)
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_new_color = new_color
                    colors[ei] = old_color
                
                if best_new_color != old_color:
                    colors[ei] = best_new_color
                    current_count = best_new_count
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                if current_count == 0:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
            
            if found:
                break
        
        if found:
            continue  # Try larger n
        else:
            break  # Can't find valid for this n, stop
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
