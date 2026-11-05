"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using precomputed C6 incidence."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin for safety
    
    # Known valid coloring for n=9: circular distance coloring
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Precompute all 6-cycles for each n
    def get_all_c6s(n):
        """Get all unique 6-cycles in K_n. Each C6 is a tuple of 6 vertices in cyclic order, 
        with the smallest vertex first and the second vertex smaller than the last (to avoid duplicates)."""
        c6s = []
        for v0 in range(n):
            for v1 in range(v0 + 1, n):
                for v2 in range(v0 + 1, n):
                    if v2 == v0 or v2 == v1:
                        continue
                    for v3 in range(v0 + 1, n):
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in range(v0 + 1, n):
                            if v4 in (v0, v1, v2, v3):
                                continue
                            for v5 in range(v0 + 1, n):
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                # Check canonical form: v0 is smallest, and v1 < v5 (to avoid reverse)
                                if v1 > v5:
                                    continue
                                # This is a valid unique C6: v0-v1-v2-v3-v4-v5-v0
                                c6s.append((v0, v1, v2, v3, v4, v5))
        return c6s
    
    # Precompute C6s for n up to 12 (beyond that, too many)
    c6_cache = {}
    for n in range(10, 13):
        c6_cache[n] = get_all_c6s(n)
    
    def edge_index(i, j, n):
        """Get the index in the colors array for edge (i,j) with i<j in lexicographic order."""
        # Number of edges before vertex i: i*(i-1)/2
        # Within vertex i's edges, j is the (j-i-1)th edge
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def count_c6s_for_color(colors, n, color, c6s):
        """Count the number of C6s where all 6 edges have the given color."""
        count = 0
        for c6 in c6s:
            v0, v1, v2, v3, v4, v5 = c6
            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            all_match = True
            for u, v in edges:
                if u > v:
                    u, v = v, u
                idx = edge_index(u, v, n)
                if colors[idx] != color:
                    all_match = False
                    break
            if all_match:
                count += 1
        return count
    
    def total_c6_count(colors, n):
        """Count total number of monochromatic C6s across all 4 color classes."""
        if n not in c6_cache:
            c6_cache[n] = get_all_c6s(n)
        c6s = c6_cache[n]
        total = 0
        for c in range(4):
            total += count_c6s_for_color(colors, n, c, c6s)
        return total
    
    # Try to find valid colorings for increasing n
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        num_edges = n * (n - 1) // 2
        found = False
        
        # For n >= 13, we don't have precomputed C6s, so use a different approach
        if n >= 13:
            # Use random search with quick validity check
            for attempt in range(100):
                if time.time() > deadline:
                    break
                
                # Random initial coloring
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                
                # Quick check: for each color, check if it contains a C6
                # Build adjacency for each color
                adj = [set() for _ in range(4)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = colors[idx]
                        adj[c].add((i, j))
                        idx += 1
                
                # Check each color for C6
                valid = True
                for c in range(4):
                    # Build adjacency list
                    adj_list = [set() for _ in range(n)]
                    for (u, v) in adj[c]:
                        adj_list[u].add(v)
                        adj_list[v].add(u)
                    
                    # Check for C6
                    for v0 in range(n):
                        if not valid:
                            break
                        for v1 in adj_list[v0]:
                            if v1 <= v0:
                                continue
                            for v2 in adj_list[v1]:
                                if v2 == v0:
                                    continue
                                for v3 in adj_list[v2]:
                                    if v3 in (v0, v1):
                                        continue
                                    for v4 in adj_list[v3]:
                                        if v4 in (v0, v1, v2, v3):
                                            continue
                                        for v5 in adj_list[v4]:
                                            if v5 in (v0, v1, v2, v3, v4):
                                                continue
                                            if v0 in adj_list[v5]:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
        else:
            # Use local search with precomputed C6s
            c6s = c6_cache[n]
            
            # Try multiple random restarts
            max_restarts = 200
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
                max_no_improve = 100
                while no_improvement < max_no_improve and time.time() < deadline:
                    # Pick a random edge to change
                    ei = random.randint(0, num_edges - 1)
                    old_color = colors[ei]
                    
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
