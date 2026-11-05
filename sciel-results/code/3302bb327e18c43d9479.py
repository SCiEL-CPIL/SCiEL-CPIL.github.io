import time
import random
import sys

def count_c4s(colors, n):
    """
    Count the number of monochromatic C4s.
    colors: list of length n*(n-1)//2
    """
    # Map (i, j) to index
    def idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
    # Actually, simpler index for n<=29:
    # i from 0 to n-2, j from i+1 to n-1
    # idx = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    
    # For speed, we can just iterate.
    # A C4 is (a,b,c,d) with edges (a,b), (b,c), (c,d), (d,a) all same color.
    # We can count by iterating pairs of vertices (a,c) and checking common neighbors.
    # If a and c have 2 common neighbors b, d with the same color, that's a C4.
    # Number of C4s = sum over all colors c, sum over all pairs (a,c) of (common_neighbors_c(a,c) choose 2).
    
    total_c4s = 0
    # Precompute adjacency lists for each color?
    # Or just iterate.
    # For n=13, edges=78. Pairs of vertices = 78.
    # For each pair (a,c), count common neighbors for each color.
    
    # Let's build color-specific adjacency sets for faster lookup
    # adj[color][vertex] = set of neighbors
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    # Fill adj
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    for c in range(5):
        for a in range(n):
            for b in range(a+1, n):
                # Count common neighbors of a and b in color c
                common = adj[c][a].intersection(adj[c][b])
                k = len(common)
                if k >= 2:
                    total_c4s += k * (k - 1) // 2
                    
    return total_c4s

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find valid colorings for n = 12, 13, 14
    for n in range(12, 16):
        if time.time() > deadline:
            break
            
        # Initialize
        num_edges = n * (n - 1) // 2
        # Heuristic initialization: random
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Local Search
        # To speed up, we can use a simpler cost: number of C4s
        # But recalculating full cost is O(n^4). For n=14, 14^4 ~ 38k. Fast.
        
        # Min-Conflicts
        # We need to track which edges are in C4s to guide search?
        # For small n, random restarts with hill climbing might work.
        
        found = False
        restarts = 0
        max_restarts = 50
        
        while restarts < max_restarts and time.time() < deadline:
            restarts += 1
            # Randomize if stuck or at start
            if restarts == 1:
                 pass # Keep initial
            else:
                 # Perturb
                 for _ in range(10):
                     idx = random.randint(0, num_edges - 1)
                     colors[idx] = random.randint(0, 4)
            
            current_cost = count_c4s(colors, n)
            if current_cost == 0:
                found = True
                break
                
            # Hill Climbing
            steps = 0
            max_steps = 200
            while steps < max_steps and time.time() < deadline:
                steps += 1
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                old_color = colors[idx]
                # Try all 5 colors
                best_new_color = old_color
                best_cost = current_cost
                
                # Calculate cost change?
                # Recalculating full cost is O(n^4).
                # 200 steps * 5 colors * O(n^4) might be slow for n=15.
                # n=15: 15^4 = 50k. 200*5*50k = 50M ops. Too slow in Python.
                
                # Optimization:
                # Only recalculate if we pick an edge that is part of a C4.
                # Or use a faster cost estimator.
                # Given the small n, let's just limit steps or use a faster check.
                
                # Let's try a different approach:
                # If n is small, we can just do full recalculation but fewer steps.
                
                new_color = random.randint(0, 4)
                colors[idx] = new_color
                new_cost = count_c4s(colors, n)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        found = True
                        break
                else:
                    # Revert or accept with prob?
                    # Simple hill climbing: revert
                    colors[idx] = old_color
                    
        if found:
            best_n = n
            best_colors = colors.copy()
            # Try next n
            continue
        else:
            # If not found, maybe try next n with more restarts?
            # Or just stop if we fail on 12?
            # Let's try a few restarts for each n.
            pass

    return {'n': best_n, 'colors': best_colors}
