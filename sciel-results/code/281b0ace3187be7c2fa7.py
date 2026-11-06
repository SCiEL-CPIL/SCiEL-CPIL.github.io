import random
import time

def construct():
    # We want to find a coloring with K colors.
    # Try K = 15, then 16.
    # If not found, fallback to 17 (seed).
    
    n = 16
    # Precompute all edges and quadruples
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Precompute quadruples and their edges
    quads = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    quads.append([(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)])
    
    def check_validity(colors):
        for q in quads:
            # Get colors of the 6 edges
            c = [colors[edge_index[e]] for e in q]
            if len(set(c)) < 5:
                return False, q, c
        return True, None, None

    def local_search(K, time_limit=100):
        # Initialize with random colors 0..K-1
        colors = [random.randint(0, K-1) for _ in range(len(edges))]
        
        start_time = time.time()
        best_colors = colors.copy()
        
        while time.time() - start_time < time_limit:
            valid, bad_quad, bad_colors = check_validity(colors)
            if valid:
                return colors
            
            # Pick a random edge in the bad quad and change its color
            # To increase the number of distinct colors, we should pick a color 
            # that is not already in the quad, if possible.
            current_colors_in_quad = set(bad_colors)
            # Available colors in palette
            all_colors = set(range(K))
            new_colors = all_colors - current_colors_in_quad
            
            if new_colors:
                new_c = random.choice(list(new_colors))
            else:
                # If all colors in palette are used, pick a random one 
                # hoping to replace a duplicate
                new_c = random.randint(0, K-1)
            
            # Pick a random edge from the quad to change
            # Prefer edges that have a color appearing more than once in the quad
            # to maximize chance of reducing duplicates
            counts = {}
            for c in bad_colors:
                counts[c] = counts.get(c, 0) + 1
            
            # Find edges with duplicate colors
            dup_edges = []
            for idx, e in enumerate(bad_quad):
                if counts[bad_colors[idx]] > 1:
                    dup_edges.append(idx)
            
            if dup_edges:
                idx_to_change = random.choice(dup_edges)
            else:
                idx_to_change = random.randint(0, 5)
            
            edge_to_change = bad_quad[idx_to_change]
            colors[edge_index[edge_to_change]] = new_c
            
            # Update best if this is better? 
            # We just keep the current state as we are doing local search.
            
        return None

    # Try 15 colors
    # Use a few restarts
    for _ in range(5):
        res = local_search(15, time_limit=10)
        if res:
            return res

    # Try 16 colors
    for _ in range(5):
        res = local_search(16, time_limit=10)
        if res:
            return res
            
    # Fallback to seed (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
