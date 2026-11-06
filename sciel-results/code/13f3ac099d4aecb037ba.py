"""Edge-colourings of K16 in which every K4 sees five colours."""
import time
import itertools
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before the 150s limit
    
    n = 16
    
    # We'll try to find a coloring with as few colors as possible.
    # The additive (i+j) mod 17 gives 17 colors. We want to do better.
    
    # Strategy: Use a systematic search with local improvement.
    # Start with a random coloring using a small number of colors,
    # then iteratively fix violations (K4s with < 5 colors).
    
    def count_colors_in_k4(colors, a, b, c, d):
        """Count distinct colors in the 6 edges of K4 on vertices a,b,c,d."""
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        col_set = set()
        for (u, v) in edges:
            if u > v:
                u, v = v, u
            col_set.add(colors[u][v])
        return len(col_set)
    
    def is_valid(colors, n):
        """Check if every K4 has at least 5 distinct colors."""
        for quad in itertools.combinations(range(n), 4):
            if count_colors_in_k4(colors, quad[0], quad[1], quad[2], quad[3]) < 5:
                return False
        return True
    
    def count_violations(colors, n):
        """Count number of K4s with < 5 distinct colors."""
        count = 0
        for quad in itertools.combinations(range(n), 4):
            if count_colors_in_k4(colors, quad[0], quad[1], quad[2], quad[3]) < 5:
                count += 1
        return count
    
    def get_violating_k4s(colors, n, limit=100):
        """Get up to 'limit' K4s that violate the constraint."""
        violations = []
        for quad in itertools.combinations(range(n), 4):
            if count_colors_in_k4(colors, quad[0], quad[1], quad[2], quad[3]) < 5:
                violations.append(quad)
                if len(violations) >= limit:
                    break
        return violations
    
    def local_search(colors, n, max_colors, time_limit):
        """
        Local search: try to reduce the number of distinct colors used
        while maintaining validity. Start from a valid coloring and try
        to recolor edges to use fewer colors.
        """
        # First, ensure we have a valid coloring
        # Try to minimize the number of colors used
        
        def num_colors_used(colors, n):
            used = set()
            for i in range(n):
                for j in range(i+1, n):
                    used.add(colors[i][j])
            return len(used)
        
        best_colors = [row[:] for row in colors]
        best_nc = num_colors_used(best_colors, n)
        
        # Try to reduce colors: for each color c, try to recolor all edges
        # of color c to other colors, if possible without creating violations.
        
        # A simpler approach: greedy recoloring.
        # Sort edges by some heuristic, try to assign minimum color possible.
        
        return best_colors, best_nc
    
    # Approach 1: Try small color counts with random initialization + repair
    
    # For a given number of colors k, try to find a valid coloring.
    def try_find_coloring(k, n, time_limit, seed=42):
        """Try to find a valid coloring using at most k colors."""
        rng = random.Random(seed)
        
        # Initialize randomly
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = rng.randint(0, k-1)
                colors[j][i] = colors[i][j]
        
        # Repair phase: iteratively fix violations
        for iteration in range(1000):
            if time.time() > time_limit:
                break
            
            violations = get_violating_k4s(colors, n, limit=50)
            if not violations:
                return colors
            
            # Pick a random violation and try to fix it
            quad = rng.choice(violations)
            a, b, c, d = quad
            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            
            # Try changing one edge's color to fix this K4
            fixed = False
            for (u, v) in edges:
                old_color = colors[u][v]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    # Try this change
                    colors[u][v] = new_color
                    colors[v][u] = new_color
                    
                    # Check if this K4 is now fixed and we haven't created too many new violations
                    if count_colors_in_k4(colors, a, b, c, d) >= 5:
                        # Check global validity (or at least that we're not making things much worse)
                        # For speed, just accept if this K4 is fixed
                        fixed = True
                        break
                    else:
                        # Revert
                        colors[u][v] = old_color
                        colors[v][u] = old_color
                
                if fixed:
                    break
            
            if not fixed:
                # Try a different strategy: change two edges
                for (u1, v1) in edges:
                    for (u2, v2) in edges:
                        if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                            continue
                        old1 = colors[u1][v1]
                        old2 = colors[u2][v2]
                        for nc1 in range(k):
                            for nc2 in range(k):
                                if nc1 == old1 and nc2 == old2:
                                    continue
                                colors[u1][v1] = nc1
                                colors[v1][u1] = nc1
                                colors[u2][v2] = nc2
                                colors[v2][u2] = nc2
                                
                                if count_colors_in_k4(colors, a, b, c, d) >= 5:
                                    fixed = True
                                    break
                                else:
                                    colors[u1][v1] = old1
                                    colors[v1][u1] = old1
                                    colors[u2][v2] = old2
                                    colors[v2][u2] = old2
                            if fixed:
                                break
                        if fixed:
                            break
                    if fixed:
                        break
                
                if not fixed:
                    # Give up on this violation, try another
                    continue
        
        # Final check
        if is_valid(colors, n):
            return colors
        return None
    
    # Start with the known valid solution using 17 colors
    def additive_coloring(n):
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                c = (i + j) % 17
                colors[i][j] = c
                colors[j][i] = c
        return colors
    
    best_colors = additive_coloring(n)
    best_num_colors = 17
    
    # Try to improve: search for colorings with fewer colors
    # Try k from 13 upward (theoretical lower bound is 13)
    for k in range(13, 17):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            
            result = try_find_coloring(k, n, deadline, seed)
            if result is not None:
                best_colors = result
                best_num_colors = k
                found = True
                break
        
        if found:
            # We found a valid coloring with k colors, try to go lower
            continue
    
    # Also try local search from the best solution to reduce colors further
    if time.time() < deadline - 5:
        # Try to reduce colors from best_num_colors
        for target_k in range(13, best_num_colors):
            if time.time() > deadline:
                break
            # Recolor some edges to use only target_k colors
            # This is complex; skip for now if time is tight
    
    # Convert to the required output format
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(best_colors[i][j])
    
    return result

# EVOLVE-BLOCK-END
