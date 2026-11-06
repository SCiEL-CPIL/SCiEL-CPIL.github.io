"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a 5-colouring of K_n with no monochromatic C4.
    Uses a local search / simulated annealing approach to maximize n.
    """
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    def is_valid_coloring(n, colors):
        """Check if the coloring has no monochromatic C4."""
        # For each color, check that no two vertices have two common neighbors
        # Build adjacency lists for each color
        adj = [dict() for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if j not in adj[c]:
                    adj[c][j] = set()
                if i not in adj[c]:
                    adj[c][i] = set()
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        # Check for C4: for each color, for each pair of vertices, 
        # they should have at most 1 common neighbor
        for c in range(5):
            # For efficiency, check pairs of vertices
            # A C4 exists if there are vertices u, v, w, x such that
            # u-w, w-v, v-x, x-u are all color c
            # Equivalently, for any two vertices u, v, they have at most 1 common neighbor in color c
            
            # Build neighbor sets
            neigh = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        neigh[i].add(j)
                        neigh[j].add(i)
                    idx += 1
            
            # Check all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    if len(neigh[i] & neigh[j]) >= 2:
                        return False
        return True
    
    def count_violations(n, colors):
        """Count the number of monochromatic C4 violations."""
        # A violation is a pair of vertices with >= 2 common neighbors in some color
        # We count the total number of "bad" pairs
        adj = [dict() for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if j not in adj[c]:
                    adj[c][j] = set()
                if i not in adj[c]:
                    adj[c][i] = set()
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        violations = 0
        for c in range(5):
            neigh = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        neigh[i].add(j)
                        neigh[j].add(i)
                    idx += 1
            
            for i in range(n):
                for j in range(i + 1, n):
                    common = len(neigh[i] & neigh[j])
                    if common >= 2:
                        violations += common - 1
        return violations
    
    def try_greedy(n):
        """Try to build a valid coloring greedily."""
        colors = [0] * (n * (n - 1) // 2)
        
        # For each edge, assign a color that minimizes violations
        # Use a simple heuristic: for each edge, try all 5 colors and pick the one
        # that causes the fewest new C4 violations
        
        for idx in range(len(colors)):
            # Find i, j from idx
            # idx = i*n - i*(i+1)//2 + (j-i-1) ... this is complex
            # Let's just iterate properly
            pass
        
        # Simpler: just use a random initialization and local search
        return None
    
    def local_search(n, colors, max_iters=10000):
        """Local search to minimize violations."""
        best_colors = list(colors)
        best_violations = count_violations(n, best_colors)
        
        if best_violations == 0:
            return best_colors, 0
        
        current_colors = list(best_colors)
        current_violations = best_violations
        
        for iteration in range(max_iters):
            # Pick a random edge to change
            idx = random.randint(0, len(current_colors) - 1)
            old_color = current_colors[idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            current_colors[idx] = new_color
            
            # Recalculate violations (expensive but simple)
            # For efficiency, we could update incrementally, but let's keep it simple
            new_violations = count_violations(n, current_colors)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = list(current_colors)
                    if best_violations == 0:
                        return best_colors, 0
            else:
                # Revert
                current_colors[idx] = old_color
        
        return best_colors, best_violations
    
    # Start with small n and try to grow
    best_n = 11
    best_colors = [0] * (11 * 10 // 2)
    
    # Initialize with the known 11-vertex construction
    for i in range(11):
        for j in range(i + 1, 11):
            d = min(j - i, 11 - j + i)
            best_colors[i * 10 - i * (i + 1) // 2 + (j - i - 1)] = d - 1
    
    # Try to extend to larger n
    for target_n in range(12, 27):
        if time.time() > deadline:
            break
        
        # Try to build a coloring for target_n
        # Start from the best coloring for target_n - 1 and extend
        # This is complex, so let's try random initialization + local search
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local search
            colors, violations = local_search(target_n, colors, max_iters=5000)
            
            if violations == 0:
                best_n = target_n
                best_colors = colors
                break
    
    # Final check
    if time.time() > deadline - 5:
        # Return what we have
        pass
    
    # Verify the final result
    if best_n > 1:
        final_violations = count_violations(best_n, best_colors)
        if final_violations > 0:
            # Fallback to known valid 11-vertex construction
            best_n = 11
            best_colors = [0] * (11 * 10 // 2)
            for i in range(11):
                for j in range(i + 1, 11):
                    d = min(j - i, 11 - j + i)
                    best_colors[i * 10 - i * (i + 1) // 2 + (j - i - 1)] = d - 1
    
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
