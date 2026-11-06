"""Edge-colourings of K16 in which every K4 sees five colours."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    """Return 120 integer colours for edges of K16 (lex order i<j).
    Every K4 must span at least 5 distinct colours. Minimise max colour used."""
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill

    n = 16
    # Precompute edge indices in lex order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120

    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    edge_idx[(j, i)] = edge_idx[(i, j)] if (i, j) in edge_idx else None
    # Actually let's be careful:
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    # Precompute all K4s and their edges
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for quad in k4s:
        a, b, c, d = quad
        e = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4_edges.append([edge_idx[x] for x in e])
    num_k4 = len(k4s)  # 1820

    # Initial solution: (i+j) mod 17
    sol = [(i + j) % 17 for i, j in edges]
    max_color = 16

    def count_colors(arr):
        return len(set(arr))

    def is_valid_solution(s):
        for ke in k4_edges:
            if len(set(s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]])) < 5:
                return False
        return True

    # Verify initial
    # is_valid_solution(sol) should be True

    # Local search: try to reduce max_color
    # Strategy: iteratively try to re-color edges that use the highest color
    # with a lower color that maintains validity
    
    def try_reduce_max_color(s, max_c, t_limit):
        """Try to reduce max color by 1 by recoloring edges with max_c."""
        # Find all edges with color max_c
        edges_with_max = [idx for idx, c in enumerate(s) if c == max_c]
        if not edges_with_max:
            return s, max_c
        
        # Try to recolor each edge with max_c to a lower color
        # Use random order
        random.shuffle(edges_with_max)
        
        for idx in edges_with_max:
            if time.time() > t_limit:
                break
            # Try colors from 0 to max_c-1
            # Random order
            candidates = list(range(max_c))
            random.shuffle(candidates)
            
            for new_c in candidates:
                if new_c >= max_c:
                    continue
                # Check if changing this edge maintains validity
                # Only need to check K4s containing this edge
                old_c = s[idx]
                s[idx] = new_c
                
                valid = True
                for quad_idx, ke in enumerate(k4_edges):
                    if idx not in ke:
                        continue
                    if len(set(s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]])) < 5:
                        valid = False
                        break
                
                if valid:
                    # Success
                    return s, max_c - 1 if all(s[e2] <= max_c - 1 for e2 in range(m)) else max_c
                else:
                    s[idx] = old_c
        
        return s, max_c

    # More efficient: track which K4s are "tight" (have exactly 5 colors)
    # and optimize accordingly
    
    # Let's do a simpler local search:
    # 1. Start with (i+j)%17
    # 2. Try to reduce number of colors by swapping colors
    
    # Actually, let's try a different approach:
    # Use the fact that we can try colorings based on (i+j) mod p for different p
    # or use a genetic algorithm / simulated annealing
    
    # Let's try: minimize max color using local search
    # State: color assignment
    # Move: recolor one edge to a different color
    # Objective: minimize max color, then minimize number of colors used
    
    # For speed, let's just try a few specific constructions and a short local search
    
    # Construction 1: (i+j) mod 17 -> 17 colors
    best_sol = sol[:]
    best_max = 16  # colors 0..16
    
    # Try to improve with local search
    def local_search(s, max_restarts=5, time_limit=100.0):
        s = s[:]
        current_max = max(s)
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Hill climbing: try to reduce max color
            improved = True
            while improved:
                if time.time() > time_limit:
                    break
                improved = False
                current_max = max(s)
                
                # Find edges with current_max color
                max_edges = [idx for idx in range(m) if s[idx] == current_max]
                random.shuffle(max_edges)
                
                for idx in max_edges:
                    if time.time() > time_limit:
                        break
                    old_c = s[idx]
                    # Try all colors 0..current_max-1
                    for new_c in range(current_max):
                        s[idx] = new_c
                        # Check validity - only K4s containing this edge
                        valid = True
                        for ke in k4_edges:
                            if idx not in ke:
                                continue
                            if len(set(s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]])) < 5:
                                valid = False
                                break
                        if valid:
                            improved = True
                            break
                        else:
                            s[idx] = old_c
                    
                    if improved:
                        break
            
            # After hill climbing, try random perturbations
            if time.time() > time_limit:
                break
            
            for _ in range(50):
                if time.time() > time_limit:
                    break
                idx = random.randint(0, m-1)
                old_c = s[idx]
                new_c = random.randint(0, max(s))
                s[idx] = new_c
                
                valid = True
                for ke in k4_edges:
                    if idx not in ke:
                        continue
                    if len(set(s[ke[0]], s[ke[1]], s[ke[2]], s[ke[3]], s[ke[4]], s[ke[5]])) < 5:
                        valid = False
                        break
                
                if not valid:
                    s[idx] = old_c
        
        return s, max(s)
    
    if time.time() < deadline:
        sol2, max2 = local_search(best_sol, max_restarts=10, time_limit=100.0)
        if max2 < best_max:
            best_sol = sol2
            best_max = max2
    
    # Count actual colors used
    colors_used = len(set(best_sol))
    
    return best_sol

# EVOLVE-BLOCK-END
