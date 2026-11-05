"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16,
    in lexicographic order.
    
    The base construction uses (i + j) % 17, which uses 17 colours.
    We attempt to reduce the number of colours via local search.
    
    Every set of four vertices must span edges of at least five distinct colours.
    """
    n = 16
    # Initialize with the known valid 17-colour construction
    colors = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            colors[i][j] = (i + j) % 17
            colors[j][i] = colors[i][j]
    
    # Precompute all K4s (sets of 4 vertices)
    # There are C(16,4) = 1820 such sets.
    # For each K4, we store the 6 edges.
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(edges)
    
    # Function to count violations
    # A violation is a K4 with fewer than 5 distinct colours.
    def count_violations():
        violations = 0
        for edges in k4s:
            c_set = set()
            for (u, v) in edges:
                c_set.add(colors[u][v])
            if len(c_set) < 5:
                violations += 1
        return violations
    
    # Function to get the list of violating K4s
    def get_violations():
        viol = []
        for edges in k4s:
            c_set = set()
            for (u, v) in edges:
                c_set.add(colors[u][v])
            if len(c_set) < 5:
                viol.append(edges)
        return viol
    
    # Local search: try to change a single edge colour to fix violations
    # We prioritize edges that are in many violations.
    
    # We want to minimize the number of colours used.
    # Strategy:
    # 1. Start with valid 17-colour solution.
    # 2. Try to reduce colour count by re-colouring edges to existing colours or lower-index colours.
    # 3. Use local search to maintain validity.
    
    # Let's define a metric:
    # Primary: Number of colours used.
    # Secondary: Number of violations (should be 0).
    
    def get_num_colours():
        used = set()
        for i in range(n):
            for j in range(i+1, n):
                used.add(colors[i][j])
        return len(used)
    
    # Initial state
    current_violations = count_violations()
    current_colours = get_num_colours()
    
    best_violations = current_violations
    best_colours = current_colours
    best_colors = [row[:] for row in colors]
    
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for 120s limit
    
    # Local search loop
    # We will attempt to change colours to reduce the number of distinct colours used.
    # We can map colours to a canonical set {0, 1, ..., k-1}.
    
    # To reduce colour count, we try to recolor an edge to a colour that is already used
    # by another edge, specifically one with a lower index if we are trying to compact the range.
    
    # However, simply reducing the range isn't enough if we skip indices.
    # Let's keep a map of used colours.
    
    # A better approach for minimization:
    # Randomly pick an edge.
    # Try changing its colour to a random colour from the set of currently used colours.
    # Or try changing it to a new colour if it reduces violations? No, we want fewer colours.
    # So we only change to existing colours.
    
    # We also need to ensure we don't introduce violations.
    
    # Let's use a simulated annealing-like approach or simple hill climbing on (violations, colours).
    # Objective: Minimize violations, then minimize colours.
    
    # Since we start valid, we must stay valid to be useful.
    # So we only accept moves that keep violations at 0.
    # Among valid moves, we accept if it reduces the number of colours used.
    
    # To reduce colours, we can try to "merge" two colour classes.
    # Pick two colours c1, c2. Try to recolor all edges of c2 to c1.
    # Check if valid. If valid, accept and merge.
    
    # This is a more structured search.
    
    # Let's try a simpler local search first:
    # While time < timeout:
    #   Pick a random edge (u,v).
    #   Pick a random target colour c from the set of used colours.
    #   If c == colors[u][v], continue.
    #   Check if changing (u,v) to c keeps validity.
    #   If valid, and it reduces the number of used colours (i.e., c was used elsewhere and colors[u][v] is not used elsewhere), accept.
    #   If valid and colour count same, accept with some probability? No, stick to strict improvement for colour count to avoid cycles.
    #   Actually, we can accept neutral moves to escape local minima, but we must be careful.
    
    # Let's implement the "merge colour classes" idea.
    # It's efficient because changing all edges of one colour to another is a bulk operation.
    # But checking validity for a bulk change is expensive if we re-check all K4s.
    # However, only K4s containing at least one edge of the changed colour are affected.
    
    # Let's stick to single edge changes for simplicity and speed of validation.
    
    # Optimization: Precompute which K4s contain which edges?
    # Edge (u,v) is in K4s that contain {u,v} and two other vertices.
    # C(14, 2) = 91 K4s per edge.
    # Total checks per move: 91 * (cost to check one K4).
    # Cost to check one K4: 6 edges, set creation. Very fast.
    
    # So single edge move is cheap enough.
    
    # Algorithm:
    # 1. List of used colours.
    # 2. Loop:
    #    a. Find an edge whose colour is unique (count 1). If we change it to a non-unique colour, we reduce colour count by 1.
    #    b. If no unique colour, we are stuck in a local minimum for this strategy.
    #    c. Try to change a unique-colour edge to a non-unique colour.
    #    d. If valid, accept.
    #    e. If not valid, try another unique-colour edge or another target colour.
    #    f. If no improvement found in a while, try random valid moves (neutral) to escape?
    #       Or restart?
    
    # Let's implement this.
    
    def count_colour_occurrences():
        counts = {}
        for i in range(n):
            for j in range(i+1, n):
                c = colors[i][j]
                counts[c] = counts.get(c, 0) + 1
        return counts
    
    # Main search loop
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() - start_time < timeout:
        counts = count_colour_occurrences()
        unique_colours = [c for c, cnt in counts.items() if cnt == 1]
        
        if not unique_colours:
            # No unique colours. All colours used at least twice.
            # We might be at a local minimum for "merge unique" strategy.
            # Try random valid moves to perturb.
            # Pick random edge, random target colour from used set.
            # If valid, accept (neutral move).
            
            # To avoid infinite loops, we limit neutral moves.
            if no_improvement_count > max_no_improvement:
                break
            
            # Try a few random neutral moves
            found_move = False
            for _ in range(10):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                old_c = colors[u][v]
                used = list(counts.keys())
                if len(used) <= 1: break
                new_c = random.choice(used)
                if new_c == old_c: continue
                
                # Check validity
                valid = True
                # Check all K4s containing edge (u,v)
                # We need to find K4s containing u and v.
                # This requires iterating all K4s or having a precomputed index.
                # Let's precompute edge_to_k4s.
                
                # For now, let's just check all K4s? 1820 checks.
                # 1820 * 10 (per iter) * 10 (random tries) = 182k checks per "step".
                # Might be slow in Python.
                
                # Let's precompute edge_to_k4s.
                # edge_to_k4s[(u,v)] = list of indices of K4s containing (u,v).
                
                # We'll do this once outside.
                
                # Since I'm writing the code now, I should have done that.
                # Let's refactor to use precomputed structures.
                
                # Break out of inner loop to implement precomputation properly.
                found_move = False
                break
            
            no_improvement_count += 1
            continue
        
        # We have unique colours.
        # Pick a random unique colour.
        target_c = random.choice(unique_colours)
        # Find the edge with this colour.
        # Since it's unique, there is exactly one edge.
        edge_uv = None
        for i in range(n):
            for j in range(i+1, n):
                if colors[i][j] == target_c:
                    edge_uv = (i, j)
                    break
            if edge_uv: break
        
        if not edge_uv: continue
        
        u, v = edge_uv
        
        # Try to change it to a non-unique colour.
        non_unique = [c for c, cnt in counts.items() if cnt > 1]
        if not non_unique:
            # All colours are unique? Impossible with 120 edges and <120 colours usually.
            # If all unique, we have 120 colours. We want to reduce.
            # We must change one to another.
            non_unique = [c for c in counts.keys() if c != target_c]
        
        if not non_unique:
            break
            
        random.shuffle(non_unique)
        
        improved = False
        for new_c in non_unique:
            # Try change (u,v) to new_c
            # Check validity
            # We need efficient validity check.
            
            # Let's implement the precomputation now.
            # I will restart the function structure to include precomputation at the start.
            
            # Since I am inside the loop, I'll assume I have `edge_to_k4s`.
            # I will add the precomputation block before the loop in the final code.
            
            # For now, let's assume `edge_to_k4s` exists.
            
            # Check validity
            valid = True
            for k4_idx in edge_to_k4s[(u,v)]:
                edges = k4s[k4_idx]
                c_set = set()
                for (a, b) in edges:
                    if (a == u and b == v) or (a == v and b == u):
                        c_set.add(new_c)
                    else:
                        c_set.add(colors[a][b])
                if len(c_set) < 5:
                    valid = False
                    break
            
            if valid:
                # Accept move
                colors[u][v] = new_c
                colors[v][u] = new_c
                improved = True
                no_improvement_count = 0
                # Update best if better
                curr_v = count_violations() # Should be 0
                curr_c = get_num_colours()
                if curr_v < best_violations or (curr_v == best_violations and curr_c < best_colours):
                    best_violations = curr_v
                    best_colours = curr_c
                    best_colors = [row[:] for row in colors]
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break

    # Return the best solution found
    # Ensure we return a valid solution.
    # If best is not valid, we should return the initial valid one?
    # The problem says "A valid colouring certifies...".
    # If we return an invalid one, score is 0.
    # So we must ensure the returned solution is valid.
    
    # Let's verify best_colors validity.
    # If invalid, fall back to initial.
    
    # Check validity of best_colors
    def is_valid(col):
        for edges in k4s:
            c_set = set()
            for (u, v) in edges:
                c_set.add(col[u][v])
            if len(c_set) < 5:
                return False
        return True
    
    if is_valid(best_colors):
        final_colors = best_colors
    else:
        # Fallback to initial
        final_colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                final_colors[i][j] = (i + j) % 17
                final_colors[j][i] = final_colors[i][j]

    # Flatten to list in lexicographic order
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(final_colors[i][j])
            
    return result

# Precomputation for edge_to_k4s
n = 16
k4s_list = []
for quad in itertools.combinations(range(n), 4):
    a, b, c, d = quad
    edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
    k4s_list.append(edges)

edge_to_k4s = {}
for idx, edges in enumerate(k4s_list):
    for (u, v) in edges:
        if (u, v) not in edge_to_k4s:
            edge_to_k4s[(u, v)] = []
        edge_to_k4s[(u, v)].append(idx)
        if (v, u) not in edge_to_k4s:
            edge_to_k4s[(v, u)] = []
        edge_to_k4s[(v, u)].append(idx)

# Re-define construct to use global precomputed data
def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16,
    in lexicographic order.
    """
    n = 16
    # Initialize with the known valid 17-colour construction
    colors = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            colors[i][j] = (i + j) % 17
            colors[j][i] = colors[i][j]
    
    # Use global k4s_list and edge_to_k4s
    
    def count_violations():
        violations = 0
        for edges in k4s_list:
            c_set = set()
            for (u, v) in edges:
                c_set.add(colors[u][v])
            if len(c_set) < 5:
                violations += 1
        return violations
    
    def get_num_colours():
        used = set()
        for i in range(n):
            for j in range(i+1, n):
                used.add(colors[i][j])
        return len(used)
    
    current_violations = count_violations()
    current_colours = get_num_colours()
    
    best_violations = current_violations
    best_colours = current_colours
    best_colors = [row[:] for row in colors]
    
    start_time = time.time()
    timeout = 110.0
    
    no_improvement_count = 0
    max_no_improvement = 20000 # Increase limit
    
    # Random seed for reproducibility? No, let it be random.
    # random.seed(42) # Uncomment for debugging
    
    while time.time() - start_time < timeout:
        counts = {}
        for i in range(n):
            for j in range(i+1, n):
                c = colors[i][j]
                counts[c] = counts.get(c, 0) + 1
        
        unique_colours = [c for c, cnt in counts.items() if cnt == 1]
        
        if not unique_colours:
            if no_improvement_count > max_no_improvement:
                break
            
            # Try random neutral moves
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                no_improvement_count += 1
                continue
            if u > v:
                u, v = v, u
            
            old_c = colors[u][v]
            used = list(counts.keys())
            if len(used) <= 1:
                break
            new_c = random.choice(used)
            if new_c == old_c:
                no_improvement_count += 1
                continue
            
            # Check validity
            valid = True
            for k4_idx in edge_to_k4s.get((u,v), []):
                edges = k4s_list[k4_idx]
                c_set = set()
                for (a, b) in edges:
                    if (a == u and b == v) or (a == v and b == u):
                        c_set.add(new_c)
                    else:
                        c_set.add(colors[a][b])
                if len(c_set) < 5:
                    valid = False
                    break
            
            if valid:
                colors[u][v] = new_c
                colors[v][u] = new_c
                # Check if this reduced colour count (unlikely for neutral move, but possible if new_c was used and old_c becomes unused? No, old_c count was >=2 if not unique, or 1 if unique. If we picked from used, and old_c was unique, we would have gone to the other branch. So here old_c count >= 2. New_c count >= 1. After move, old_c count >= 1, new_c count >= 2. Colour count stays same.)
                # So this is a neutral move.
                no_improvement_count += 1 # Count as no improvement in colour count
            else:
                no_improvement_count += 1
            continue
        
        # We have unique colours
        target_c = random.choice(unique_colours)
        edge_uv = None
        for i in range(n):
            for j in range(i+1, n):
                if colors[i][j] == target_c:
                    edge_uv = (i, j)
                    break
            if edge_uv: break
        
        if not edge_uv:
            no_improvement_count += 1
            continue
        
        u, v = edge_uv
        
        non_unique = [c for c, cnt in counts.items() if cnt > 1]
        if not non_unique:
            non_unique = [c for c in counts.keys() if c != target_c]
        
        if not non_unique:
            break
            
        random.shuffle(non_unique)
        
        improved = False
        for new_c in non_unique:
            valid = True
            for k4_idx in edge_to_k4s.get((u,v), []):
                edges = k4s_list[k4_idx]
                c_set = set()
                for (a, b) in edges:
                    if (a == u and b == v) or (a == v and b == u):
                        c_set.add(new_c)
                    else:
                        c_set.add(colors[a][b])
                if len(c_set) < 5:
                    valid = False
                    break
            
            if valid:
                colors[u][v] = new_c
                colors[v][u] = new_c
                improved = True
                no_improvement_count = 0
                
                # Update best
                curr_v = count_violations()
                curr_c = get_num_colours()
                if curr_v < best_violations or (curr_v == best_violations and curr_c < best_colours):
                    best_violations = curr_v
                    best_colours = curr_c
                    best_colors = [row[:] for row in colors]
                break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                break

    # Verify and return
    def is_valid(col):
        for edges in k4s_list:
            c_set = set()
            for (u, v) in edges:
                c_set.add(col[u][v])
            if len(c_set) < 5:
                return False
        return True
    
    if is_valid(best_colors):
        final_colors = best_colors
    else:
        final_colors = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                final_colors[i][j] = (i + j) % 17
                final_colors[j][i] = final_colors[i][j]

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(final_colors[i][j])
            
    return result

# EVOLVE-BLOCK-END
