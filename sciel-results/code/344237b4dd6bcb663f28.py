"""Combined program: Deterministic seed + Local Search Extension."""
import time
import random
import copy

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # 115 seconds to leave margin for overhead/return

    # 1. Seed: Greenwood-Gleason colouring of K16 (Clebsch) doubled to n=32
    # GF(16) multiplication
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Precompute log table for GF(16)
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Initial n=32 construction
    def get_seed_graph(n):
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    c = log[(i % 16) ^ (j % 16)] % 3
                else:
                    c = 3
                graph[i][j] = c
                graph[j][i] = c
        return graph

    current_n = 32
    graph = get_seed_graph(current_n)
    best_n = current_n
    best_graph = copy.deepcopy(graph)

    def is_valid_graph(g, n):
        """Check if the graph has no monochromatic triangles."""
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        return False
        return True

    def count_mono_triangles(g, n):
        """Count monochromatic triangles in the graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = g[i][j]
                for k in range(j + 1, n):
                    if g[i][k] == c_ij and g[j][k] == c_ij:
                        count += 1
        return count

    def local_search_repair(g, n, max_iter=10000, time_limit=None):
        """
        Attempt to repair the graph by fixing monochromatic triangles.
        Returns True if the graph is valid after repair, False otherwise.
        """
        if time_limit is None:
            time_limit = deadline - time.time()
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                return False
            
            # Find a monochromatic triangle
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = g[i][j]
                    for k in range(j + 1, n):
                        if g[i][k] == c_ij and g[j][k] == c_ij:
                            # Found mono triangle (i, j, k) with color c_ij
                            # Try to break it by changing one of the three edges
                            # Heuristic: change the edge that appears in the most other mono triangles
                            # For simplicity, randomly pick one of the three edges to recolor
                            edge_choices = [(i, j), (i, k), (j, k)]
                            random.shuffle(edge_choices)
                            
                            fixed = False
                            for (a, b) in edge_choices:
                                old_color = g[a][b]
                                # Try all other colors
                                for new_color in [0, 1, 2, 3]:
                                    if new_color == old_color:
                                        continue
                                    g[a][b] = new_color
                                    g[b][a] = new_color
                                    
                                    # Check if this creates new mono triangles involving (a,b)
                                    creates_new = False
                                    for m in range(n):
                                        if m == a or m == b:
                                            continue
                                        # Check triangle (m, a, b)
                                        if g[m][a] == new_color and g[m][b] == new_color:
                                            creates_new = True
                                            break
                                        # Check triangle (a, m, b) - same as above
                                        # Check triangle (a, b, m) - same as above
                                        
                                    if not creates_new:
                                        # Also check if we fixed the original triangle
                                        # The original triangle was (i,j,k) with color c_ij
                                        # If we changed edge (a,b), check if (i,j,k) is still mono
                                        if a == i and b == j:
                                            # Changed (i,j), so triangle (i,j,k) is broken
                                            pass
                                        elif a == i and b == k:
                                            # Changed (i,k), so triangle (i,j,k) is broken
                                            pass
                                        elif a == j and b == k:
                                            # Changed (j,k), so triangle (i,j,k) is broken
                                            pass
                                        
                                        found = True
                                        fixed = True
                                        break
                                    else:
                                        # Revert
                                        g[a][b] = old_color
                                        g[b][a] = old_color
                            
                            if fixed:
                                break
                        
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # No more mono triangles
                return True
        
        # Check if valid
        return is_valid_graph(g, n)

    def add_vertex_backtrack(g, n, time_limit=1.0):
        """
        Attempts to add a new vertex at index n to graph g (size n).
        Returns True if successful (g is updated to size n+1), False otherwise.
        Uses backtracking with pruning and time limit.
        """
        # We will assign colors to edges (i, n) for i in 0..n-1
        # assignments[i] will be the color of edge (i, n)
        assignments = [-1] * n
        backtrack_start = time.time()
        
        # Precompute for each vertex i, which colors are forbidden for edge (i,n)
        # based on already-assigned edges (k,n) where k < i
        # Actually, the constraint is: for each k < i, if g[k][i] == c, then assignments[k] != c
        # So for a given i and candidate color c, we need: for all k < i with g[k][i] == c, assignments[k] != c
        
        def backtrack(i):
            if time.time() - backtrack_start > time_limit:
                return False
            
            if i == n:
                return True
            
            # Try each color for edge (i, n)
            # Heuristic: try colors that are less constrained
            # Count how many k < i have g[k][i] == c and assignments[k] == c
            # Actually, we need: for all k < i with g[k][i] == c, assignments[k] != c
            # So the constraint is violated if there exists k < i with g[k][i] == c AND assignments[k] == c
            
            colors_to_try = [0, 1, 2, 3]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                assignments[i] = c
                # Pruning: Check triangles (k, i, n) for all k < i
                valid = True
                for k in range(i):
                    if g[k][i] == c:
                        if assignments[k] == c:
                            valid = False
                            break
                
                if valid:
                    if backtrack(i + 1):
                        return True
                
            assignments[i] = -1
            return False

        if backtrack(0):
            # Success: Update graph
            for i in range(n):
                row = g[i]
                row.append(assignments[i])
            g.append([0] * (n + 1))
            last_row = g[n]
            for i in range(n):
                last_row[i] = assignments[i]
            last_row[n] = 0 # self loop, irrelevant
            return True
        return False

    def try_add_vertex_with_repair(g, n):
        """
        Try to add a vertex. If backtracking fails, try local search repair
        on the existing graph to create room, then retry.
        Returns True if vertex was added successfully.
        """
        # First try direct backtracking
        if add_vertex_backtrack(g, n, time_limit=0.5):
            return True
        
        # If backtracking failed, try to repair the graph to create more flexibility
        # Save state
        old_graph = copy.deepcopy(g)
        old_n = n
        
        # Try local search repair
        if local_search_repair(g, n, max_iter=5000):
            # After repair, try to add vertex again
            if add_vertex_backtrack(g, n, time_limit=1.0):
                return True
        
        # If still failed, revert
        g[:] = [row[:] for row in old_graph]
        # Note: g is a list of lists, so we need to restore properly
        # Actually, let's just keep the repaired graph if it's valid, even if we couldn't add a vertex
        # This might help in future iterations
        if not is_valid_graph(g, n):
            # Revert to old state
            for i in range(n):
                for j in range(n):
                    g[i][j] = old_graph[i][j]
        
        return False

    attempts = 0
    max_attempts = 50000
    consecutive_failures = 0
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Try to add a vertex
        if try_add_vertex_with_repair(graph, current_n):
            current_n += 1
            best_n = current_n
            best_graph = copy.deepcopy(graph)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            
            # If we fail to add a vertex, we need to modify the existing graph to create "room".
            # Strategy: Randomly flip some edges (Kick).
            # We must ensure the graph remains valid after the kick.
            
            # Probability of kicking: Higher if we've failed many times in a row.
            kick_prob = min(0.5, 0.1 * consecutive_failures)
            
            if random.random() < kick_prob:
                # Save state
                old_graph = copy.deepcopy(graph)
                old_n = current_n
                
                # Perform Kick: Flip K edges
                # K should be small, e.g., 5-10% of edges
                num_edges = current_n * (current_n - 1) // 2
                num_flips = max(1, num_edges // 20)
                
                for _ in range(num_flips):
                    i = random.randint(0, current_n - 1)
                    j = random.randint(0, current_n - 1)
                    if i != j:
                        graph[i][j] = random.randint(0, 3)
                        graph[j][i] = graph[i][j]
                
                # Validate
                if is_valid_graph(graph, current_n):
                    # Kick was successful in maintaining validity.
                    pass
                else:
                    # Kick broke validity. Try to repair.
                    if not local_search_repair(graph, current_n, max_iter=2000):
                        # Revert
                        for i in range(current_n):
                            for j in range(current_n):
                                graph[i][j] = old_graph[i][j]
            
            # If we have failed many times and kicking isn't helping, we might be stuck.
            # But we keep trying until deadline.

    # Finalize
    n = best_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(best_graph[i][j])
            
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
