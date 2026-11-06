import time
import random

# EVOLVE-BLOCK-START
def construct():
    # Initial 32-vertex construction
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    n_base = 32
    colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    # Store as adjacency matrix for easier access
    # adj[i][j] = color
    adj = [[-1] * n_base for _ in range(n_base)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors[idx]
            adj[i][j] = c
            adj[j][i] = c
            idx += 1
            
    # Greedy extension
    n = n_base
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Precompute triangles for base? 
    # For a new vertex v, and existing i, j:
    # If adj[i][j] == c, and we set adj[v][i]=c and adj[v][j]=c, we have a mono-triangle (v,i,j) with color c.
    # So for each color c, the set of neighbors of v with color c must be an independent set in the graph of edges of color c.
    
    # Let's try to add vertices one by one.
    # We maintain adj as a list of lists.
    
    def can_add_vertex(adj, n, new_v_idx):
        # Try to assign colors to edges (new_v_idx, 0..n-1)
        # This is a constraint satisfaction problem.
        # Greedy: for each edge (new_v, i), pick a color that doesn't form a triangle.
        # A triangle (new_v, i, j) is monochromatic with color c if:
        # adj[new_v][i] == c, adj[new_v][j] == c, and adj[i][j] == c.
        
        # We assign colors to adj[new_v][i] for i in 0..n-1.
        # Initialize adj[new_v][i] = -1.
        
        # Heuristic: process i in random order? Or fixed?
        # Let's try fixed order first, with backtracking if simple greedy fails.
        # Since n is small (32->50), simple greedy might work.
        
        new_adj = [-1] * n
        # Shuffle indices to break symmetry?
        # indices = list(range(n))
        # random.shuffle(indices)
        
        # For each i, determine allowed colors.
        # Allowed colors for edge (new_v, i) are those c such that:
        # For all j < i (already assigned), if adj[new_v][j] == c, then adj[i][j] != c.
        # Wait, if adj[new_v][j] == c and adj[i][j] == c, then (new_v, i, j) is a mono-triangle.
        # So c is forbidden for edge (new_v, i) if there exists j < i such that adj[new_v][j] == c AND adj[i][j] == c.
        
        # This is a greedy assignment.
        # If we get stuck (no color allowed for some i), we fail.
        
        for i in range(n):
            allowed = [0, 1, 2, 3]
            for j in range(i):
                c_new_j = new_adj[j]
                if c_new_j != -1:
                    if adj[i][j] == c_new_j:
                        # If we assign c_new_j to (new_v, i), we form triangle (new_v, i, j)
                        # So c_new_j is forbidden for (new_v, i)
                        if c_new_j in allowed:
                            allowed.remove(c_new_j)
            if not allowed:
                return None
            # Pick one. Random or first?
            # Random might help escape local minima.
            c = random.choice(allowed)
            new_adj[i] = c
            
        return new_adj

    # Try to extend
    # We will try to add up to 50 vertices.
    # If we fail, we return the best found.
    
    current_adj = [row[:] for row in adj]
    current_n = n_base
    best_n = n_base
    best_adj = [row[:] for row in adj]
    
    # Try multiple random restarts of the greedy process?
    # Or just try to add sequentially.
    
    # To maximize success, we might want to shuffle the order of existing vertices when adding a new one?
    # No, the "j < i" dependency in the greedy check relies on a fixed order.
    # We can shuffle the *labels* of the existing vertices? No, that changes the graph structure.
    # We can shuffle the *order* in which we assign colors to the new vertex's edges.
    
    def try_extend(adj, n):
        # Returns new_adj if successful, else None
        new_adj = [-1] * n
        # Shuffle the order of assignment
        order = list(range(n))
        random.shuffle(order)
        
        # We need to check constraints against *all* previously assigned vertices, not just j < i in the shuffled order.
        # So we maintain a set of assigned vertices.
        assigned = set()
        
        # Map shuffled index to original index
        # For each step k in order:
        # i = order[k]
        # Check against all j in assigned.
        
        for i in order:
            allowed = [0, 1, 2, 3]
            for j in assigned:
                c_new_j = new_adj[j]
                if c_new_j != -1:
                    if adj[i][j] == c_new_j:
                        if c_new_j in allowed:
                            allowed.remove(c_new_j)
            if not allowed:
                return None
            c = random.choice(allowed)
            new_adj[i] = c
            assigned.add(i)
            
        return new_adj

    # Loop to add vertices
    # We have 100s. 
    # Adding one vertex is O(n^2) roughly. 50^2 = 2500. Very fast.
    # We can try many times.
    
    # Strategy: 
    # 1. Try to add vertex 32.
    # 2. If success, add 33, etc.
    # 3. If fail at some step, retry that step with different random seeds/shuffles.
    # 4. If fail multiple times, stop.
    
    max_attempts_per_vertex = 100
    target_n = 50
    
    # Copy current state
    curr_adj = [row[:] for row in current_adj]
    curr_n = current_n
    
    for target in range(curr_n, target_n):
        success = False
        for attempt in range(max_attempts_per_vertex):
            if time.time() > deadline:
                break
            new_row = try_extend(curr_adj, curr_n)
            if new_row is not None:
                # Success
                # Append new row to curr_adj
                curr_adj.append(new_row)
                # Update existing rows
                for i in range(curr_n):
                    curr_adj[i].append(new_row[i])
                curr_n += 1
                success = True
                break
        if not success:
            break
        # Update best
        if curr_n > best_n:
            best_n = curr_n
            best_adj = [row[:] for row in curr_adj]
            
    # Convert best_adj to list of colors
    final_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            final_colors.append(best_adj[i][j])
            
    return {'n': best_n, 'colors': final_colors}
# EVOLVE-BLOCK-END
