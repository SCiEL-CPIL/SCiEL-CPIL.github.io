import time
import random
import math

def construct():
    deadline = time.time() + 135
    random.seed(42)
    
    # Seed: n=32 Greenwood-Gleason
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
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
        
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count

    def try_add_vertex_sa(adj, n, time_limit=5.0):
        """
        Try to add vertex n to the existing coloring of n vertices.
        Uses Simulated Annealing to find a valid assignment of colors for edges (n, 0..n-1).
        """
        start_time = time.time()
        
        # Precompute for each color k, the graph G_k among existing vertices.
        # We need to check if a set of vertices is an independent set in G_k.
        # Instead of checking sets, we can just count conflicts.
        
        # Initial assignment: random
        assignment = [random.randint(0, 3) for _ in range(n)]
        
        def get_energy(assign):
            # Energy is the number of pairs (u, v) such that assign[u] == assign[v] == k and adj[u][v] == k
            energy = 0
            for u in range(n):
                cu = assign[u]
                for v in range(u + 1, n):
                    if assign[v] == cu and adj[u][v] == cu:
                        energy += 1
            return energy

        current_energy = get_energy(assignment)
        
        if current_energy == 0:
            return assignment

        best_assignment = assignment[:]
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            # Pick a random vertex to change color
            u = random.randint(0, n - 1)
            old_color = assignment[u]
            new_color = random.choice([0, 1, 2, 3])
            if new_color == old_color:
                continue
                
            # Calculate delta energy
            # Change assign[u] from old_color to new_color
            delta = 0
            for v in range(n):
                if v == u:
                    continue
                cv = assignment[v]
                
                # Remove old contributions involving u
                # Pair (u, v) was (old_color, cv)
                # If old_color == cv and adj[u][v] == old_color, it contributed 1.
                if old_color == cv and adj[u][v] == old_color:
                    delta -= 1
                
                # Add new contributions involving u
                # Pair (u, v) is now (new_color, cv)
                # If new_color == cv and adj[u][v] == new_color, it contributes 1.
                if new_color == cv and adj[u][v] == new_color:
                    delta += 1
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                assignment[u] = new_color
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
                    
                    if best_energy == 0:
                        return best_assignment

            T *= cooling
            
        return best_assignment if best_energy == 0 else None

    # Start
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Try to expand to 50
    for target_n in range(33, 51):
        if time.time() > deadline - 10:
            break
            
        n = len(adj)
        # We are adding vertex n (0-indexed, so it becomes the n-th vertex in a list of n+1)
        # The existing vertices are 0..n-1
        
        found = False
        # Try multiple times with different random seeds for SA
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            random.seed(42 + target_n + attempt)
            assignment = try_add_vertex_sa(adj, n, time_limit=2.0)
            if assignment is not None:
                # Success! Add vertex
                new_n = n + 1
                new_adj = [row[:] for row in adj]
                for _ in range(new_n):
                    new_adj.append([-1] * new_n)
                for i in range(n):
                    new_adj[i][n] = new_adj[n][i] = assignment[i]
                new_adj[n][n] = -1 # Self-loop, doesn't matter
                
                adj = new_adj
                best_n = new_n
                best_colors = adj_to_colors(adj, best_n)
                found = True
                break
                
        if not found:
            # If we can't add the next vertex, we might try to repair the current one
            # But for simplicity, we stop if we can't extend.
            # However, maybe a different ordering or repair could help?
            # For now, break.
            break
            
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
