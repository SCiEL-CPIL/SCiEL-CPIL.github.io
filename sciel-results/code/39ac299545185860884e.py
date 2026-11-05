import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    best_score = 18
    best_edges = []
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def find_independent_set_size_4(adj):
        # Returns a list of independent sets of size 4 in adj (which is H)
        # Independent in H means no edges in H
        indep_sets = []
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if not (adj[u][v] or adj[u][w] or adj[u][x] or
                    adj[v][w] or adj[v][x] or
                    adj[w][x]):
                indep_sets.append(quad)
        return indep_sets

    def enforce_alpha_le_3(adj):
        # Ensure no independent set of size 4 in adj (H)
        # This means G (complement) is K4-free
        adj = [row[:] for row in adj]
        while True:
            indep_sets = find_independent_set_size_4(adj)
            if not indep_sets:
                break
            # Pick a random independent set and add an edge to H (remove from G)
            # Adding an edge to H breaks the independence
            quad = random.choice(indep_sets)
            u, v, w, x = quad
            # Add a random edge among these 4 in H
            pair = random.choice(list(itertools.combinations(quad, 2)))
            adj[pair[0]][pair[1]] = 1
            adj[pair[1]][pair[0]] = 1
        return adj

    def is_good_subset(adj, subset):
        # Check if H[subset] is triangle-free and alpha(H[subset]) <= 2
        # Triangle-free in H: no 3 vertices with all edges in H
        # Alpha <= 2: no 3 vertices with no edges in H
        
        verts = list(subset)
        k = len(verts)
        
        # Check for triangle in H
        for i in range(k):
            for j in range(i + 1, k):
                if adj[verts[i]][verts[j]]:
                    for l in range(j + 1, k):
                        if adj[verts[i]][verts[l]] and adj[verts[j]][verts[l]]:
                            return False # Has triangle in H
        
        # Check for independent set of size 3 in H (alpha >= 3)
        for i in range(k):
            for j in range(i + 1, k):
                if not adj[verts[i]][verts[j]]:
                    for l in range(j + 1, k):
                        if not adj[verts[i]][verts[l]] and not adj[verts[j]][verts[l]]:
                            return False # Has independent set of size 3 in H
        
        return True

    def calculate_score(adj):
        # Max size of subset S such that H[S] is triangle-free and alpha(H[S]) <= 2
        # We know score <= 5
        for k in [5, 4, 3]:
            for subset in itertools.combinations(range(n), k):
                if is_good_subset(adj, subset):
                    return k
        return 2 # Should not happen for n=18

    # Main Loop
    for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
        while time.time() < deadline:
            # Generate random H
            adj = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Enforce alpha(H) <= 3
            adj = enforce_alpha_le_3(adj)
            
            # Calculate score
            score = calculate_score(adj)
            
            if score < best_score:
                best_score = score
                # Convert H to G (complement)
                g_adj = [[0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if not adj[i][j]:
                            g_adj[i][j] = 1
                            g_adj[j][i] = 1
                best_edges = get_edges_from_adj(g_adj)
                
                if best_score == 4:
                    # Optimal (theoretical lower bound)
                    return best_edges

    return best_edges
