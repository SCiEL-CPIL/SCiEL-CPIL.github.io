"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin before 330s kill
    
    # Seed: Paley graph on 37 vertices (verified (5,5)-free)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Build adjacency matrix for seed
    adj = [[False] * q for _ in range(q)]
    for i, j in seed_edges:
        adj[i][j] = True
        adj[j][i] = True
    
    best_n = q
    best_edges = seed_edges
    
    # Helper: check if a set of vertices forms a clique
    def is_clique(vertices, adj):
        for a, b in itertools.combinations(vertices, 2):
            if not adj[a][b]:
                return False
        return True
    
    # Helper: check if a set of vertices is independent
    def is_independent(vertices, adj):
        for a, b in itertools.combinations(vertices, 2):
            if adj[a][b]:
                return False
        return True
    
    # Precompute all 4-cliques and 4-independent-sets for the seed graph
    def get_4_cliques(n, adj):
        cliques = []
        for combo in itertools.combinations(range(n), 4):
            if is_clique(combo, adj):
                cliques.append(combo)
        return cliques
    
    def get_4_independent(n, adj):
        indep = []
        for combo in itertools.combinations(range(n), 4):
            if is_independent(combo, adj):
                indep.append(combo)
        return indep
    
    # For the seed graph, compute 4-cliques and 4-independents
    four_cliques = get_4_cliques(q, adj)
    four_indep = get_4_independent(q, adj)
    
    # Try to grow the graph by adding vertices
    # Strategy: for each new vertex, try random connection patterns
    # and check if they preserve (5,5)-freeness
    
    random.seed(42)
    
    current_n = q
    current_adj = [row[:] for row in adj]
    
    # Precompute 4-cliques and 4-independent sets for fast checking
    def get_4_cliques_fast(n, adj):
        cliques = []
        for combo in itertools.combinations(range(n), 4):
            all_conn = True
            for a, b in itertools.combinations(combo, 2):
                if not adj[a][b]:
                    all_conn = False
                    break
            if all_conn:
                cliques.append(combo)
        return cliques
    
    def get_4_independent_fast(n, adj):
        indep = []
        for combo in itertools.combinations(range(n), 4):
            all_disc = True
            for a, b in itertools.combinations(combo, 2):
                if adj[a][b]:
                    all_disc = False
                    break
            if all_disc:
                indep.append(combo)
        return indep
    
    # For efficiency, let's use a different approach:
    # When adding a new vertex v, it must NOT connect to all vertices of any 4-clique
    # and it must NOT be non-connected to all vertices of any 4-independent set
    
    four_cliques = get_4_cliques_fast(current_n, current_adj)
    four_indep = get_4_independent_fast(current_n, current_adj)
    
    # Try to add up to a few more vertices
    max_add_attempts = 500
    
    attempts = 0
    while time.time() < deadline and attempts < max_add_attempts:
        attempts += 1
        current_n = len(current_adj)
        
        # Try random connection pattern for new vertex
        # For symmetry breaking, we can try different densities
        new_conn = []
        for v in range(current_n):
            # Random connection with bias toward ~0.5
            new_conn.append(random.random() < 0.5)
        
        # Check 4-cliques: new vertex cannot connect to all 4 vertices of any 4-clique
        valid = True
        for clique in four_cliques:
            if all(new_conn[v] for v in clique):
                valid = False
                break
        
        if valid:
            # Check 4-independent: new vertex cannot be non-connected to all 4 vertices of any 4-ind
            for ind in four_indep:
                if not any(new_conn[v] for v in ind):
                    valid = False
                    break
        
        if valid:
            # Add the vertex
            new_row = [False] * (current_n + 1)
            for v in range(current_n):
                current_adj[v].append(new_conn[v])
                new_row[v] = new_conn[v]
            new_row[current_n] = False
            current_adj.append(new_row)
            
            # Update four_cliques and four_indep
            four_cliques = get_4_cliques_fast(len(current_adj), current_adj)
            four_indep = get_4_independent_fast(len(current_adj), current_adj)
            
            # Update best
            edges = []
            for i in range(len(current_adj)):
                for j in range(i + 1, len(current_adj)):
                    if current_adj[i][j]:
                        edges.append((i, j))
            best_n = len(current_adj)
            best_edges = edges
            
            if time.time() > deadline - 15:
                break
    
    # Also try a few more specific strategies if time permits
    if time.time() < deadline - 20:
        # Try to see if we can reach 42 (Exoo's bound) with more systematic search
        # For now, return what we have
        pass
    
    # Convert to edge list
    n = best_n
    edges = best_edges
    
    # Safety: verify the result is valid
    adj_check = [[False] * n for _ in range(n)]
    for i, j in edges:
        adj_check[i][j] = True
        adj_check[j][i] = True
    
    # Verify no 5-clique
    for combo in itertools.combinations(range(n), 5):
        all_conn = True
        for a, b in itertools.combinations(combo, 2):
            if not adj_check[a][b]:
                all_conn = False
                break
        if all_conn:
            # Fallback to seed
            return seed_n, seed_edges
    
    # Verify no 5-independent
    for combo in itertools.combinations(range(n), 5):
        all_disc = True
        for a, b in itertools.combinations(combo, 2):
            if adj_check[a][b]:
                all_disc = False
                break
        if all_disc:
            # Fallback to seed
            return seed_n, seed_edges
    
    return n, edges
# EVOLVE-BLOCK-END
