import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin

    # Helper to check if a graph is (5,5)-free
    def is_valid(adj):
        n = adj.shape[0]
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            if all(adj[i][j] == 1 for i, j in itertools.combinations(combo, 2)):
                return False
        # Check for Independent Set 5
        for combo in itertools.combinations(range(n), 5):
            if all(adj[i][j] == 0 for i, j in itertools.combinations(combo, 2)):
                return False
        return True

    # Convert adj matrix to edge list
    def to_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    edges.append((i, j))
        return edges

    # Fallback: Paley 37
    def paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    # Precompute 4-cliques and 4-independent sets for a given adjacency matrix
    def get_4_sets(adj):
        n = adj.shape[0]
        four_cliques = []
        four_indeps = []
        for combo in itertools.combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i, j in itertools.combinations(combo, 2):
                if adj[i][j] == 1:
                    is_indep = False
                else:
                    is_clique = False
                if not is_clique and not is_indep:
                    break
            if is_clique:
                four_cliques.append(combo)
            elif is_indep:
                four_indeps.append(combo)
        return four_cliques, four_indeps

    # Try to add a new vertex to the graph using min-conflicts heuristic
    def try_add_vertex(adj, four_cliques, four_indeps, time_limit):
        n = adj.shape[0]
        # Initial random connection vector
        c = np.random.randint(0, 2, n)
        
        # Function to count violations
        def count_violations(c):
            v_clique = 0
            v_indep = 0
            for clique in four_cliques:
                if all(c[v] == 1 for v in clique):
                    v_clique += 1
            for indep in four_indeps:
                if all(c[v] == 0 for v in indep):
                    v_indep += 1
            return v_clique + v_indep, v_clique, v_indep
        
        current_violations, _, _ = count_violations(c)
        
        if current_violations == 0:
            return c
        
        max_steps = 10000
        for step in range(max_steps):
            if time.time() > time_limit:
                break
            
            if current_violations == 0:
                return c
            
            # Pick a random violated constraint
            if random.random() < 0.5:
                if four_cliques:
                    clique = random.choice(four_cliques)
                    if all(c[v] == 1 for v in clique):
                        v_to_flip = random.choice(clique)
                        c[v_to_flip] = 0
                        current_violations, _, _ = count_violations(c)
                        continue
            
            if random.random() < 0.5:
                if four_indeps:
                    indep = random.choice(four_indeps)
                    if all(c[v] == 0 for v in indep):
                        v_to_flip = random.choice(indep)
                        c[v_to_flip] = 1
                        current_violations, _, _ = count_violations(c)
                        continue
            
            # Random flip
            v_to_flip = random.randint(0, n - 1)
            c[v_to_flip] = 1 - c[v_to_flip]
            current_violations, _, _ = count_violations(c)
        
        return None

    # Main logic: Start with Paley 37 and try to extend
    seed_n, seed_edges = paley_37()
    best_n = seed_n
    best_edges = seed_edges
    
    adj = np.zeros((seed_n, seed_n), dtype=int)
    for i, j in seed_edges:
        adj[i][j] = 1
        adj[j][i] = 1
        
    four_cliques, four_indeps = get_4_sets(adj)
    
    target_n = 42
    for current_n in range(seed_n, target_n):
        if time.time() > deadline - 20:
            break
            
        added = False
        attempts = 0
        max_attempts = 10
        
        while not added and attempts < max_attempts:
            attempts += 1
            attempt_deadline = time.time() + 20.0
            
            c = try_add_vertex(adj, four_cliques, four_indeps, attempt_deadline)
            
            if c is not None:
                new_n = current_n + 1
                new_adj = np.zeros((new_n, new_n), dtype=int)
                new_adj[:current_n, :current_n] = adj
                new_adj[:current_n, current_n] = c
                new_adj[current_n, :current_n] = c
                
                best_n = new_n
                best_edges = to_edges(new_adj)
                
                four_cliques, four_indeps = get_4_sets(new_adj)
                adj = new_adj
                added = True
                break
        
        if not added:
            break
            
    if best_n > seed_n:
        check_adj = np.zeros((best_n, best_n), dtype=int)
        for i, j in best_edges:
            check_adj[i][j] = 1
            check_adj[j][i] = 1
            
        if is_valid(check_adj):
            return best_n, best_edges
        else:
            return seed_n, seed_edges
            
    return seed_n, seed_edges
