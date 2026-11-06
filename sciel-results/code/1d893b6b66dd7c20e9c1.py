import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin

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

    def to_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    edges.append((i, j))
        return edges

    def paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    # Precompute 4-cliques and 4-independent sets
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

    # Optimized Min-Conflicts with Simulated Annealing and Restarts
    def try_add_vertex(adj, four_cliques, four_indeps, time_limit):
        n = adj.shape[0]
        
        # Pre-compute which 4-sets contain which vertex for fast updates
        cliques_by_vertex = [[] for _ in range(n)]
        indeps_by_vertex = [[] for _ in range(n)]
        
        for idx, clique in enumerate(four_cliques):
            for v in clique:
                cliques_by_vertex[v].append(idx)
                
        for idx, indep in enumerate(four_indeps):
            for v in indep:
                indeps_by_vertex[v].append(idx)

        # Initial random connection vector
        c = np.random.randint(0, 2, n)
        
        # State: sets of violated indices
        violated_cliques = set()
        violated_indeps = set()
        
        # Initialize violation sets
        for idx, clique in enumerate(four_cliques):
            if all(c[v] == 1 for v in clique):
                violated_cliques.add(idx)
                
        for idx, indep in enumerate(four_indeps):
            if all(c[v] == 0 for v in indep):
                violated_indeps.add(idx)
                
        current_violations = len(violated_cliques) + len(violated_indeps)
        
        if current_violations == 0:
            return c

        # Simulated Annealing parameters
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        max_steps_per_restart = 2000
        max_restarts = 50
        
        best_c = c.copy()
        best_violations = current_violations
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_c = c.copy()
                if best_violations == 0:
                    return best_c
            
            # Reset temperature for each restart or keep cooling?
            # Let's try cooling across restarts but reset if stuck
            T = max(T, 0.5) 
            
            for step in range(max_steps_per_restart):
                if time.time() > time_limit:
                    break
                
                if current_violations == 0:
                    return c
                    
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_c = c.copy()
                    if best_violations == 0:
                        return best_c

                # Pick a random violated constraint
                # We prefer to pick from the larger set to balance
                if violated_cliques and (not violated_indeps or random.random() < 0.5):
                    # Pick a violated clique
                    if not violated_cliques: continue
                    idx = random.choice(list(violated_cliques))
                    clique = four_cliques[idx]
                    # Flip a random vertex in this clique from 1 to 0
                    v_to_flip = random.choice(clique)
                    old_val = c[v_to_flip]
                    if old_val != 1: continue # Should be 1 if it's a violated clique? 
                    # Actually, a violated clique means ALL are 1. So old_val is 1.
                    c[v_to_flip] = 0
                    
                    # Update violations for cliques containing v_to_flip
                    for c_idx in cliques_by_vertex[v_to_flip]:
                        cl = four_cliques[c_idx]
                        # Check if it was violated before (it was if c_idx in violated_cliques)
                        # It is violated if all c[v]==1
                        # Since we set c[v_to_flip]=0, it is NO LONGER violated.
                        if c_idx in violated_cliques:
                            violated_cliques.remove(c_idx)
                            current_violations -= 1
                        # Could it become violated? No, we set a bit to 0, so it can't become all 1s.
                        
                    # Update violations for indeps containing v_to_flip
                    for i_idx in indeps_by_vertex[v_to_flip]:
                        ind = four_indeps[i_idx]
                        # It was violated if all c[v]==0.
                        # We set c[v_to_flip] from 1 to 0.
                        # So it MIGHT become violated if all OTHERS are 0.
                        if i_idx not in violated_indeps:
                            # Check if all other vertices in ind are 0
                            # We know c[v_to_flip] is now 0.
                            all_zero = True
                            for v in ind:
                                if v == v_to_flip: continue
                                if c[v] != 0:
                                    all_zero = False
                                    break
                            if all_zero:
                                violated_indeps.add(i_idx)
                                current_violations += 1
                                
                elif violated_indeps:
                    # Pick a violated indep
                    if not violated_indeps: continue
                    idx = random.choice(list(violated_indeps))
                    indep = four_indeps[idx]
                    # Flip a random vertex in this indep from 0 to 1
                    v_to_flip = random.choice(indep)
                    old_val = c[v_to_flip]
                    if old_val != 0: continue
                    c[v_to_flip] = 1
                    
                    # Update violations for indeps containing v_to_flip
                    for i_idx in indeps_by_vertex[v_to_flip]:
                        ind = four_indeps[i_idx]
                        # It was violated if all c[v]==0.
                        # We set c[v_to_flip] from 0 to 1.
                        # So it is NO LONGER violated.
                        if i_idx in violated_indeps:
                            violated_indeps.remove(i_idx)
                            current_violations -= 1
                        # Could it become violated? No, we set a bit to 1.
                        
                    # Update violations for cliques containing v_to_flip
                    for c_idx in cliques_by_vertex[v_to_flip]:
                        cl = four_cliques[c_idx]
                        # It was violated if all c[v]==1.
                        # We set c[v_to_flip] from 0 to 1.
                        # So it MIGHT become violated if all OTHERS are 1.
                        if c_idx not in violated_cliques:
                            all_one = True
                            for v in cl:
                                if v == v_to_flip: continue
                                if c[v] != 1:
                                    all_one = False
                                    break
                            if all_one:
                                violated_cliques.add(c_idx)
                                current_violations += 1
                else:
                    # No violations? Should have returned.
                    break

                # Simulated Annealing: Allow worse moves
                # We don't have a "delta" easily here without storing previous state.
                # For simplicity, let's stick to Hill Climbing with Restarts first.
                # If we want SA, we need to store prev_c and prev_violations.
                # Given the complexity, Restarts are often effective for Min-Conflicts.
                
                T *= cooling_rate
                if T < min_T:
                    break

        return best_c if best_violations == 0 else None

    # Main logic
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
        if time.time() > deadline - 30:
            break
            
        added = False
        # Try multiple times
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            attempt_deadline = time.time() + 5.0 # 5s per attempt
            
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
