import numpy as np
from itertools import combinations
import random
import time

def construct():
    # Target N from 43 down to 38
    # We will use Simulated Annealing
    
    def count_violations(adj, n):
        """Count K5 and Independent 5-sets."""
        count = 0
        for combo in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = combo[i], combo[j]
                    if adj[u, v]:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                count += 1
        return count

    def simulate_annealing(n, time_limit=20):
        """
        Try to find a valid graph on n vertices using SA.
        Returns (n, edges) if found, else None.
        """
        # Initialize random graph
        adj = np.random.randint(0, 2, size=(n, n), dtype=np.int8)
        np.fill_diagonal(adj, 0)
        # Make symmetric
        adj = (adj + adj.T) % 2
        
        # Precompute list of 5-sets containing each edge?
        # For efficiency in delta, we iterate over all 5-sets containing the flipped edge.
        # Total 5-sets is C(n, 5).
        # For a specific edge (u,v), the 5-sets are {u, v, a, b, c} where {a,b,c} is a subset of V \ {u,v}.
        
        # Initial cost
        current_cost = count_violations(adj, n)
        
        if current_cost == 0:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            return n, edges

        T = 1.0
        cooling = 0.9995
        start_time = time.time()
        
        while T > 0.01 and (time.time() - start_time) < time_limit:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Calculate delta cost
            # Only 5-sets containing u and v are affected
            others = [x for x in range(n) if x != u and x != v]
            delta_cost = 0
            
            # For each 3-subset of others
            for abc in combinations(others, 3):
                full_set = [u, v] + list(abc)
                
                # Check state BEFORE flip
                is_clique_old = True
                is_indep_old = True
                for i in range(5):
                    for j in range(i+1, 5):
                        x, y = full_set[i], full_set[j]
                        if x == u or x == v:
                            # Edge involving u or v
                            other = v if x == u else u
                            # Wait, if x==u, other is v. If x==v, other is u.
                            # But we are flipping (u,v).
                            # In the "old" state, adj[u,v] is the current value.
                            # We need to check the edge (u,v) specifically.
                            pass # This logic is getting messy.
                            
                # Let's simplify: 
                # Just recalculate the property of the 5-set based on current adj, 
                # then flip, recalculate, then unflip? No, that's 2x work.
                
                # Better: 
                # old_clique = is_clique(full_set, adj)
                # old_indep = is_indep(full_set, adj)
                # new_adj_uv = 1 - adj[u,v]
                # new_clique = ...
                
                # Let's do a helper function
                def check_set(s, a, b, val_ab):
                    # s is list of 5 vertices
                    # a, b are the two vertices whose edge is being considered with value val_ab
                    # other edges use adj matrix
                    is_c = True
                    is_i = True
                    for i in range(5):
                        for j in range(i+1, 5):
                            x, y = s[i], s[j]
                            if (x == a and y == b) or (x == b and y == a):
                                edge_val = val_ab
                            else:
                                edge_val = adj[x, y]
                            
                            if edge_val:
                                is_i = False
                            else:
                                is_c = False
                            if not is_c and not is_i:
                                break
                        if not is_c and not is_i:
                            break
                    return is_c, is_i

                old_c, old_i = check_set(full_set, u, v, adj[u, v])
                new_val = 1 - adj[u, v]
                new_c, new_i = check_set(full_set, u, v, new_val)
                
                old_violation = 1 if (old_c or old_i) else 0
                new_violation = 1 if (new_c or new_i) else 0
                delta_cost += (new_violation - old_violation)
            
            # Acceptance
            if delta_cost < 0:
                # Flip
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                current_cost += delta_cost
            else:
                # Probabilistic accept
                if random.random() < np.exp(-delta_cost / T):
                    adj[u, v] = 1 - adj[u, v]
                    adj[v, u] = adj[u, v]
                    current_cost += delta_cost
            
            T *= cooling
            
            if current_cost == 0:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            edges.append((i, j))
                return n, edges

        return None

    # Try N from 43 down to 38
    for n in range(43, 37, -1):
        # Run SA for a few seconds
        result = simulate_annealing(n, time_limit=20)
        if result:
            return result
            
    # Fallback: Return the seed Paley graph (37 vertices) to ensure non-zero score?
    # The task says "invalid result ... scores 0". 
    # The seed program returns a valid graph. 
    # If my SA fails, I should probably return a valid smaller graph.
    # Let's just return the Paley graph construction as fallback.
    
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    edges = []
    for i in range(q):
        for j in range(i+1, q):
            if adj[i, j]:
                edges.append((i, j))
    return q, edges
