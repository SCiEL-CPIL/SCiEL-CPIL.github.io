import itertools
import time
import random

def construct():
        start_time = time.time()
        timeout = 290
        
        # Fallback: Paley 37
        def get_paley(q):
            residues = {(x * x) % q for x in range(1, q)}
            edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
            return q, edges
        
        n_seed, edges_seed = get_paley(37)
        
        # Try to build n=42
        n_target = 42
        # Precompute all 5-sets
        all_5sets = list(itertools.combinations(range(n_target), 5))
        
        # Helper to check if a 5-set is a clique or independent given adj
        def is_bad_5set(combo, adj):
            # Check Clique
            ok_clique = True
            # Check Independent
            ok_indep = True
            for i in range(5):
                u = combo[i]
                for j in range(i+1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        ok_indep = False
                    else:
                        ok_clique = False
                    if not ok_clique and not ok_indep:
                        return False
            return ok_clique or ok_indep

        # Local Search
        best_result = (n_seed, edges_seed)
        
        for restart in range(20): # 20 restarts
            if time.time() - start_time > timeout:
                break
            
            # Initialize random graph
            random.seed(42 + restart)
            adj = [0] * n_target
            edges = set()
            for u in range(n_target):
                for v in range(u+1, n_target):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
            
            # Initial violation set
            violations = set()
            for c in all_5sets:
                if is_bad_5set(c, adj):
                    violations.add(c)
            
            # Local Search Loop
            max_steps = 5000
            for step in range(max_steps):
                if time.time() - start_time > timeout:
                    break
                
                if not violations:
                    # Success!
                    edge_list = list(edges)
                    return n_target, edge_list
                
                # Pick a random violation
                v_set = random.choice(list(violations))
                # Pick a random edge within this 5-set to flip
                # We want to flip an edge that is "responsible" for the violation
                # For a clique, we should remove an edge. For an independent set, we should add an edge.
                # Let's determine the type
                is_clique = all(adj[u] & (1 << v) for i, u in enumerate(v_set) for v in v_set[i+1:])
                
                # Generate pairs in v_set
                pairs = list(itertools.combinations(v_set, 2))
                u, v = random.choice(pairs)
                if u > v: u, v = v, u
                
                # Flip edge (u, v)
                if is_clique:
                    # Remove edge
                    if (u, v) in edges:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        edges.remove((u, v))
                    else:
                        # Should not happen if is_clique is true for this set
                        continue 
                else:
                    # Add edge
                    if (u, v) not in edges:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
                    else:
                        continue
                        
                # Update violations
                # Only 5-sets containing (u,v) can change status
                # Find all 5-sets containing u and v
                # Precomputed: map from (u,v) to list of 5-sets?
                # For n=42, binom(40,3)=9880 sets per edge.
                # Iterating 9880 sets and checking them is fast.
                
                # Remove old violations involving (u,v)
                # We need to check if they are still violations
                # It's easier to just re-evaluate the status of all 5-sets containing (u,v)
                
                # Optimization: Precompute which 5-sets contain which edges?
                # Too much memory/time.
                # Just iterate all_5sets? No, that's 850k.
                # We need the list of 5-sets containing u and v.
                
                # Let's generate them on the fly?
                # u and v are in the 5-set. The other 3 are from the remaining 40 vertices.
                remaining = [x for x in range(n_target) if x != u and x != v]
                for combo3 in itertools.combinations(remaining, 3):
                    full_combo = tuple(sorted((u, v) + tuple(combo3)))
                    # Check if full_combo is a violation
                    if is_bad_5set(full_combo, adj):
                        violations.add(full_combo)
                    else:
                        violations.discard(full_combo)
            
        return best_result
