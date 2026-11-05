"""Combined approach: multi-restart greedy set cover with 2-for-1 local search and random tie-breaking."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                u = max(P, key=lambda v: len(adj[v] & P))
            elif X:
                u = max(X, key=lambda v: len(adj[v] & X))
            else:
                return
            
            # Branch on vertices in P \\ N(u)
            for v in list(P - adj[u]):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = instance(n, p, seed)
        
        # Find all maximal cliques of size >= 2
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        C = len(cliques)
        
        # Precompute incidence: vertex_cliques[v] = set of clique indices containing v
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)
        
        # Also precompute for each clique, the set of vertices in it (for quick membership checks)
        clique_vertices = [set(c) for c in cliques]
        
        best_set = None
        best_size = float('inf')
        
        # Number of restarts based on time budget
        # n=30: ~50 cliques, n=35: ~40 cliques, n=40: ~30 cliques
        # More restarts for smaller instances
        remaining_time = deadline - time.time()
        base_restarts = 100
        if n <= 32:
            num_restarts = min(base_restarts * 2, int(remaining_time * 20))
        elif n <= 37:
            num_restarts = min(base_restarts, int(remaining_time * 15))
        else:
            num_restarts = min(base_restarts // 2, int(remaining_time * 10))
        
        num_restarts = max(10, num_restarts)
        rng_solver = random.Random(seed + 12345)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Greedy Phase with random tie-breaking
            hit_cliques = set()
            current_set = set()
            
            while len(hit_cliques) < C:
                best_v = -1
                max_hits = -1
                candidates = []
                
                for v in range(n):
                    if v in current_set:
                        continue
                    hits = 0
                    for ci in vertex_cliques[v]:
                        if ci not in hit_cliques:
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)
                        
                if max_hits == 0:
                    # Should not happen if there are unhit cliques
                    break
                    
                # Random tie-breaking
                v = rng_solver.choice(candidates)
                current_set.add(v)
                for ci in vertex_cliques[v]:
                    hit_cliques.add(ci)
                    
            # Local Search Phase
            # 1. Removal pass: remove redundant vertices
            improved = True
            while improved:
                improved = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline - 5.0:
                        break
                    can_remove = True
                    for ci in vertex_cliques[v]:
                        # Check if clique ci is hit by any other vertex in current_set
                        c = clique_vertices[ci]
                        found_other = False
                        for u in c:
                            if u != v and u in current_set:
                                found_other = True
                                break
                        if not found_other:
                            can_remove = False
                            break
                    if can_remove:
                        current_set.remove(v)
                        improved = True
            
            # 2. 2-for-1 move: replace two vertices with one that covers all their uniquely-covered cliques
            # This is a structural improvement that greedy alone misses
            improved2 = True
            while improved2 and time.time() < deadline - 5.0:
                improved2 = False
                vs = list(current_set)
                for i in range(len(vs)):
                    for j in range(i + 1, len(vs)):
                        v1 = vs[i]
                        v2 = vs[j]
                        
                        # Find cliques that are ONLY covered by v1 or v2 (i.e., not by any other vertex in current_set)
                        # Actually, we want to find if there exists a vertex w not in current_set such that:
                        # - w covers all cliques that are currently only covered by v1 or v2
                        # - Removing v1 and v2 and adding w still covers all cliques
                        # - This reduces size by 1
                        
                        # For each clique, determine which vertices in current_set cover it
                        # A clique is "uniquely covered" by v1 if v1 is in it and no other vertex in current_set is in it
                        # We need: for all cliques c, if c is covered only by {v1, v2} (i.e., exactly one of v1,v2 is in c, 
                        # and no other vertex in current_set is in c), then w must be in c.
                        
                        # Simpler approach: try each candidate w not in current_set
                        # Check if (current_set - {v1, v2}) ∪ {w} is a valid cover
                        # And len((current_set - {v1, v2}) ∪ {w}) < len(current_set)
                        # i.e., w not in current_set and w not in {v1, v2} (already true)
                        
                        # But we need this to be efficient. Let's precompute which cliques each vertex uniquely covers.
                        # Actually, let's just try a limited set of candidates.
                        
                        # Find cliques that would become uncovered if we remove v1 and v2
                        # A clique c is uncovered if no vertex in (current_set - {v1, v2}) is in c
                        at_risk_cliques = []
                        for ci in range(C):
                            c = clique_vertices[ci]
                            # Check if any vertex in current_set - {v1, v2} is in c
                            covered = False
                            for u in current_set:
                                if u == v1 or u == v2:
                                    continue
                                if u in c:
                                    covered = True
                                    break
                            if not covered:
                                at_risk_cliques.append(ci)
                        
                        if not at_risk_cliques:
                            # Removing v1 and v2 doesn't uncover any cliques, so just remove one?
                            # This should have been caught by removal pass
                            continue
                        
                        # Now find a vertex w not in current_set that covers all at_risk_cliques
                        # w must be in the intersection of all at_risk_cliques
                        candidate_set = set(range(n)) - current_set
                        for w in candidate_set:
                            covers_all = True
                            for ci in at_risk_cliques:
                                if w not in clique_vertices[ci]:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Check that the new set is valid
                                new_set = (current_set - {v1, v2}) | {w}
                                # Verify validity
                                valid = True
                                for ci in range(C):
                                    c = clique_vertices[ci]
                                    if not any(u in c for u in new_set):
                                        valid = False
                                        break
                                if valid and len(new_set) < len(current_set):
                                    current_set = new_set
                                    improved2 = True
                                    break
                        if improved2:
                            break
                    if improved2:
                        break
            
            # 3. Single swap: try to replace one vertex with another to enable further removals
            # This helps escape local optima
            improved3 = True
            while improved3 and time.time() < deadline - 5.0:
                improved3 = False
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v_remove in vs:
                    if time.time() > deadline - 5.0:
                        break
                    for v_add in range(n):
                        if v_add in current_set:
                            continue
                        new_set = (current_set - {v_remove}) | {v_add}
                        # Check validity
                        valid = True
                        for ci in range(C):
                            c = clique_vertices[ci]
                            if not any(u in c for u in new_set):
                                valid = False
                                break
                        if valid:
                            # After swap, try to remove redundant vertices
                            current_set = new_set
                            # Removal pass
                            imp = True
                            while imp:
                                imp = False
                                vs2 = list(current_set)
                                rng_solver.shuffle(vs2)
                                for v in vs2:
                                    if time.time() > deadline - 5.0:
                                        break
                                    can_remove = True
                                    for ci in vertex_cliques[v]:
                                        c = clique_vertices[ci]
                                        found_other = False
                                        for u in c:
                                            if u != v and u in current_set:
                                                found_other = True
                                                break
                                        if not found_other:
                                            can_remove = False
                                            break
                                    if can_remove:
                                        current_set.remove(v)
                                        imp = True
                            improved3 = True
                            break
                    if improved3:
                        break
            
            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
                # Early stopping if we hit a good solution
                # Lower bound: at least 1 if there are any cliques
                # Heuristic: if best_size <= C // 10, we're likely near optimal
                if best_size <= max(1, C // 15):
                    break
        
        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 5.0:
            # Not enough time; return empty lists as fallback (will be invalid but avoid crash)
            results.append([])
            continue
        result = solve_instance(n, p, seed)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
