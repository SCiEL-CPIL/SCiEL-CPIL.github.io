import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def greedy_partition(edges, n, rng, strategy="maxdeg"):
        if not edges:
            return []
        
        remaining = set(edges)
        cliques = []
        
        while remaining:
            if time.time() > DEADLINE:
                break
            
            # Build adjacency
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            if not adj:
                break
            
            # Pick starting vertex
            if strategy == "maxdeg":
                start = max(adj.keys(), key=lambda x: len(adj[x]))
            else:
                start = rng.choice(list(adj.keys()))
            
            # Greedily build clique
            clique = [start]
            candidates = set(adj[start])
            
            while candidates:
                # Score candidates
                scored = []
                for c in candidates:
                    score = sum(1 for m in clique if m in adj.get(c, set()))
                    # Tie-break by degree
                    deg = len(adj.get(c, set()))
                    scored.append((score, deg, c))
                scored.sort(reverse=True)
                
                best_c = scored[0][2]
                clique.append(best_c)
                
                # Update candidates
                new_candidates = set()
                for c in candidates:
                    if c != best_c:
                        ok = True
                        for m in clique:
                            if m not in adj.get(c, set()):
                                ok = False
                                break
                        if ok:
                            new_candidates.add(c)
                candidates = new_candidates
            
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove edges
                to_remove = set()
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        u, v = clique[i], clique[j]
                        e = (min(u,v), max(u,v))
                        if e in remaining:
                            to_remove.add(e)
                remaining -= to_remove
            else:
                e = next(iter(remaining))
                cliques.append([e[0], e[1]])
                remaining.remove(e)
        
        return cliques
    
    def verify_partition(cliques, edges, n):
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u,v), max(u,v))
                    if e not in edges:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edges
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        num_edges = len(edges)
        
        best_cliques = None
        best_count = num_edges
        
        # Try multiple strategies and seeds
        for trial in range(100):
            if time.time() > DEADLINE:
                break
            rng = random.Random(trial)
            strategy = "maxdeg" if trial % 2 == 0 else "random"
            cliques = greedy_partition(edges, n, rng, strategy)
            
            if not cliques:
                continue
            
            if verify_partition(cliques, edges, n):
                count = len(cliques)
                if count < best_count:
                    best_count = count
                    best_cliques = cliques
        
        if best_cliques is None:
            best_cliques = [[list(e) for e in edges]]
        
        results.append(best_cliques)
    
    return results
