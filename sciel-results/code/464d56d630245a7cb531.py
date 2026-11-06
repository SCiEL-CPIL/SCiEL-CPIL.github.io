import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def find_triangles(edges, adj, n):
        """Find all triangles in the graph."""
        triangles = []
        for u, v in edges:
            common = adj[u] & adj[v]
            for w in common:
                if w > v:  # avoid duplicates
                    triangles.append((u, v, w))
        return triangles
    
    def greedy_clique_partition(edges, n, adj, triangles, max_time):
        """Greedy: repeatedly find a large clique among remaining edges, prioritizing edge coverage."""
        remaining = set(edges)
        cliques = []
        
        while remaining:
            if time.time() > max_time:
                break
            
            # Strategy: find the clique that covers the most remaining edges
            # Seed from triangles (which are guaranteed cliques of size 3)
            # or from high-degree edges
            
            best_clique = None
            best_score = 0  # score = number of remaining edges covered
            
            # Try triangles as seeds
            for tri in triangles:
                u, v, w = tri
                # Check if all three edges are in remaining
                e1 = (min(u,v), max(u,v))
                e2 = (min(v,w), max(v,w))
                e3 = (min(u,w), max(u,w))
                if e1 in remaining and e2 in remaining and e3 in remaining:
                    clique = [u, v, w]
                    # Try to extend
                    candidates = adj[u] & adj[v] & adj[w]
                    candidates.discard(u)
                    candidates.discard(v)
                    candidates.discard(w)
                    
                    # Sort candidates by degree in remaining graph
                    cand_list = list(candidates)
                    cand_list.sort(key=lambda x: sum(1 for y in adj[x] if (min(x,y), max(x,y)) in remaining), reverse=True)
                    
                    for w2 in cand_list:
                        ok = True
                        for c in clique:
                            e = (min(c, w2), max(c, w2))
                            if e not in remaining:
                                ok = False
                                break
                        if ok:
                            clique.append(w2)
                    
                    # Score: count remaining edges covered by this clique
                    score = 0
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                            if e in remaining:
                                score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_clique = clique
            
            # If no triangle works, try individual edges
            if best_clique is None or best_score < 2:
                # Find edge with highest degree in remaining
                edge_list = list(remaining)
                random.shuffle(edge_list)
                best_edge = None
                best_edge_score = 0
                for e in edge_list[:50]:
                    u, v = e
                    # Count how many common neighbors have both edges in remaining
                    common = adj[u] & adj[v]
                    common.discard(u)
                    common.discard(v)
                    score = 1  # the edge itself
                    for w in common:
                        if (min(u,w), max(u,w)) in remaining and (min(v,w), max(v,w)) in remaining:
                            score += 2
                    if score > best_edge_score:
                        best_edge_score = score
                        best_edge = e
                
                if best_edge is not None:
                    u, v = best_edge
                    best_clique = [u, v]
                    best_score = best_edge_score
                else:
                    e = edge_list[0]
                    best_clique = list(e)
                    best_score = 1
            
            cliques.append(best_clique)
            
            # Remove all edges within best_clique
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                    remaining.discard(e)
        
        return cliques
    
    def improve_partition(cliques, edges, adj, n, max_time):
        """Try to merge cliques that share vertices to reduce total count."""
        if time.time() > max_time:
            return cliques
        
        # Build mapping from edge to clique index
        edge_to_clique = {}
        for idx, clique in enumerate(cliques):
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    edge_to_clique[e] = idx
        
        # Try to merge pairs of cliques
        changed = True
        while changed and time.time() < max_time:
            changed = False
            new_cliques = list(cliques)
            for i in range(len(new_cliques)):
                for j in range(i+1, len(new_cliques)):
                    if time.time() > max_time:
                        return cliques
                    ci = set(new_cliques[i])
                    cj = set(new_cliques[j])
                    # Check if union is a clique
                    merged = ci | cj
                    is_clique = True
                    for u in merged:
                        for v in merged:
                            if u < v and v not in adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique and len(merged) >= 2:
                        # Merge them
                        new_cliques[i] = list(merged)
                        new_cliques[j] = []
                        changed = True
                        break
                if changed:
                    break
            
            # Remove empty cliques
            new_cliques = [c for c in new_cliques if len(c) >= 2]
            if len(new_cliques) < len(cliques):
                cliques = new_cliques
        
        return cliques
    
    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)
        triangles = find_triangles(edges, adj, n)
        
        best_cliques = None
        best_count = float('inf')
        end_time = time.time() + time_budget
        
        # Try multiple random restarts with different random seeds
        rng = random.Random(seed + 1000)
        
        while time.time() < end_time:
            # Shuffle triangles for diversity
            shuffled_triangles = list(triangles)
            rng.shuffle(shuffled_triangles)
            
            cliques = greedy_clique_partition(edges, n, adj, shuffled_triangles, end_time - 1)
            
            # Verify completeness
            covered = set()
            for c in cliques:
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        e = (min(c[i], c[j]), max(c[i], c[j]))
                        covered.add(e)
            
            if covered == edges:
                # Try to improve by merging
                cliques = improve_partition(cliques, edges, adj, n, end_time - 1)
                
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
            
            if time.time() > end_time - 3:
                break
        
        if best_cliques is None:
            # Fallback: each edge as a clique
            best_cliques = [list(e) for e in edges]
        
        return best_cliques
    
    # Solve both instances
    t1 = time.time()
    budget1 = min(60, DEADLINE - (t1 - start_time) - 10)
    cliques1 = solve_instance(32, 0.6, 601, budget1)
    
    t2 = time.time()
    budget2 = min(60, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)
    
    return [cliques1, cliques2]
