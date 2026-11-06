def construct():
    import time
    import random
    from collections import defaultdict

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Map edge to boolean if covered
        uncovered = set(edges)
        
        # Adjacency list for quick lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        # Time limit guard
        start_time = time.time()
        timeout = 140 # Leave margin for 150s limit
        
        while uncovered:
            if time.time() - start_time > timeout:
                # Fallback: cover remaining edges individually
                for e in list(uncovered):
                    cliques.append(list(e))
                break
                
            # Pick vertex with max degree in uncovered subgraph
            deg = {v: 0 for v in range(n)}
            for u, v in uncovered:
                deg[u] += 1
                deg[v] += 1
                
            # Find max degree vertex
            max_deg = -1
            v = -1
            for u in range(n):
                if deg[u] > max_deg:
                    max_deg = deg[u]
                    v = u
                    
            if v == -1:
                break
                
            # Find a large clique containing v using uncovered edges
            # Candidates are neighbors of v connected via uncovered edges
            candidates = [u for u in adj[v] if (min(v, u), max(v, u)) in uncovered]
            
            # Greedily build clique
            clique = [v]
            # Sort candidates by degree in the induced subgraph of candidates? 
            # Or just degree in original graph? 
            # Let's sort by degree in the "uncovered" subgraph restricted to candidates
            # Actually, just sorting by original degree is a good heuristic for dense graphs
            
            # We need to ensure edges between clique members are uncovered.
            # Let's maintain a set of valid candidates.
            valid_cands = set(candidates)
            
            # To make it faster, we can just iterate.
            # Since graphs are small (n=32, 64), we can afford O(n^2) or O(n^3) per step.
            
            # Sort valid_cands by degree in original graph descending
            sorted_cands = sorted(valid_cands, key=lambda x: deg[x], reverse=True)
            
            for u in sorted_cands:
                # Check if u is connected to all in clique via uncovered edges
                can_add = True
                for w in clique:
                    edge = (min(u, w), max(u, w))
                    if edge not in uncovered:
                        can_add = False
                        break
                if can_add:
                    clique.append(u)
                    # Remove u from other potential candidates? 
                    # Actually, we just need to ensure future additions are valid.
                    # The check "edge not in uncovered" handles it.
                    # But we should probably remove u from consideration for other branches? 
                    # No, we are building ONE clique.
                    
            # Add clique to partition
            cliques.append(clique)
            
            # Remove all edges in this clique from uncovered
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, w = clique[i], clique[j]
                    edge = (min(u, w), max(u, w))
                    if edge in uncovered:
                        uncovered.remove(edge)
                        
        return cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from collections import defaultdict

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        uncovered = set(edges)
        cliques = []
        
        # Precompute adjacency for original graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        start_time = time.time()
        timeout = 145 # Safety margin
        
        while uncovered:
            if time.time() - start_time > timeout:
                # Fallback
                for e in list(uncovered):
                    cliques.append(list(e))
                break
                
            # Calculate degrees in uncovered subgraph
            deg = [0] * n
            for u, v in uncovered:
                deg[u] += 1
                deg[v] += 1
                
            # Find vertex with max degree
            v = max(range(n), key=lambda i: deg[i])
            
            if deg[v] == 0:
                # Should not happen if uncovered is not empty
                break

            # Find candidates for clique extension
            # Neighbors of v in original graph, such that edge (v, u) is uncovered
            candidates = [u for u in adj[v] if (min(v, u), max(v, u)) in uncovered]
            
            # Sort candidates by degree in uncovered subgraph (approximated by deg)
            candidates.sort(key=lambda x: deg[x], reverse=True)
            
            clique = [v]
            
            # Greedy expansion
            for u in candidates:
                # Check if u can be added to clique
                # u must be connected to all existing clique members via uncovered edges
                can_add = True
                for w in clique:
                    if w == u:
                        continue
                    edge = (min(u, w), max(u, w))
                    if edge not in uncovered:
                        can_add = False
                        break
                if can_add:
                    clique.append(u)
                    
            # Add clique
            cliques.append(clique)
            
            # Remove edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, w = clique[i], clique[j]
                    edge = (min(u, w), max(u, w))
                    if edge in uncovered:
                        uncovered.remove(edge)
                        
        return cliques

    return [solve_instance(32, 0.6, 601), solve_instance(64, 0.55, 602)]
# EVOLVE-BLOCK-END
