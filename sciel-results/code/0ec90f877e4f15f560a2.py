import random


def construct():
        n = 15
        best_edges = []
        
        # Helper to check validity and return list of invalid edges
        def get_invalid_edges(edges):
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            deg = {i: len(adj[i]) for i in range(n)}
            
            invalid = []
            for u, v in edges:
                # Get degrees of neighbors of u (excluding v)
                degs_u = set()
                for x in adj[u]:
                    if x != v:
                        degs_u.add(deg[x])
                        
                # Get degrees of neighbors of v (excluding u)
                degs_v = set()
                for y in adj[v]:
                    if y != u:
                        degs_v.add(deg[y])
                        
                if degs_u & degs_v:
                    invalid.append((u, v))
            return invalid

        # Try multiple random initializations
        for seed in range(100):
            random.seed(seed)
            # Start with a dense graph, e.g., all edges
            current_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            
            # Repair loop
            while True:
                invalids = get_invalid_edges(current_edges)
                if not invalids:
                    break
                # Remove a random invalid edge
                # To improve quality, maybe remove the one that appears in most conflicts?
                # For simplicity and speed, random removal is okay, but let's try removing one
                # that is "most central" to conflicts.
                # Actually, just removing any is fine for now, randomization helps.
                edge_to_remove = random.choice(invalids)
                current_edges.remove(edge_to_remove)
            
            if len(current_edges) > len(best_edges):
                best_edges = current_edges[:]
                
        # Convert to list of lists
        return [list(e) for e in best_edges]
