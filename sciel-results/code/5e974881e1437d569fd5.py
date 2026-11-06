"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import sys
    from random import Random
    
    sys.setrecursionlimit(10000)
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def generate_graph(n, p, seed):
        """Generate random graph, return adjacency as list of sets."""
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch(adj, n):
        """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot: vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(frozenset(), set(range(n)), set())
        return cliques
    
    def greedy_transversal(cliques, n):
        """Find a transversal (hitting set) using greedy set cover approach."""
        if not cliques:
            return set()
        
        # For each vertex, count how many cliques it hits
        vertex_clique_count = [0] * n
        for clique in cliques:
            for v in clique:
                vertex_clique_count[v] += 1
        
        selected = set()
        remaining = set(cliques)
        
        while remaining:
            # Find vertex that hits the most remaining cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in selected:
                    continue
                count = sum(1 for c in remaining if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1 or best_count <= 0:
                break
            
            selected.add(best_v)
            remaining = {c for c in remaining if best_v not in c}
        
        return selected
    
    def local_search_improve(transversal, cliques, n, max_iters=500):
        """Try to improve the transversal via local search."""
        if not cliques:
            return transversal
        
        selected = set(transversal)
        
        # Check if valid
        def is_valid(sel):
            return all(any(v in sel for v in c) for c in cliques)
        
        if not is_valid(selected):
            return selected
        
        for _ in range(max_iters):
            improved = False
            
            # Try removing vertices
            for v in list(selected):
                new_sel = selected - {v}
                if is_valid(new_sel):
                    selected = new_sel
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for v_remove in list(selected):
                for v_add in range(n):
                    if v_add in selected:
                        continue
                    new_sel = (selected - {v_remove}) | {v_add}
                    if is_valid(new_sel):
                        # Only accept if it doesn't increase size (same size swap)
                        # Actually we want same or smaller, but since we removed one and added one, size is same
                        # This is for potential future improvements, but for now just keep
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    def solve_instance(n, p, seed):
        """Solve the clique transversal problem for one instance."""
        adj = generate_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return set()
        
        # Greedy initial solution
        transversal = greedy_transversal(cliques, n)
        
        # Local search improvement
        if time.time() < deadline - 5:
            transversal = local_search_improve(transversal, cliques, n)
        
        return transversal
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() >= deadline:
            # Fallback: return all vertices that appear in any edge
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            out.append(sorted(touched))
        else:
            transversal = solve_instance(n, p, seed)
            out.append(sorted(transversal))
    
    return out
# EVOLVE-BLOCK-END
