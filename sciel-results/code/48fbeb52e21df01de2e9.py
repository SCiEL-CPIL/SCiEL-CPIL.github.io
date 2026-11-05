import random
from collections import defaultdict
import numpy as np

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        if P:
            u_candidates = list(P) + list(X)
            if not u_candidates:
                return
            max_deg = -1
            pivot = None
            for u in u_candidates:
                deg = len(adj[u].intersection(P))
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            for v in list(P):
                if v in adj[pivot]:
                    continue
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)
        else:
             pass

    def find_maximal_cliques(n, adj):
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        return cliques

    def solve_hitting_set_exact(n, cliques):
        if not cliques:
            return []
        
        # Filter to cliques of size >= 2
        relevant_cliques = [c for c in cliques if len(c) >= 2]
        if not relevant_cliques:
            return []
        
        # Map cliques to indices
        m = len(relevant_cliques)
        # vertex_clique_mask[v] is a bitmask of which cliques contain v
        vertex_clique_mask = [0] * n
        for i, c in enumerate(relevant_cliques):
            mask_bit = 1 << i
            for v in c:
                vertex_clique_mask[v] |= mask_bit
        
        # Store clique sizes for heuristic
        clique_sizes = [len(c) for c in relevant_cliques]
        clique_vertices = [list(c) for c in relevant_cliques]
        
        best_size = float('inf')
        best_solution = None
        
        # To speed up, we can use a recursive search with memoization?
        # State is `uncovered_mask` (int).
        # However, the number of states might be large.
        # But n is small (<=40), m (number of maximal cliques) is usually small for sparse graphs?
        # For random graphs with p=0.2-0.3, number of maximal cliques can be significant but manageable.
        
        # We will use a simple DFS with pruning.
        # We can also use a greedy solution as an upper bound (initial best_size).
        
        def greedy_upper_bound():
            remaining = list(relevant_cliques)
            T = set()
            while remaining:
                counts = defaultdict(int)
                for c in remaining:
                    for v in c:
                        counts[v] += 1
                best_v = max(counts, key=counts.get)
                T.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            return len(T), T

        init_size, init_sol = greedy_upper_bound()
        best_size = init_size
        best_solution = init_sol

        # Recursive search
        # To avoid recursion depth issues or slow performance, we limit search or rely on n being small.
        # For n=40, exact hitting set can be hard.
        # However, the structure of "maximal cliques" is specific.
        
        # Let's try a branch and bound that picks the smallest uncovered clique.
        
        def dfs(uncovered_mask, current_depth, current_set):
            nonlocal best_size, best_solution
            
            if uncovered_mask == 0:
                if current_depth < best_size:
                    best_size = current_depth
                    best_solution = set(current_set)
                return
            
            if current_depth >= best_size:
                return
            
            # Find the uncovered clique with the smallest size
            # We need to iterate through bits in uncovered_mask
            min_clique_idx = -1
            min_clique_size = float('inf')
            
            # Iterate through set bits
            temp_mask = uncovered_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                if clique_sizes[idx] < min_clique_size:
                    min_clique_size = clique_sizes[idx]
                    min_clique_idx = idx
                    if min_clique_size == 2: # Can't get smaller than 2 (since we filtered >=2)
                        break
                temp_mask ^= lsb
            
            if min_clique_idx == -1:
                return
            
            # Branch on vertices in this clique
            clique_verts = clique_vertices[min_clique_idx]
            
            # Heuristic: order vertices by how many other uncovered cliques they cover
            # This is expensive to compute every time.
            # Simple order is fine for small n.
            
            for v in clique_verts:
                # If we pick v, we cover all cliques that contain v
                new_uncovered_mask = uncovered_mask & ~vertex_clique_mask[v]
                
                # Pruning: if new_uncovered_mask is same as uncovered_mask, v covers nothing new? 
                # No, v is in min_clique_idx, so it covers at least that one.
                
                # Additional pruning: 
                # If there is an uncovered clique C' that is a subset of the cliques covered by v? No.
                
                dfs(new_uncovered_mask, current_depth + 1, current_set + [v])
                
        # Initial uncovered mask is all bits set
        initial_mask = (1 << m) - 1
        dfs(initial_mask, 0, [])
        
        return sorted(best_solution)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges, adj = generate_graph(n, p, seed)
        cliques = find_maximal_cliques(n, adj)
        
        # Use the exact solver
        transversal = solve_hitting_set_exact(n, cliques)
        
        # Fallback if exact solver fails or times out (unlikely for n=40 with good pruning, but safe)
        # We can add a timeout check, but Python standard library doesn't have easy timeout for functions.
        # Given n=40, the number of maximal cliques is the bottleneck.
        # For random graphs, maximal cliques are usually small.
        
        results.append(transversal)
        
    return results
