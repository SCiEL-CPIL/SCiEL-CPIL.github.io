import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # Leave 5s margin for safety
    
    def is_c4_free(n, adj):
        """
        Check if the graph defined by adj (list of sets) is C4-free.
        adj is a list of sets of neighbors.
        """
        for u in range(n):
            # Only check pairs (u, v) where v > u to avoid double checking
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if len(common) >= 2:
                        return False
        return True

    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph C4-free.
        This is true if u and v have no common neighbors.
        """
        if v in adj[u]:
            return False
        common = adj[u] & adj[v]
        return len(common) == 0

    def has_independent_set_11(adj):
        """
        Check if there is an independent set of size 11.
        Uses a branch and bound approach.
        Returns True if found, False otherwise (or if time runs out, we assume False to be safe? 
        No, if we can't prove it's < 11, we should assume it might be >= 11 and be conservative.
        However, for the purpose of finding a *valid* graph, we need to be sure.
        If the search is too slow, we might incorrectly reject a good graph.
        But if we accept a bad graph, we score 0.
        So we should be conservative: if we can't prove alpha < 11, we treat it as alpha >= 11.
        """
        n = len(adj)
        if n < 11:
            return False
        
        # Precompute degrees for ordering
        # We want to pick vertices with high degree first to prune faster?
        # Or low degree?
        # Standard: pick vertex with max degree in the remaining subgraph.
        
        best_found = 0
        # We just need to find ONE independent set of size 11.
        
        def backtrack(remaining, chosen):
            nonlocal best_found
            if len(chosen) >= 11:
                return True
            if not remaining:
                return False
            # Pruning: if len(chosen) + len(remaining) < 11, return False
            if len(chosen) + len(remaining) < 11:
                return False
            
            # Heuristic: pick a vertex with high degree in the induced subgraph on 'remaining'
            # to remove many vertices from consideration.
            # Actually, for independent set, we pick a vertex v, then we can either include v (and remove its neighbors) or exclude v.
            # To find a large independent set, it's often better to branch on a vertex with high degree.
            
            # Let's just iterate through remaining vertices.
            # For efficiency, we can use a simple greedy upper bound.
            
            # Pick the first vertex in remaining (or one with max degree)
            # Let's pick the vertex with the maximum degree within 'remaining'
            max_deg = -1
            v_max = -1
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    v_max = v
            
            if v_max == -1:
                return False
            
            # Branch 1: Include v_max
            # New remaining: remaining - {v_max} - N(v_max)
            new_remaining_1 = [u for u in remaining if u != v_max and u not in adj[v_max]]
            if backtrack(new_remaining_1, chosen + [v_max]):
                return True
            
            # Branch 2: Exclude v_max
            new_remaining_2 = [u for u in remaining if u != v_max]
            if backtrack(new_remaining_2, chosen):
                return True
            
            return False

        # Start with all vertices
        initial_remaining = list(range(n))
        # To speed up, we can order vertices by degree globally first
        # But the dynamic degree is better.
        
        # Simple check first: if sum of degrees is too low? No.
        
        return backtrack(initial_remaining, [])

    def try_construct(n_target, seed):
        random.seed(seed)
        # Start with Clebsch graph (16 vertices) or just start empty and build?
        # Starting with Clebsch is good because it has small alpha.
        # Clebsch graph edges:
        # Vertices 0-15.
        # It is the folded 5-cube.
        # Let's just start with a random C4-free graph or a known small one.
        # For simplicity and robustness, let's start with 10 triangles (n=30) and try to extend?
        # Or start with Clebsch (n=16) and extend to n_target.
        
        # Let's start with Clebsch.
        n = 16
        adj = [set() for _ in range(n)]
        # Clebsch graph construction:
        # Vertices are binary strings of length 5 with even parity? No, that's 16 vertices.
        # Actually, Clebsch graph is the folded 5-cube.
        # Two vertices are adjacent if they differ in 1 or 4 positions.
        for i in range(16):
            for j in range(i+1, 16):
                # Count bit differences
                diff = bin(i ^ j).count('1')
                if diff == 1 or diff == 4:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Verify Clebsch is C4-free
        if not is_c4_free(n, adj):
            # Fallback to triangles if Clebsch construction is wrong
            n = 30
            adj = [set() for _ in range(n)]
            for t in range(10):
                a, b, c = 3*t, 3*t+1, 3*t+2
                adj[a].add(b); adj[b].add(a)
                adj[a].add(c); adj[c].add(a)
                adj[b].add(c); adj[c].add(b)
        
        # Extend to n_target
        while n < n_target:
            # Find a set of vertices in current graph that are pairwise at distance >= 3
            # This set will be the neighborhood of the new vertex.
            # Greedy approach:
            # Start with empty set S.
            # Iterate through vertices in random order.
            # If v can be added to S (i.e., no u in S has common neighbor with v), add v.
            
            S = []
            candidates = list(range(n))
            random.shuffle(candidates)
            
            for v in candidates:
                can_add = True
                for u in S:
                    if adj[u] & adj[v]:
                        can_add = False
                        break
                if can_add:
                    S.append(v)
            
            # Add new vertex n
            new_v = n
            adj.append(set())
            for u in S:
                adj[new_v].add(u)
                adj[u].add(new_v)
            n += 1
            
            # Try to add more edges to existing vertices to reduce alpha
            # This is a local improvement step
            # Check all non-edges (u, v) where u < v
            # If can_add_edge(adj, u, v), add it.
            # Do this a few times or until no more edges can be added.
            improved = True
            while improved:
                improved = False
                for u in range(n):
                    for v in range(u+1, n):
                        if v not in adj[u] and can_add_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            improved = True
                
                # Check time
                if time.time() > deadline - 20:
                    break

        # Verify C4-free
        if not is_c4_free(n, adj):
            return None

        # Check independence number
        # We need alpha < 11.
        # has_independent_set_11 returns True if alpha >= 11.
        # So we want it to return False.
        
        # To save time, we only do the exact check if n is large enough to matter
        # or if we are close to the deadline.
        # Actually, we must check.
        
        if has_independent_set_11(adj):
            return None
        
        return n, adj

    best_n = 30
    best_adj = None
    # Initialize with 10 triangles
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        if best_adj is None:
            best_adj = [set() for _ in range(30)]
        best_adj[a].add(b); best_adj[b].add(a)
        best_adj[a].add(c); best_adj[c].add(a)
        best_adj[b].add(c); best_adj[c].add(b)
    
    # Try to construct larger graphs
    # We try n_target = 39, 38, 37... down to 31
    for n_target in range(40, 30, -1):
        if time.time() > deadline - 30:
            break
        found = False
        for seed in range(100):
            if time.time() > deadline - 20:
                break
            res = try_construct(n_target, seed)
            if res:
                n, adj = res
                if n > best_n:
                    best_n = n
                    best_adj = adj
                found = True
                # If we found a valid graph for this n_target, we can stop trying this n_target
                # and move to the next larger one? No, we iterate downwards.
                # So if we found n=39, we are done.
                break
        if found and best_n == n_target:
            break

    # Convert best_adj to edge list
    edges = []
    for u in range(best_n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Constructed graph with n={res['n']}")
