import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Fallback: Andrásfai graph And(9) on 26 vertices
    def get_andrasfai():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    # Function to check if a graph (defined by edges or adjacency) has an independent set of size k
    # We will work with the complement graph to find a clique of size k
    # G is triangle-free. We want alpha(G) < 10.
    # This is equivalent to: Complement(G) has no clique of size 10.
    
    def has_clique_size_k(adj_bitsets, n, k):
        """
        Returns True if there is a clique of size k in the graph defined by adj_bitsets.
        adj_bitsets[i] is a bitset (int) where bit j is 1 if i-j is an edge.
        """
        # Initial candidates: all vertices
        candidates = (1 << n) - 1
        clique = []
        
        # Greedy coloring for bounding
        # We use a simple recursive B&B
        
        def expand(clique_size, cand):
            if clique_size == k:
                return True
            if cand == 0:
                return False
            
            # Heuristic: bound by number of candidates
            # Count bits in cand
            if clique_size + bin(cand).count('1') < k:
                return False
                
            # Greedy coloring of the subgraph induced by cand to get an upper bound on clique size
            # This is a simplified version. For n=39, we can just iterate.
            
            # Order candidates: try those with high degree in the remaining graph first?
            # Or just iterate.
            
            # To speed up, we can use a coloring bound.
            # Let's implement a simple greedy coloring bound.
            # Colors are assigned to vertices in cand.
            # The max color index + 1 is an upper bound on the clique size in cand.
            
            uncolored = cand
            colors = {}
            max_color = 0
            # Simple greedy coloring
            # We need the subgraph adjacency.
            # adj_bitsets[i] & cand gives neighbors of i in cand.
            
            # This inner loop might be slow if done naively for every node.
            # For n=39, let's try a simpler bound first: 
            # If the graph is dense, cliques are likely.
            # If the graph is sparse, no cliques.
            
            # Let's use the "Tomita" style coloring bound.
            # It requires ordering vertices.
            
            # Given the small n, we can use a simpler recursive step without complex coloring 
            # if we prune well.
            
            # Pruning: if max possible clique < k, return False.
            
            # Let's iterate through vertices in cand
            # We pick a vertex v, add to clique, recurse with neighbors of v in cand.
            
            # To improve, we order vertices by degree in cand (descending).
            # But computing degrees is O(n^2).
            
            # Let's just pick the first available vertex? No, bad for B&B.
            # Let's pick a vertex with high degree.
            
            # Get list of vertices in cand
            verts = [i for i in range(n) if (cand >> i) & 1]
            
            # Sort by degree in cand (descending)
            # deg(i) = popcount(adj_bitsets[i] & cand)
            def deg(i):
                return bin(adj_bitsets[i] & cand).count('1')
            
            verts.sort(key=deg, reverse=True)
            
            for v in verts:
                if not ((cand >> v) & 1):
                    continue
                # Try adding v
                new_clique_size = clique_size + 1
                if new_clique_size >= k:
                    return True
                new_cand = cand & adj_bitsets[v]
                # Remove v from new_cand? 
                # adj_bitsets[v] does not contain v (simple graph).
                # So new_cand is neighbors of v in cand.
                
                if expand(new_clique_size, new_cand):
                    return True
                
                # Remove v from cand for subsequent iterations
                cand &= ~(1 << v)
                
                # Check bound again
                if clique_size + bin(cand).count('1') < k:
                    return False
            
            return False

        return expand(0, candidates)

    def is_valid_graph(n, edges):
        # Check triangle-free
        # Build adjacency bitsets
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check triangles: for each edge (u,v), check if adj[u] & adj[v] != 0
        # This is O(E).
        for u, v in edges:
            if adj[u] & adj[v]:
                return False
        
        # Check alpha < 10
        # Build complement adjacency bitsets
        comp_adj = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = full_mask & ~(adj[i] | (1 << i))
            
        if has_clique_size_k(comp_adj, n, 10):
            return False
            
        return True

    # Search for a valid graph on n=39
    target_n = 39
    
    # Generate random sum-free sets in Z_39
    # Sum-free means x+y != z mod 39 for x,y,z in S.
    # Actually, for Cayley graph to be triangle-free, S must be sum-free:
    # If x, y in S, then x+y not in S.
    # Also -S = S for undirected graph.
    
    found = False
    best_result = get_andrasfai()
    
    while time.time() < deadline:
        # Try to construct a Cayley graph
        S = set()
        # Randomly add elements
        elements = list(range(1, target_n // 2 + 1))
        random.shuffle(elements)
        
        # We need S to be symmetric: if x in S, then -x mod n in S.
        # Pairs: (1, 38), (2, 37), ..., (19, 20).
        # Let's pick pairs.
        pairs = [(i, (target_n - i) % target_n) for i in range(1, target_n // 2 + 1)]
        random.shuffle(pairs)
        
        valid_sum_free = True
        current_S = set()
        
        for x, y in pairs:
            # Try adding x and y
            # Check if adding x breaks sum-free property
            # New sums: x+x, x+y, y+x, y+y
            # And interactions with existing S
            # This is complex. Let's just build S randomly and check at the end.
            pass
            
        # Alternative: Build S by picking random elements and checking sum-free property incrementally.
        S = set()
        candidates = list(range(1, target_n // 2 + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if we can add x and -x
            # Sums to check:
            # 1. x+x
            # 2. x+(-x) = 0 (not in S usually)
            # 3. For all z in S: x+z, -x+z, x-z, -x-z
            # If any of these are in S, we cannot add x.
            
            can_add = True
            for z in list(S):
                # Check x+z
                if (x + z) % target_n in S or (x + z) % target_n == x or (x + z) % target_n == (target_n - x) % target_n:
                    # If x+z is x, then z=0 (not in S).
                    # If x+z is -x, then 2x=0.
                    pass
                # Actually, just check if (x+z)%n is in S or is x or is -x
                # And (x-z)%n etc.
                
                # Simpler: Just build S, then verify triangle-free at the end.
                # It's faster to generate many candidates and verify.
                pass
            
            # Let's just add x and -x if they don't immediately create a triangle with existing S.
            # Triangle in Cayley graph exists if there exist a,b,c in S such that a+b=c or a+b=-c etc.
            # Specifically, if x, y in S, then x+y not in S and x+y not in -S.
            
            # Let's just pick a random subset of pairs and test the whole graph.
            pass

        # Let's use a different strategy: 
        # Pick a random set of pairs.
        num_pairs = random.randint(10, 15) # Degree approx 2*num_pairs
        selected_pairs = random.sample(pairs, num_pairs)
        S = set()
        for x, y in selected_pairs:
            S.add(x)
            S.add(y)
            
        # Build edges
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                diff = (j - i) % target_n
                if diff in S:
                    edges.append([i, j])
                    
        # Verify
        if is_valid_graph(target_n, edges):
            best_result = {'n': target_n, 'edges': edges}
            found = True
            break # Found a valid one, return immediately
            
    return best_result
