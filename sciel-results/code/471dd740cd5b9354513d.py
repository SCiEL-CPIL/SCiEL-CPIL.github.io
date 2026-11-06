import random
import time
import sys

def _has_c5(adj, n):
    # Check for any 5-cycle (C5) in the graph
    # A 5-cycle is a sequence v0-v1-v2-v3-v4-v0 where all edges exist
    # and all vertices are distinct.
    # We can enumerate all 5-cycles by fixing v0=0 and checking all permutations of 4 other vertices
    # But that's O(n^4 * 5!) which might be too slow for n=33-35.
    # Better: enumerate all 5-cycles by finding all length-5 paths that close.
    # Actually, for n<=35, we can use a smarter enumeration.
    # A 5-cycle has 5 vertices. We can enumerate all 5-subsets and check if they contain a C5.
    # Number of 5-subsets of 35 is C(35,5) = 324632. For each, check if it contains a C5.
    # A 5-vertex graph contains a C5 iff it is a 5-cycle itself (since it has exactly 5 vertices).
    # So we need to check if the induced subgraph on those 5 vertices is a 5-cycle.
    # A 5-cycle has exactly 5 edges. So we can check: does the induced subgraph have exactly 5 edges and is it connected?
    # Actually, a 5-vertex graph with 5 edges could also be a tree with an extra edge (not a cycle) or a cycle plus a chord.
    # Wait, a C5 has exactly 5 edges. If the induced subgraph has more than 5 edges, it's not a C5.
    # If it has exactly 5 edges, it could be a C5 or a path of length 4 plus one extra edge (which would create a triangle or 4-cycle, not a 5-cycle).
    # Actually, the only 5-vertex, 5-edge connected graph that is a cycle is C5.
    # So: for each 5-subset, count edges. If exactly 5, check if it's a cycle (each vertex has degree 2 in the induced subgraph).
    
    from itertools import combinations
    
    # Precompute edge count for each 5-subset
    # This might be slow for n=35. Let's estimate: 324632 subsets * (check edges + check degrees)
    # Checking edges: for each 5-subset, there are C(5,2)=10 possible edges. We check if each exists.
    # That's 324632 * 10 = 3.2M operations. Should be fine in Python within a few seconds.
    
    for subset in combinations(range(n), 5):
        # Count edges in induced subgraph
        edge_count = 0
        for i in range(5):
            for j in range(i+1, 5):
                u = subset[i]
                v = subset[j]
                if adj[u][v]:
                    edge_count += 1
        if edge_count != 5:
            continue
        # Check if each vertex has degree 2 in the induced subgraph
        is_c5 = True
        for i in range(5):
            u = subset[i]
            deg = 0
            for j in range(5):
                if i == j:
                    continue
                v = subset[j]
                if adj[u][v]:
                    deg += 1
            if deg != 2:
                is_c5 = False
                break
        if is_c5:
            return True
    return False

def _has_independent_set_9(adj, n):
    # Check if there's an independent set of size 9
    # Independent set: no edges between any pair of vertices in the set
    # We can use a branch and bound or just enumerate all 9-subsets
    # C(35,9) = 70607460 which is too many to enumerate directly.
    # We need a smarter approach.
    # Use a recursive backtracking with pruning.
    # Or use the fact that if there's an independent set of size 9, then the complement graph has a clique of size 9.
    # But checking for K9 is also hard.
    # Let's use a simple recursive approach with pruning.
    
    # Sort vertices by degree (ascending) to help pruning
    # Actually, for independent set, we want vertices with low degree in the original graph (high degree in complement)
    # But let's just do a simple DFS with memoization or pruning.
    
    # Alternative: use a greedy approach to find a lower bound, then use branch and bound.
    # For n<=35, a simple branch and bound might work.
    
    # Let's try a different approach: use the complement graph and look for a clique of size 9.
    # Complement adjacency: comp_adj[u][v] = not adj[u][v] (for u != v)
    # We need to find a clique of size 9 in the complement graph.
    
    # Use a recursive clique finder with pruning.
    # At each step, we have a current clique and a set of candidates.
    # Pruning: if len(current) + len(candidates) < 9, prune.
    
    comp_adj = [[False]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            comp_adj[u][v] = not adj[u][v]
            comp_adj[v][u] = not adj[v][u]
    
    # Precompute candidate sets for each vertex
    # For a vertex u, its candidates are neighbors in the complement graph
    
    def find_clique(current, candidates, target):
        if len(current) >= target:
            return True
        if len(current) + len(candidates) < target:
            return False
        if not candidates:
            return False
        
        # Pick a vertex from candidates
        # Heuristic: pick the vertex with the highest degree within candidates
        best_idx = 0
        best_deg = -1
        for i, v in enumerate(candidates):
            deg = 0
            for w in candidates:
                if w != v and comp_adj[v][w]:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_idx = i
        v = candidates[best_idx]
        
        # Try adding v
        new_candidates = [w for w in candidates if w != v and comp_adj[v][w]]
        if find_clique(current + [v], new_candidates, target):
            return True
        
        # Try not adding v (but this is expensive; we can just return False here for simplicity,
        # but that might miss solutions. Let's do proper backtracking.)
        # Actually, the above is not correct. We need to try all vertices.
        # Let me rewrite this properly.
        
        return False
    
    # The above function is incomplete. Let me write a proper one.
    
    def find_clique2(current_size, candidates, target):
        if current_size >= target:
            return True
        if current_size + len(candidates) < target:
            return False
        if not candidates:
            return False
        
        for i in range(len(candidates)):
            v = candidates[i]
            new_candidates = [w for j, w in enumerate(candidates) if j > i and comp_adj[v][w]]
            if find_clique2(current_size + 1, new_candidates, target):
                return True
        return False
    
    # Start with all vertices as candidates
    all_vertices = list(range(n))
    return find_clique2(0, all_vertices, 9)

def _is_valid(adj, n):
    if _has_c5(adj, n):
        return False
    if _has_independent_set_9(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Start with the known 32-vertex solution
    n = 32
    block_size = 4
    num_blocks = 8
    edges = []
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        for a in range(block_size):
            for b in range(a + 1, block_size):
                edges.append([start + a, start + b])
    
    best_n = n
    best_edges = edges
    
    # Try to find a 33-vertex solution using local search
    # We'll try to add a 33rd vertex and adjust edges
    
    # For n=33, we need to check if there's a valid graph
    # Let's try a random search: start with the 32-vertex graph, add a 33rd vertex,
    # and randomly add/remove edges to try to satisfy the constraints.
    
    # Actually, let's try a more systematic approach.
    # We'll use simulated annealing or random restarts.
    
    # First, let's try to extend the 32-vertex graph by adding a 33rd vertex.
    # The 32-vertex graph is 8 disjoint K4's.
    # We need to add edges from the new vertex to the existing 32 vertices.
    # The new vertex must not create a C5 and must not create an independent set of size 9.
    
    # Let's try all possible neighborhoods for the new vertex.
    # There are 2^32 possible neighborhoods, which is too many.
    # We need to use constraints to prune.
    
    # Constraint 1: No C5.
    # Constraint 2: No independent set of size 9.
    
    # For the independent set constraint: the new vertex v33 can be part of an independent set.
    # If v33 is in an independent set, then its non-neighbors in the 32 vertices must not contain an independent set of size 8.
    # In the 32-vertex graph (8 K4's), the maximum independent set size is 8 (one from each block).
    # So if v33 has a non-neighbor in each block, then those 8 non-neighbors plus v33 form an independent set of size 9.
    # Therefore, v33 must be adjacent to all vertices in at least one block.
    
    # For the C5 constraint: we need to check that adding v33 doesn't create a C5.
    # A C5 involving v33 would be v33 - a - b - c - d - v33 where a,b,c,d are in the 32 vertices.
    # Since the 32-vertex graph is a union of K4's, any path of length 3 in the 32-vertex graph must be within a single block (since there are no edges between blocks).
    # So a, b, c, d must be in the same block.
    # But a block is a K4, so it has 4 vertices. A path of length 3 in a K4 would use all 4 vertices.
    # So the C5 would be v33 - a - b - c - d - v33 where {a,b,c,d} is a block.
    # For this to be a C5, we need edges v33-a, a-b, b-c, c-d, d-v33.
    # Since a,b,c,d form a K4, all edges among them exist.
    # So we need v33 to be adjacent to a and d, and a-b-c-d to be a path (which it is in a K4).
    # But wait, in a K4, any 4 vertices form a clique, so there are many paths.
    # The key is: if v33 is adjacent to at least 2 vertices in a block, can we form a C5?
    # Let's say v33 is adjacent to a and b in a block.
    # Then we need a path of length 3 from a to b in the block that doesn't use v33.
    # In a K4, there are paths of length 2 from a to b (a-c-b) and paths of length 3 (a-c-d-b).
    # A C5 would be v33-a-c-d-b-v33 if v33 is adjacent to a and b.
    # So if v33 is adjacent to 2 or more vertices in a block, we can form a C5.
    # Therefore, v33 can be adjacent to at most 1 vertex in each block to avoid C5.
    
    # But we also need v33 to be adjacent to all vertices in at least one block to avoid independent set of size 9.
    # This is a contradiction: v33 can be adjacent to at most 1 vertex per block, but must be adjacent to all 4 vertices in at least one block.
    
    # So we cannot extend the 32-vertex graph by just adding a 33rd vertex without changing the structure.
    
    # Therefore, we need to modify the 32-vertex graph as well.
    # Let's try a random search for n=33.
    
    # We'll use a random graph with random edges and check if it's valid.
    # If not, we'll modify it.
    
    # This is a complex search space. Let's try a few random restarts.
    
    random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try n=33
        n_try = 33
        # Generate a random graph
        # We'll start with a random graph and try to optimize it.
        
        # For simplicity, let's just try a few random graphs and see if any is valid.
        # This is unlikely to work, but let's try.
        
        # Actually, let's try a more structured approach.
        # We'll use the 32-vertex graph as a base and try to add a 33rd vertex with a modified neighborhood.
        
        # But we've shown that's impossible without modifying the base graph.
        
        # Let's try a completely random graph for n=33.
        adj_try = [[False]*n_try for _ in range(n_try)]
        for u in range(n_try):
            for v in range(u+1, n_try):
                if random.random() < 0.5:
                    adj_try[u][v] = True
                    adj_try[v][u] = True
        
        if _is_valid(adj_try, n_try):
            best_n = n_try
            best_edges = []
            for u in range(n_try):
                for v in range(u+1, n_try):
                    if adj_try[u][v]:
                        best_edges.append([u, v])
    
    # If we didn't find a 33-vertex solution, return the 32-vertex one
    if best_n == 32:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': best_n, 'edges': best_edges}
