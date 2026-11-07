import random

N = 15

def construct():
    # Vertices 0..14
    # Strategy: Construct a graph with many edges while ensuring no simple path s-u-v-t (length 3)
    # has d(s) == d(t).
    
    # Observation: If we use a "threshold" structure or a "Ferrers" type graph, we can control degrees.
    # However, the constraint is local to paths of length 3.
    
    # Let's try a specific construction:
    # Partition vertices into two sets A and B?
    # Or use a "ladder" or "chain" structure with added chords?
    
    # A known good structure for avoiding equal-degree endpoints on length-3 paths:
    # If we ensure that any two vertices connected by a path of length 3 have different degrees,
    # it's hard.
    
    # Alternative: Random Greedy with Backtracking/Repair.
    # 1. Start with empty graph.
    # 2. Generate all possible edges.
    # 3. Shuffle.
    # 4. Try adding edges one by one. If adding an edge (u,v) creates a violation, skip it.
    #    A violation is created if there exists a path s-u-v-t or s-v-u-t such that d(s)==d(t) 
    #    (degrees in the *current* graph + the new edge? No, degrees in the FULL candidate graph).
    #    The contract says: "Reject if any simple three-edge path has endpoints of equal degree in the FULL candidate graph."
    #    This implies we must check the validity of the ENTIRE graph at the end, or ensure that adding an edge
    #    doesn't break the property relative to the final degrees.
    #    Since degrees change as we add edges, a simple greedy add is tricky because adding an edge
    #    changes degrees, which might invalidate previously "safe" paths.
    
    # Better Strategy:
    # 1. Start with a dense random graph.
    # 2. Check for violations.
    # 3. Remove edges involved in violations.
    # 4. Repeat until no violations.
    # 5. Try to add back random edges if valid.
    
    # Let's implement the "Remove Violations" approach.
    
    def check_validity(edges_set):
        # Build adjacency list and degree counts
        adj = {i: set() for i in range(N)}
        deg = [0] * N
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all paths s-u-v-t where (s,u), (u,v), (v,t) are edges
        # s != t
        # and s, t are not adjacent to u or v in a way that creates a cycle? 
        # "simple three-edge path" means s, u, v, t are distinct.
        for u in range(N):
            for v in adj[u]:
                # Path s-u-v-t
                # s is in adj[u], s != v
                # t is in adj[v], t != u
                # s != t
                # s, u, v, t distinct
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        # Check if s-u-v-t is simple
                        # s, u, v, t are distinct by construction above?
                        # s != v, t != u, t != s.
                        # Also need s != u (true, no loops), v != t (true, no loops).
                        # Is it possible s == v? No, checked.
                        # Is it possible t == u? No, checked.
                        # Is it possible s == t? No, checked.
                        # Are s, u, v, t all distinct?
                        # Yes.
                        if deg[s] == deg[t]:
                            return False, (s, u, v, t)
        return True, None

    def random_graph(num_edges):
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def repair(edges_set):
        # Remove edges causing violations
        # Simple approach: while violation exists, remove one edge from the violation path.
        max_iter = 1000
        for _ in range(max_iter):
            valid, viol = check_validity(edges_set)
            if valid:
                break
            if viol is None:
                break
            s, u, v, t = viol
            # Remove one of the edges (s,u), (u,v), (v,t)
            # Randomly choose which one to remove to escape local minima?
            # Or remove (u,v) as it's the middle edge.
            # Let's try removing the middle edge (u,v) first.
            edges_set.discard((min(u,v), max(u,v)))
            # If that doesn't work in next iter, it will try another.
            # Actually, removing any edge on the path breaks that specific path.
            # But it might create new paths.
        return edges_set

    # 1. Start with a dense random graph
    # Max possible edges is 15*14/2 = 105.
    # Let's start with 80 edges.
    best_edges = set()
    best_len = 0
    
    # Run multiple trials with different densities
    for trial in range(5):
        # Start with a relatively dense graph
        initial_density = 0.6 + 0.1 * trial
        num_edges = int(N * (N-1) / 2 * initial_density)
        edges_set = random_graph(num_edges)
        
        # Repair
        edges_set = repair(edges_set)
        
        # Try to add back edges (Greedy Add)
        all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if (u, v) in edges_set:
                continue
            # Try adding
            edges_set.add((u, v))
            valid, _ = check_validity(edges_set)
            if not valid:
                edges_set.discard((u, v))
            else:
                # Keep it
                if len(edges_set) > best_len:
                    best_len = len(edges_set)
                    best_edges = set(edges_set)
            best_len = len(edges_set)
            best_edges = set(edges_set)

    # Final check on best_edges
    valid, _ = check_validity(best_edges)
    if not valid:
        # Fallback: Empty graph is always valid
        return []

    # Convert to list of lists of ints
    result = [[int(u), int(v)] for u, v in best_edges]
    return result

# Ensure deterministic output if needed? 
# The task says "Maximize edge count". Randomness is allowed as long as it finishes in time.
# To make it reproducible for testing, we might set a seed, but usually these tasks allow randomness.
# However, if the evaluator runs it multiple times, it should be robust.
# Let's set a seed to ensure a good result is found reliably? 
# No, better to let it explore. But if I want to guarantee a score > 1.0 (which is 14 edges),
# I need to be confident the repair finds > 14 edges.
# A star has 14 edges. My algorithm starts with ~60-90 edges and removes conflicts.
# It is highly likely to find a graph with > 20 edges.

# Let's add a seed for reproducibility in case of flakiness?
# random.seed(42) 
# Actually, let's not seed, to allow better exploration if the environment allows.
# But wait, if the result varies, the score might vary.
# The task says "Maximize edge count". 
# I will add a final "polish" step: try to add edges in a fixed order (0,1), (0,2)... to ensure stability?
# No, the random repair is better.

# One optimization: The `check_validity` function is O(E^2) or worse.
# N=15 is small. Max edges 105.
# Paths: For each edge (u,v), check neighbors.
# 105 edges. Each edge checks deg(u)*deg(v) pairs.
# deg ~ 10. 100 pairs per edge. 105*100 = 10k checks.
# Very fast.

# Let's refine the repair to be more aggressive.
# Instead of just removing one edge, maybe we can identify a set of edges to remove?
# For N=15, simple removal is fine.

# Re-evaluating the "Star" baseline: 14 edges.
# If my algorithm fails and returns empty, score is 0.
# I must ensure it returns a valid graph.
# The empty graph is valid.
# The star is valid.
# My algorithm starts dense and removes. It will never return invalid.
# It will likely return > 14 edges.

# Let's double check the path definition.
# "simple three-edge path s-u-v-t"
# Edges: (s,u), (u,v), (v,t).
# Vertices s, u, v, t are distinct.
# My check:
# for u in range(N):
#   for v in adj[u]:
#     for s in adj[u]: # s != v
#       for t in adj[v]: # t != u, t != s
#         if deg[s] == deg[t]: fail
# This covers s-u-v-t.
# Does it cover s-v-u-t?
# Yes, because the loop iterates over all u, v.
# When u and v are swapped in the outer loops, it covers the other direction.
# Specifically, path s-v-u-t corresponds to u'=v, v'=u, s'=s, t'=t?
# No.
# Path is s-v-u-t. Edges (s,v), (v,u), (u,t).
# In my loop:
# Outer u becomes v. Outer v becomes u.
# s is neighbor of v (which is adj[u] in code? No, adj[u] in code is neighbors of current u).
# Let's trace carefully.
# Code:
# for u in range(N):  # Let's call this node A
#   for v in adj[u]:  # Let's call this node B. Edge (A,B)
#     for s in adj[u]: # s is neighbor of A. Edge (s,A). s != B.
#       for t in adj[v]: # t is neighbor of B. Edge (B,t). t != A, t != s.
#         # Path s-A-B-t.
#         # This covers paths where the middle edge is (A,B).
#         # The endpoints are s and t.
#         # This covers ALL simple paths of length 3, because any such path has a unique middle edge.
#         # And for each middle edge (A,B), the endpoints are s (neighbor of A) and t (neighbor of B).
#         # The condition s!=t and s!=B, t!=A ensures simplicity.
#         # What if s==A? No, no loops.
#         # What if s is connected to B?
#         # If s is connected to B, then s-B is an edge.
#         # The path s-A-B-t is still a simple path IF s != t and s != B (true) and t != A (true) and s != A (true) and t != B (true).
#         # Does the existence of chord (s,B) make it not "simple"?
#         # "Simple path" usually means vertices are distinct.
#         # s, A, B, t are distinct. So it is a simple path.
#         # The chord (s,B) makes it an induced path? No, the contract says "paths need not be induced".
#         # So even if (s,B) exists, the path s-A-B-t is a valid path to check.
#         # My code checks it.
#         # Correct.

# So the logic is sound.

# One more check: Types.
# "integers must be Python int ... sequences must be list or tuple"
# My return is list of lists of ints. Correct.

# Let's add a hard-coded fallback if random fails (unlikely).
# If best_len == 0, return star.
# Star: [(0,1), (0,2)...(0,14)]
# This is valid.

# I'll include the star as a fallback candidate in the comparison.

def construct():
    N = 15
    
    def check_validity(edges_set):
        adj = {i: set() for i in range(N)}
        deg = [0] * N
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        for u in range(N):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False, (s, u, v, t)
        return True, None

    def random_graph(num_edges):
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def repair(edges_set):
        max_iter = 200
        for _ in range(max_iter):
            valid, viol = check_validity(edges_set)
            if valid:
                break
            if viol is None:
                break
            s, u, v, t = viol
            # Remove middle edge (u,v)
            edges_set.discard((min(u,v), max(u,v)))
        return edges_set

    best_edges = set()
    best_len = 0
    
    # Fallback: Star
    star_edges = {(0, i) for i in range(1, N)}
    best_edges = star_edges
    best_len = len(star_edges)
    
    # Try random dense graphs
    for trial in range(10):
        initial_density = 0.5 + 0.05 * trial
        num_edges = int(N * (N-1) / 2 * initial_density)
        edges_set = random_graph(num_edges)
        edges_set = repair(edges_set)
        
        # Greedy add
        all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if (u, v) in edges_set:
                continue
            edges_set.add((u, v))
            valid, _ = check_validity(edges_set)
            if not valid:
                edges_set.discard((u, v))
        
        if len(edges_set) > best_len:
            best_len = len(edges_set)
            best_edges = set(edges_set)
            
    # Final validation
    valid, _ = check_validity(best_edges)
    if not valid:
        return [[0, i] for i in range(1, N)]
        
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-START
# (The code above is the implementation. I will place it here as the main block)
# EVOLVE-BLOCK-END
