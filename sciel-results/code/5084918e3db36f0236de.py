import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 128
    elements = list(range(1, n + 1))

    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Build conflict graph: edge between x,y if they form a bad triple with some z
    # Bad triple: lcm(a,b) = lcm(a,c) = lcm(b,c) for distinct a,b,c
    # Two elements x,y are in conflict if there exists z such that (x,y,z) is a bad triple
    # For independent set: we want to pick elements with NO bad triple among them
    # So conflict edge: x~y if for some z, (x,y,z) is a bad triple
    
    # Actually, the constraint is: no three distinct elements form a bad triple.
    # This is a 3-uniform hypergraph independence problem.
    # Convert to graph: x and y are adjacent if they participate in a bad triple together.
    # Then an independent set in this graph has no pair that's in a bad triple... 
    # Wait, that's not quite right. If x~y~z all pairwise conflict, the triple is bad.
    # But we need: no triple (a,b,c) where ALL THREE pairwise LCMS are equal.
    
    # Let me reconsider. The condition is: for no three distinct a,b,c do we have
    # lcm(a,b) = lcm(a,c) = lcm(b,c).
    # 
    # Key insight: if lcm(a,b) = lcm(a,c) = lcm(b,c) = L, then a|L, b|L, c|L,
    # and L is the lcm of any pair. This means L is the lcm of {a,b,c}.
    # 
    # For the graph-based MIS approach: define conflict(x,y) = True if there exists
    # some z such that (x,y,z) is a bad triple. Then a set S is valid iff for every
    # pair (x,y) in S, there is no z in S that makes (x,y,z) a bad triple.
    # This is NOT the same as an independent set in the conflict graph!
    # Because in a conflict graph independent set, we just need no edge between any two,
    # but here we need that no THREE elements form a bad triple.
    #
    # However, if we define the graph where x~y means "x and y are in some bad triple",
    # then an independent set in this graph means no two elements are in a common bad triple,
    # which is STRONGER than needed (we allow pairs to be in a bad triple as long as the
    # third element isn't also present).
    #
    # Better approach: Use the 3-uniform hypergraph directly.
    # But for branch-and-bound, let's use a different formulation.
    #
    # Actually, let me reconsider the structure. If lcm(a,b)=lcm(a,c)=lcm(b,c)=L,
    # then a,b,c all divide L, and for each pair, their lcm is exactly L.
    # This means: for the triple to be "bad", we need that no proper subset's lcm is L.
    # Specifically, lcm(a,b)=L means a and b together generate L (neither alone is L 
    # unless the other divides it... actually if a=L then lcm(a,b)=L requires b|L).
    
    # Let me just precompute all bad triples and use a hypergraph MIS approach.
    # For branch-and-bound on hypergraphs, we can use:
    # - Include a vertex: remove it, and remove all hyperedges that don't contain it
    #   (those are now "satisfied" as broken)
    # - Exclude a vertex: remove it and all hyperedges containing it
    #
    # But this is complex. Let me use a simpler but effective approach:
    # The conflict graph approach (edge if pair is in a bad triple) gives a SUPERSET of
    # constraints. An independent set in this graph is valid (no bad triple), but may not
    # be maximum. However, it's a good lower bound, and we can use it as the initial solution
    # for a more careful search.
    #
    # Actually, for this specific problem, let me check: if (a,b,c) is a bad triple,
    # then any subset of size 2 from {a,b,c} is "conflicted". So the conflict graph
    # has triangle on {a,b,c}. An independent set avoids having all three, which is
    # exactly what we want! Because if we have an independent set, no two are connected,
    # meaning no pair is in a bad triple, so certainly no triple is bad.
    #
    # Wait, that's the converse. Independent set means NO edge between any two vertices
    # in the set. Edge means "in a bad triple together". So independent set means no pair
    # in the set is in a bad triple. This is sufficient for validity (no bad triple) but
    # not necessary (we could have pairs in bad triples as long as the third isn't present).
    #
    # So MIS on the conflict graph gives a valid but potentially suboptimal solution.
    # For the exact answer, I need to handle the 3-uniform constraint properly.
    #
    # Let me use a different strategy: 
    # 1. Compute the conflict graph (pair is in a bad triple)
    # 2. Use branch-and-bound MIS on this graph to get a good lower bound
    # 3. Then do a final verification and local improvement

    # Precompute bad triples
    bad_triples = []  # list of frozensets of size 3
    conflict_pairs = set()  # pairs that are in at least one bad triple
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a, b, c))
                    conflict_pairs.add((a, b))
                    conflict_pairs.add((a, c))
                    conflict_pairs.add((b, c))

    # Build adjacency for conflict graph
    adj = [set() for _ in range(n + 1)]
    for x, y in conflict_pairs:
        adj[x].add(y)
        adj[y].add(x)

    # Tomita's branch-and-bound for Maximum Independent Set
    # Uses greedy coloring bound
    
    best_size = 0
    best_set = set()
    
    # Precompute degree for ordering
    degrees = [len(adj[i]) for i in range(1, n + 1)]
    
    # Color the graph for upper bound
    # Greedy coloring: assign colors so adjacent vertices get different colors
    # Number of colors = upper bound on independent set size
    
    def greedy_color(vertices, adj):
        """Greedy coloring of subgraph induced by vertices. Returns list of color classes."""
        color_classes = []
        colored = set()
        remaining = set(vertices)
        
        while remaining:
            color_class = set()
            # Greedy: pick vertices that can share a color (no edges between them)
            candidates = list(remaining)
            for v in candidates:
                if v in color_class:
                    continue
                # Check if v conflicts with any in color_class
                if not any(u in color_class for u in adj[v]):
                    color_class.add(v)
            if not color_class:
                # Shouldn't happen, but safety
                color_class = {remaining.pop()}
            color_classes.append(color_class)
            remaining -= color_class
        
        return color_classes

    # Branch and bound
    # State: candidate set P (vertices that can still be added), current independent set I
    # Upper bound: |I| + max_coloring_bound(P)
    
    def branch_and_bound(P, I, depth=0):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        P_size = len(P)
        if P_size == 0:
            if len(I) > best_size:
                best_size = len(I)
                best_set = set(I)
            return
        
        # Upper bound via coloring
        color_classes = greedy_color(P, adj)
        upper_bound = len(I) + len(color_classes)
        
        if upper_bound <= best_size:
            return
        
        # Branch: pick a vertex from P
        # Order: pick vertex with highest degree in P for better pruning
        # Actually for independent set, we want to branch on vertices in P
        # Include or exclude
        
        # Select vertex to branch on: highest degree in induced subgraph P
        max_deg = -1
        branch_v = None
        for v in P:
            deg_in_P = len(adj[v] & P)
            if deg_in_P > max_deg:
                max_deg = deg_in_P
                branch_v = v
        
        if branch_v is None:
            branch_v = next(iter(P))
        
        # Option 1: Include branch_v
        # New I = I ∪ {branch_v}
        # New P = P ∩ (V \ N(branch_v) \ {branch_v})
        # i.e., P minus branch_v and its neighbors in P
        new_P_include = P - {branch_v} - (adj[branch_v] & P)
        new_I = I | {branch_v}
        
        # Prune: if |new_I| + len(new_P_include) <= best_size, skip
        if len(new_I) + len(new_P_include) > best_size:
            branch_and_bound(new_P_include, new_I, depth + 1)
        
        # Option 2: Exclude branch_v
        new_P_exclude = P - {branch_v}
        # I unchanged
        if len(I) + len(new_P_exclude) > best_size:
            branch_and_bound(new_P_exclude, I, depth + 1)

    # Initial solution: greedy independent set
    def greedy_independent_set():
        S = set()
        remaining = set(range(1, n + 1))
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            v = None
            for u in remaining:
                d = len(adj[u] & remaining)
                if d < min_deg:
                    min_deg = d
                    v = u
            S.add(v)
            remaining -= adj[v]
            remaining.discard(v)
        return S

    # Get initial lower bound
    init_set = greedy_independent_set()
    best_size = len(init_set)
    best_set = set(init_set)

    # Run branch and bound
    full_P = set(range(1, n + 1))
    branch_and_bound(full_P, set())

    # Final verification
    def is_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                for k in range(j + 1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if lcm(a, b) == lcm(a, c) == lcm(b, c):
                        return False
        return True

    # Verify and if invalid, fall back to greedy
    if not is_valid(best_set):
        best_set = init_set

    return list(best_set)
