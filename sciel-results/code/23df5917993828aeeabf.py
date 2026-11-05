"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from functools import lru_cache
from collections import defaultdict

def construct():
    """
    Return a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
    satisfying 1/a = 1/b + 1/c. Maximize cardinality.
    
    Approach: Branch and Bound on the Hypergraph MIS.
    1. Precompute forbidden triples.
    2. Define a conflict graph G_tri where u~v if they share a triple.
    3. B&B:
       - State: S (selected), U (undecided).
       - Bound: 
         - Construct graph H on U where edges are:
           a) u~v if exists w in U s.t. {u,v,w} is a triple.
           b) u~v if exists s in S s.t. {s,u,v} is a triple.
         - Find a large Clique C in H using greedy heuristic.
         - Upper Bound = |S| + |U| - |C| + 1.
       - Branch on vertex with max degree in H.
    """
    N = 180
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples
    forbidden = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted((a, b, c)))
                    forbidden.add(triple)
    
    # Precompute pairs for each vertex
    # pairs_for[v] = list of (u, w) such that {v, u, w} is a forbidden triple
    pairs_for = defaultdict(list)
    for (a, b, c) in forbidden:
        pairs_for[a].append((b, c))
        pairs_for[b].append((a, c))
        pairs_for[c].append((a, b))
        
    # Precompute triples fully contained in a set? Hard.
    # Instead, for the bound, we need to build H(U) dynamically.
    # H(U) edges:
    # 1. From S: for each s in S, for each (u, w) in pairs_for[s], if u in U and w in U, add edge (u, w).
    # 2. From U: for each triple (u, w, z) in forbidden, if u,w,z all in U, add edges (u,w), (w,z), (u,z).
    
    # To make this fast, we can precompute the list of triples.
    triples_list = list(forbidden)
    
    # Greedy Clique Finder on a graph represented by adjacency sets or bitsets.
    # For simplicity in B&B, we'll represent U as a set of ints.
    # H is built on the fly.
    
    def greedy_clique_size(vertices, adj):
        """
        vertices: set of vertices
        adj: dict mapping vertex to set of neighbors in the graph
        Returns size of a clique found by greedy heuristic.
        """
        if not vertices:
            return 0
        # Sort by degree in the subgraph
        # Degree of v in subgraph = len(adj[v] & vertices)
        sorted_v = sorted(vertices, key=lambda v: len(adj[v] & vertices), reverse=True)
        clique = set()
        candidates = set(vertices)
        
        # Simple greedy: pick highest degree, add to clique, remove non-neighbors
        # This is a heuristic for Max Clique, which is a lower bound on Max Clique.
        # We want a LARGE clique to get a TIGHT upper bound on MIS.
        # So finding a large clique is good.
        
        # A better greedy:
        # 1. Pick v with max degree.
        # 2. Add v to clique.
        # 3. candidates = candidates \ {v}
        # 4. candidates = candidates \ (candidates \ adj[v])  => candidates = candidates & adj[v]
        # 5. Repeat until candidates empty.
        
        while candidates:
            # Find v in candidates with max degree in candidates
            best_v = None
            max_deg = -1
            for v in candidates:
                deg = len(adj[v] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v is None:
                break
            clique.add(best_v)
            candidates.discard(best_v)
            # Intersect candidates with neighbors of best_v
            # adj[best_v] is the set of all neighbors in the full graph H
            # We need neighbors within the current candidates
            candidates = candidates & adj[best_v]
            
        return len(clique)

    # B&B
    best_set = set()
    best_size = 0
    
    # Initial Greedy to get a good incumbent
    # Use the logic from the previous program to get a strong start
    def greedy_initial():
        S_local = set()
        # Try descending
        for x in range(N, 0, -1):
            can_add = True
            for (u, v) in pairs_for[x]:
                if u in S_local and v in S_local:
                    can_add = False
                    break
            if can_add:
                S_local.add(x)
        return S_local
        
    inc_set = greedy_initial()
    best_size = len(inc_set)
    best_set = inc_set
    
    # Also try ascending
    S_asc = set()
    for x in range(1, N + 1):
        can_add = True
        for (u, v) in pairs_for[x]:
            if u in S_asc and v in S_asc:
                can_add = False
                break
        if can_add:
            S_asc.add(x)
    if len(S_asc) > best_size:
        best_size = len(S_asc)
        best_set = S_asc

    # B&B
    start_time = time.time()
    timeout = 55.0  # Leave 5s for cleanup
    
    # State: S (set), U (set)
    # We use a stack for DFS
    stack = [(set(best_set), set(numbers))] 
    # Actually, we should start from empty S and all U, but we have an incumbent.
    # We can start B&B from the incumbent? No, B&B explores the space.
    # We can use the incumbent to prune.
    
    # To speed up, we can order vertices.
    # Let's just run B&B from scratch but with strong pruning.
    # If it's too slow, we return the incumbent.
    
    # Optimization: If U is small, we can solve exactly or use better heuristics.
    
    def solve(S, U):
        nonlocal best_set, best_size
        
        if time.time() - start_time > timeout:
            return
        
        if not U:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
            return
        
        # Calculate Upper Bound
        # Build H(U)
        # H is a graph on U.
        # Edges:
        # 1. From S: for s in S, for (u,w) in pairs_for[s], if u in U and w in U, edge (u,w).
        # 2. From U: for (u,w,z) in triples_list, if u,w,z in U, edges (u,w), (w,z), (u,z).
        
        # To make this efficient, we can precompute adjacency for the "internal" part?
        # No, S changes.
        
        # Let's build adj for H(U)
        adj = {v: set() for v in U}
        
        # Part 1: Edges from S
        for s in S:
            for (u, w) in pairs_for[s]:
                if u in U and w in U:
                    adj[u].add(w)
                    adj[w].add(u)
                    
        # Part 2: Edges from U (internal triples)
        # This is expensive if done naively for every node.
        # But triples_list is small (~1000).
        for (u, w, z) in triples_list:
            if u in U and w in U and z in U:
                adj[u].add(w); adj[w].add(u)
                adj[w].add(z); adj[z].add(w)
                adj[u].add(z); adj[z].add(u)
                
        # Find Clique in H(U)
        clique_size = greedy_clique_size(U, adj)
        
        # Upper Bound: |S| + |U| - clique_size + 1
        ub = len(S) + len(U) - clique_size + 1
        
        if ub <= best_size:
            return
            
        # Branching
        # Pick vertex in U with max degree in H(U)
        # Or max conflict in original hypergraph?
        # Degree in H(U) is a good proxy.
        
        max_deg = -1
        branch_v = None
        for v in U:
            deg = len(adj[v])
            if deg > max_deg:
                max_deg = deg
                branch_v = v
                
        if branch_v is None:
            # No edges, so we can add all remaining
            new_S = S | U
            if len(new_S) > best_size:
                best_size = len(new_S)
                best_set = new_S
            return
            
        # Branch 1: Exclude branch_v
        new_U_excl = U - {branch_v}
        solve(S, new_U_excl)
        
        # Branch 2: Include branch_v
        # If we include branch_v, we must remove any u in U such that {branch_v, u, w} is a triple for some w in U?
        # No, we just add branch_v to S.
        # The new S is S | {branch_v}.
        # The new U is U - {branch_v}.
        # The constraints on the new U will be handled in the next call's bound calculation.
        
        new_S_incl = S | {branch_v}
        new_U_incl = U - {branch_v}
        solve(new_S_incl, new_U_incl)

    # Run B&B
    # Start with empty S and all U?
    # The incumbent is already good.
    # B&B might take too long.
    # Let's limit the search or just use the incumbent if B&B is slow.
    
    # To prevent stack overflow or infinite loop, we rely on the timeout.
    # However, Python recursion limit might be an issue if depth > 1000.
    # Depth is at most 180. So it's fine.
    
    # We start with S=empty, U=all.
    # But we have a good incumbent.
    # Let's just run it.
    
    solve(set(), set(numbers))
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
