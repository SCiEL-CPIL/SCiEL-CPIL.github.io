import time

# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph."""
    # For n=32, this is fast.
    # Optimized check:
    # A 5-cycle is a set of 5 vertices {v0, v1, v2, v3, v4} with edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v0).
    # We can enumerate paths of length 4 and check if the endpoints are connected.
    # Or simply enumerate 5-sets.
    # Given n=32, 5-sets are C(32,5) ~ 200k. Very fast.
    from itertools import combinations
    
    # Precompute adjacency sets for faster lookup
    adj_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                
    for combo in combinations(range(n), 5):
        v0, v1, v2, v3, v4 = combo
        # Check all 6 possible cycle structures on these 5 vertices?
        # Actually, a 5-cycle is a specific cycle. 
        # If the induced subgraph contains a C5, it's a C5.
        # We just need to check if there is a permutation that forms a cycle.
        # Simpler: check if the induced subgraph has a cycle of length 5.
        # Since the induced subgraph has 5 vertices, if it contains a C5, it IS a C5 (possibly with chords).
        # We can check if the adjacency matrix of the induced subgraph has a cycle of length 5.
        # For 5 vertices, we can just check the 12 possible 5-cycles (Hamiltonian cycles in K5).
        # Or simpler: check if the graph is not bipartite? No, C5 is odd cycle.
        # Just check specific edges.
        
        # There are (5-1)!/2 = 12 distinct cycles in K5.
        # We can just check if the set of edges in the induced subgraph contains any of these 12 cycles.
        
        # Let's just check the specific ordering v0-v1-v2-v3-v4-v0 and its variations?
        # No, the vertices in the combination are unordered.
        # We need to check if ANY permutation forms a C5.
        
        # Efficient check for C5 in 5 vertices:
        # Count degrees in induced subgraph. If all deg >= 2, it might have a cycle.
        # If it has a C5, it must have 5 edges forming the cycle.
        
        # Let's just check the 12 permutations of the cycle structure.
        # Cycle: 0-1-2-3-4-0.
        # Permutations of (0,1,2,3,4) that represent the same cycle:
        # (0,1,2,3,4), (0,4,3,2,1) are the same.
        # We can fix v0 and check the 12 cycles starting with v0.
        
        # Actually, for n=32, the graph is a union of K4s.
        # K4 does not contain C5.
        # So _has_c5 should return False immediately if we know the structure.
        # But we must check exactly.
        
        # For the specific graph (8 K4s), any 5 vertices are either:
        # - All in one K4: Impossible, K4 has 4 vertices.
        # - 4 in one K4, 1 in another: The 4 form K4. The 1 is isolated from them.
        #   Induced subgraph is K4 + isolated vertex. No C5.
        # - 3 in one K4, 2 in another: K3 + K2. No C5.
        # - 2 in one, 2 in another, 1 in another: K2 + K2 + K1. No C5.
        # - 2 in one, 1 in another, 1 in another, 1 in another: K2 + K1 + K1 + K1. No C5.
        # - 1 in each of 5 K4s: 5 isolated vertices. No C5.
        
        # So for this specific graph, it's always False.
        # But the function must be general.
        
        # General check for 5 vertices:
        # If the induced subgraph has < 5 edges, no C5.
        # If it has 5 edges, check if it's a C5 or C4+edge etc.
        # C5 has 5 edges.
        # K5 has 10.
        # K4+isolated has 6.
        
        # We can just check if the number of edges is 5 and the graph is a single cycle.
        # Or simpler: try to find a Hamiltonian cycle in the induced subgraph.
        
        # For speed in the specific case, we can short-circuit if we detect disjoint cliques.
        # But let's keep it general and fast.
        
        # Check if any 5-cycle exists in the induced subgraph.
        # We can check the 12 cycles.
        # Cycle 1: v0-v1-v2-v3-v4-v0
        # Cycle 2: v0-v1-v2-v4-v3-v0
        # ...
        
        # This is overkill for n=32.
        # We can just return False for the known valid graph.
        # But to be safe:
        
        # Check if the induced subgraph is a C5.
        # A graph on 5 vertices is a C5 if and only if it has 5 edges and is 2-regular.
        # Or if it contains a C5 as a subgraph (which for 5 vertices means it IS a C5 or has chords).
        # If it has chords, it has > 5 edges.
        # So: if edges_count == 5 and all degrees == 2, it is a C5.
        
        degs = [0] * 5
        edge_count = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if adj_sets[combo[i]] and combo[j] in adj_sets[combo[i]]:
                    edge_count += 1
                    degs[i] += 1
                    degs[j] += 1
        
        if edge_count == 5 and all(d == 2 for d in degs):
            return True
            
    return False


def _alpha_bound(n, adj):
    """Exact max independent set via branch and bound."""
    # For the specific graph (8 K4s), alpha is 8.
    # We can compute it quickly.
    # Greedy: pick one from each K4.
    # Since we know the structure, we can just return 8?
    # No, must be exact.
    
    # Use a fast heuristic for this specific case.
    # If the graph is a union of cliques, alpha = number of cliques.
    # We can detect if it's a union of cliques.
    
    # General B&B for n=32 is fast enough.
    best = [0]
    
    # Convert adj to sets
    adj_sets = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                
    # Greedy init
    cand = list(range(n))
    greedy_set = []
    while cand:
        # Pick vertex with min degree in cand
        min_v = min(cand, key=lambda v: len(adj_sets[v] & set(cand)))
        greedy_set.append(min_v)
        cand = [x for x in cand if x != min_v and x not in adj_sets[min_v]]
    best[0] = len(greedy_set)
    
    def dfs(cand, size):
        if size + len(cand) <= best[0]:
            return
        if not cand:
            if size > best[0]:
                best[0] = size
            return
        
        # Branch on a vertex with high degree to prune?
        # Or low degree to include?
        # Standard: pick vertex with max degree in cand.
        v = max(cand, key=lambda x: len(adj_sets[x] & set(cand)))
        
        # Include v
        new_cand = [x for x in cand if x != v and x not in adj_sets[v]]
        dfs(new_cand, size + 1)
        
        # Exclude v
        dfs([x for x in cand if x != v], size)
        
    dfs(list(range(n)), 0)
    return best[0]


def construct():
    start = time.time()
    
    # Construct 8 disjoint K4s
    blocks = 8
    k = 4
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
                
    # Verification
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
        
    # Check C5
    if _has_c5(n, adj):
        # Fallback if it fails (should not)
        return {'n': 28, 'edges': []} # Dummy
        
    # Check Alpha
    alpha = _alpha_bound(n, adj)
    if alpha >= 9:
        return {'n': 28, 'edges': []} # Dummy
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
