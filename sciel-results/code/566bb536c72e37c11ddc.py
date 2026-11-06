import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(edges, n):
        # edges is a set of tuples (u, v) with u < v
        remaining_edges = set(edges)
        cliques = []
        
        # Adjacency list for quick neighbor checks
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We will maintain the remaining graph structure implicitly or explicitly.
        # For efficiency, let's just work with the set of remaining edges.
        # But finding cliques is easier with adjacency.
        # Since we remove edges, we need to update adjacency or just check membership.
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return current best plus edges as size-2 cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                return cliques
                
            # Find a vertex with max degree in remaining graph
            # To avoid O(E) scan every time, we can just pick a random edge's endpoint
            # or scan all vertices. N is small (32, 64).
            # Let's find the vertex with the most remaining edges.
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            # Pick vertex with max degree
            v_start = max(deg, key=deg.get)
            
            # Greedy clique construction starting from v_start
            current_clique = [v_start]
            # Candidates are neighbors of v_start that are connected to all in current_clique
            # Initially, candidates are neighbors of v_start
            candidates = set(adj[v_start])
            # Filter candidates to only those that have an edge to v_start in remaining_edges?
            # adj contains ALL edges. We need to ensure edges are in remaining_edges.
            
            # Let's refine:
            # We want to build a clique C such that all edges in C are in remaining_edges.
            # Start with C = {v_start}.
            # Candidates for next vertex u:
            # 1. u must be connected to all v in C via edges in remaining_edges.
            # 2. u not in C.
            
            C = {v_start}
            
            while True:
                # Find valid candidates
                # A candidate u is valid if for all v in C, (min(u,v), max(u,v)) is in remaining_edges
                # And u != v
                valid_candidates = []
                # To speed this up, we can intersect neighbor sets, but checking edge existence is O(1)
                
                # Who are the potential neighbors?
                # They must be connected to v_start.
                # Let's just iterate through all neighbors of v_start?
                # Or better, iterate through all vertices? N is small.
                
                best_candidate = None
                best_score = -1
                
                for u in range(n):
                    if u in C:
                        continue
                    # Check if u is connected to all in C
                    is_connected = True
                    for v in C:
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        # Score: number of edges we would cover if we add u
                        # This is the number of vertices w in C such that (u,w) is in remaining_edges
                        # Since we checked connectivity, all (u,w) for w in C are in remaining_edges.
                        # So the number of edges added is len(C).
                        # This is constant for all valid candidates!
                        # So we just pick any valid candidate?
                        # To get a larger clique, we should pick the one that allows more future extensions.
                        # Heuristic: pick u with highest degree in the subgraph induced by valid candidates?
                        # Or simply highest degree in remaining graph?
                        
                        # Let's use degree in remaining graph as a tie-breaker/heuristic
                        score = deg[u]
                        if score > best_score:
                            best_score = score
                            best_candidate = u
                
                if best_candidate is None:
                    break
                
                C.add(best_candidate)
            
            if len(C) >= 2:
                cliques.append(sorted(C))
                # Remove all edges within C from remaining_edges
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = sorted((C[i], C[j]))
                        remaining_edges.discard((u, v))
            
            # Update degrees? 
            # Recalculating degrees from scratch is O(E). 
            # E is at most ~600 for n=32, ~2000 for n=64. 
            # Number of iterations is bounded by number of cliques.
            # This should be fast enough.
            
        return cliques

    # Instance 1
    edges1 = generate_graph(32, 0.6, 601)
    partition1 = greedy_partition(edges1, 32)
    
    # Instance 2
    edges2 = generate_graph(64, 0.55, 602)
    partition2 = greedy_partition(edges2, 64)
    
    return [partition1, partition2]
