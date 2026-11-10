"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 150s limit, aim for 110s to be safe.

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition_fast(n, edges, seed_heur):
        """
        Fast greedy clique partition.
        1. Calculate initial degrees.
        2. While edges remain:
           a. Pick vertex v with max remaining degree.
           b. Build a clique C starting with v by greedily adding neighbors that connect to all in C.
           c. Remove all edges within C from the graph and update degrees.
        """
        # Adjacency list for quick neighbor lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_deg = [len(adj[i]) for i in range(n)]
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        # Track which vertices are still "active" (have remaining degree > 0)
        active = [i for i in range(n) if remaining_deg[i] > 0]
        
        while active:
            # Find vertex with max remaining degree
            # To speed up, we can just pick the max from the active list.
            # Since n is small (<=64), a linear scan is fine.
            max_idx = -1
            max_deg = -1
            for u in active:
                if remaining_deg[u] > max_deg:
                    max_deg = remaining_deg[u]
                    max_idx = u
            
            if max_idx == -1:
                break
                
            v = max_idx
            
            # Build clique containing v
            C = [v]
            # Candidates must be neighbors of v that are active
            candidates = [u for u in adj[v] if remaining_deg[u] > 0]
            
            # Greedily add vertices to C
            # We want to add vertices that are connected to all current members of C
            while candidates:
                # Find candidate u that is connected to all vertices in C
                # And perhaps has high degree to cover more edges
                best_u = None
                best_score = -1
                for u in candidates:
                    # Check if u is connected to all in C
                    is_connected = True
                    for w in C:
                        if w not in adj[u]:
                            is_connected = False
                            break
                    if is_connected:
                        # Score: prefer higher remaining degree to cover more edges
                        score = remaining_deg[u]
                        if score > best_score:
                            best_score = score
                            best_u = u
                
                if best_u is None:
                    break
                    
                C.append(best_u)
                # Remove best_u from candidates
                candidates.remove(best_u)
                # New candidates must be neighbors of best_u AND connected to all in C (except best_u itself, but we check full C)
                # Actually, the next candidate must be connected to ALL vertices in the new C.
                # So we filter candidates to those connected to best_u.
                new_candidates = []
                for u in candidates:
                    if best_u in adj[u]:
                        # We also need to ensure u is connected to all previous C members.
                        # Since u was in 'candidates', it was connected to all previous C members.
                        # Now it must be connected to best_u.
                        new_candidates.append(u)
                candidates = new_candidates

            # Remove edges within C from remaining_edges and update degrees
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    e = (u, v2) if u < v2 else (v2, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        # Update degrees
                        remaining_deg[u] -= 1
                        remaining_deg[v2] -= 1
            
            cliques.append(C)
            
            # Update active list: remove vertices with 0 remaining degree
            # We can just filter the active list
            active = [u for u in active if remaining_deg[u] > 0]
            # Also, if a vertex in C had its degree reduced to 0, it's removed.
            # If a vertex NOT in C had its degree reduced to 0 (because it was connected to a vertex in C? No, only edges within C are removed).
            # Wait, the degree `remaining_deg` tracks the number of remaining edges incident to the vertex.
            # When we remove edges within C, we only decrement degrees of vertices in C.
            # Vertices outside C are not affected by removing edges *within* C.
            # So only vertices in C might drop to 0.
            # However, my `active` list update above is safe but slightly redundant if I just check `remaining_deg[u] > 0`.
            # Let's ensure active is correct.
            # Actually, the loop `active = [u for u in active if remaining_deg[u] > 0]` is correct.
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        # Try a few different random seeds for the heuristic to see if we get a better partition
        # The greedy algorithm is deterministic given the seed, but different tie-breaking might help?
        # Actually, my greedy algorithm picks max degree, then highest degree in candidates.
        # It's mostly deterministic. Let's just run it once. It's fast.
        # If we have time, we can try a "randomized" version where we pick a random vertex from top-k degrees.
        
        attempt = 0
        while time.time() < time_limit:
            heur_seed = seed + attempt * 7919
            cliques = greedy_partition_fast(n, edges, heur_seed)
            attempt += 1
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques[:]
                
            # If we have a very good solution (e.g., close to theoretical min), break early?
            # Theoretical min is roughly E / (k(k-1)/2) where k is clique size.
            # For dense graphs, cliques are small.
            # Let's just run a few times if fast.
            if attempt > 5:
                break
            
        if best_cliques is None:
            # Fallback: trivial partition (each edge is a clique)
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time: 
    # n=32 is small. Give it 10s.
    # n=64 is larger. Give it the rest.
    t1_limit = start_time + 10.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
