import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # Edges within clique C
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Edges between C and I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        
        if time.time() - start_time > 140:
            # Safety break, though unlikely to be reached early
            break

        # Reconstruct neighbors
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        
        # Store cross edges in a list of sets for each I vertex, and mark C neighbors
        cross_neighbors = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    cross_neighbors[v].add(u)
                    cross_neighbors[u].add(v) # Not strictly needed for u in C, but good for symmetry if we used it

        cliques = []
        covered_internal = set()
        
        # 1. Process I vertices
        # Sort by degree in C descending
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(cross_neighbors[v]), reverse=True)
        
        C_vertices = list(range(k))
        
        for v in I_vertices:
            N_v = cross_neighbors[v]
            if not N_v:
                continue
            
            # Find a maximal clique in the graph of available vertices in N_v
            # Available means no internal edge is blocked by covered_internal
            # Let's build the graph of conflicts or just check pairs.
            # Since k is small (<=30), we can just greedily build S.
            
            S = set()
            # Sort N_v to ensure determinism if needed, or just iterate
            for u in sorted(N_v):
                # Check if u can be added to S
                # 1. (u, w) must not be in covered_internal for any w in S
                # 2. (u, w) must not be in covered_internal for any w in S (same)
                # Basically, u must not form a covered pair with any existing member of S.
                valid = True
                for w in S:
                    pair = (min(u, w), max(u, w))
                    if pair in covered_internal:
                        valid = False
                        break
                if valid:
                    S.add(u)
            
            if len(S) >= 2:
                # Add clique [v] + S
                cliques.append([v] + list(S))
                # Mark all pairs in S as covered
                for u, w in combinations(sorted(S), 2):
                    covered_internal.add((u, w))
            elif len(S) == 1:
                # If S has size 1, we can't form a clique with v and S that covers internal edges.
                # We must cover the cross edges (v, u) separately.
                # But wait, if S has size 1, it means u is the only vertex in N_v that doesn't conflict with covered_internal?
                # No, the greedy logic above adds u if it doesn't conflict with *existing* S.
                # If S is empty, it adds the first u.
                # If the second u conflicts with the first (pair in covered_internal), it's not added.
                # So S is a maximal set of vertices in N_v such that no pair is in covered_internal.
                # If |S| == 1, we have one vertex u.
                # We can form clique [v, u]. This covers edge (v, u). It does NOT cover any internal edge.
                # So covered_internal is unchanged.
                u = next(iter(S))
                cliques.append([v, u])
            else:
                # S is empty? This means N_v is empty? No, we checked N_v.
                # It means every u in N_v conflicts with... wait.
                # If S is empty, the loop didn't add anything.
                # This happens if N_v is empty. But we checked `if not N_v: continue`.
                # So S must have at least 1 element?
                # Ah, the loop `for u in sorted(N_v)` adds the first u always (S is empty, so no conflict).
                # So |S| >= 1 always if N_v is not empty.
                pass

        # 2. Handle remaining internal edges in C
        # R = { (u, w) in C : (u, w) not in covered_internal }
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_internal:
                    remaining_edges.add((u, w))
        
        rem_set = set(remaining_edges)
        
        while rem_set:
            if time.time() - start_time > 145:
                # Emergency: just add single edges to finish quickly?
                # But this might be invalid if we don't cover all?
                # The requirement is "every edge must lie in exactly one listed clique".
                # If we time out, we should return what we have? No, "Any broken rule makes the whole result invalid".
                # So we must finish.
                # For k=30, the number of remaining edges is likely small.
                # This loop should be fast.
                pass

            # Pick an edge
            u, w = next(iter(rem_set))
            clique_vertices = {u, w}
            
            # Try to grow clique
            # Candidates: any x in C such that (x, y) in rem_set for all y in clique_vertices
            # We can iterate through all vertices in C
            for x in range(k):
                if x in clique_vertices:
                    continue
                is_connected = True
                for y in clique_vertices:
                    pair = (min(x, y), max(x, y))
                    if pair not in rem_set:
                        is_connected = False
                        break
                if is_connected:
                    clique_vertices.add(x)
            
            # Remove edges within clique_vertices
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_set.discard((u1, u2))
            
            cliques.append(list(clique_vertices))
        
        results.append(cliques)

    return results
