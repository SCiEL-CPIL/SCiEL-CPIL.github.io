import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def find_max_clique_greedy(candidates, adj):
        """Find a large clique in graph (candidates, adj) using greedy heuristic."""
        if not candidates:
            return set()
        C = set()
        cand = set(candidates)
        while cand:
            # Pick vertex with max degree within cand
            best_v = None
            max_deg = -1
            for v in cand:
                deg = len(adj[v] & cand)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            if best_v is None:
                break
            # Check compatibility with C
            can_add = True
            for u in C:
                if u not in adj[best_v]:
                    can_add = False
                    break
            if can_add:
                C.add(best_v)
                cand = cand & adj[best_v]
            else:
                cand.discard(best_v)
        return C

    def clique_partition(n, k, p, seed):
        rng = random.Random(seed)
        # Build adjacency for cross edges
        adj_I = [set() for _ in range(n - k)]  # adj_I[i] = set of K-neighbors of vertex k+i
        
        for i in range(n - k):
            v = k + i
            for u in range(k):
                if rng.random() < p:
                    adj_I[i].add(u)
        
        used_K_edges = set()
        cliques = []
        
        # Process I vertices in order of decreasing degree (greedy)
        i_indices = sorted(range(n - k), key=lambda i: len(adj_I[i]), reverse=True)
        
        for i in i_indices:
            v = k + i
            U = set(adj_I[i])
            while U:
                # Build available graph on U: edge (u,w) exists if (u,w) NOT in used_K_edges
                # Find max clique in available graph
                # adj_avail: for each u in U, neighbors in U that are available
                adj_avail = {u: set() for u in U}
                for u in U:
                    for w in U:
                        if u < w and (u, w) not in used_K_edges:
                            adj_avail[u].add(w)
                            adj_avail[w].add(u)
                
                S = find_max_clique_greedy(U, adj_avail)
                
                if not S:
                    # Fallback: pick one vertex
                    u = next(iter(U))
                    S = {u}
                
                # Add clique {v} U S
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Update used_K_edges
                for u, w in combinations(sorted(S), 2):
                    used_K_edges.add((u, w))
                U -= S
        
        # Cover remaining K-K edges
        rem_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_K_edges:
                    rem_edges.append((u, w))
        
        if rem_edges:
            # Build graph on K with remaining edges
            adj_rem = [set() for _ in range(k)]
            for u, w in rem_edges:
                adj_rem[u].add(w)
                adj_rem[w].add(u)
            
            remaining = set(rem_edges)
            
            while remaining:
                # Find vertices involved
                verts = set()
                for u, w in remaining:
                    verts.add(u)
                    verts.add(w)
                
                # Find max clique in remaining graph
                C = find_max_clique_greedy(verts, adj_rem)
                
                if not C:
                    # Fallback: pick an edge
                    u, w = next(iter(remaining))
                    C = {u, w}
                
                clique = sorted(C)
                cliques.append(clique)
                
                # Remove edges in C from remaining
                for u, w in combinations(sorted(C), 2):
                    remaining.discard((u, w))
                
                # Update adj_rem
                for u in C:
                    for w in C:
                        if u != w:
                            adj_rem[u].discard(w)
                            adj_rem[w].discard(u)
        
        # Convert to proper int lists
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    # Local search: try to merge cliques or reassign to reduce count
    def try_improve(cliques, n, k, p, seed, time_limit):
        """Try to reduce number of cliques via local search."""
        if time.time() > time_limit:
            return cliques
        
        best = cliques
        best_count = len(cliques)
        
        # Build edge -> clique index mapping
        def build_edge_map(cliques):
            emap = {}
            for idx, c in enumerate(cliques):
                for u, w in combinations(c, 2):
                    emap[(min(u, w), max(u, w))] = idx
            return emap
        
        # Try merging pairs of cliques if their union is a clique
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            emap = build_edge_map(best)
            # Check if union of two cliques forms a clique
            for i in range(len(best)):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > time_limit:
                        break
                    # Check if all pairs between best[i] and best[j] are edges
                    # In a split graph, edges exist if: both in K, both in I (no), or cross
                    # We need to check actual edge existence
                    union = set(best[i]) | set(best[j])
                    is_clique = True
                    for u, w in combinations(union, 2):
                        if (u, w) not in emap:
                            is_clique = False
                            break
                    if is_clique and len(union) >= 2:
                        # Merge
                        new_cliques = [c for idx, c in enumerate(best) if idx != i and idx != j]
                        new_cliques.append(sorted(union))
                        if len(new_cliques) < best_count:
                            best = new_cliques
                            best_count = len(best)
                            improved = True
                            break
                if improved:
                    break
        
        return best

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        
        # Try to improve with local search
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            cliques = try_improve(cliques, n, k, p, seed, time.time() + min(remaining_time * 0.5, 10))
        
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
