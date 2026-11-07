import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Precompute edge to candidates
        k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        edge_to_vs = {e: [] for e in k_edges}
        for v in range(k, n):
            N_v = set(neighbors[v])
            for e in k_edges:
                if e[0] in N_v and e[1] in N_v:
                    edge_to_vs[e].append(v)
        
        # Sort edges by scarcity (number of candidates)
        # We will re-shuffle ties in each iteration
        edges_by_scarcity = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
        
        best_result = None
        best_count = float('inf')
        
        # Number of iterations
        # We have ~140s total for 3 instances. ~40s per instance.
        # Greedy is fast. Let's do 100 iterations.
        max_iters = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            used_edges = set()
            v_matched = {v: set() for v in range(k, n)}
            v_cliques = {v: [] for v in range(k, n)}
            
            # Create a list of edges to process
            # To add randomness, we can shuffle the edges that have the same scarcity
            # Or simply shuffle the whole list and then stable sort by scarcity?
            # Stable sort preserves the original order for ties.
            # So we shuffle the input list first.
            edges_to_process = edges_by_scarcity[:]
            random.shuffle(edges_to_process)
            # Stable sort by scarcity
            edges_to_process.sort(key=lambda e: len(edge_to_vs[e]))
            
            for e in edges_to_process:
                u, w = e
                # Find valid candidates
                candidates = [v for v in edge_to_vs[e] 
                              if u not in v_matched[v] and w not in v_matched[v]]
                if not candidates:
                    continue
                
                # Pick best candidate
                # Heuristic: Maximize remaining degree
                # To add randomness, with small prob pick random
                if random.random() < 0.1:
                    v = random.choice(candidates)
                else:
                    v = max(candidates, key=lambda v: len(set(neighbors[v]) - v_matched[v]))
                
                # Assign
                v_matched[v].add(u)
                v_matched[v].add(w)
                v_cliques[v].append([u, w, v])
                used_edges.add(e)
            
            # Build cliques
            cliques = []
            
            # 3-cliques from matching
            for v in range(k, n):
                cliques.extend(v_cliques[v])
            
            # 2-cliques for unmatched neighbors
            for v in range(k, n):
                N_v = set(neighbors[v])
                matched = v_matched[v]
                for u in N_v - matched:
                    cliques.append([u, v])
            
            # Cover remaining K-edges
            remaining_edges = set(k_edges) - used_edges
            
            # Greedy clique cover on remaining K-edges
            # We want to find cliques in the graph H formed by remaining_edges
            rem_set = set(remaining_edges)
            
            while rem_set:
                # Find vertex with max degree in rem_set
                deg = defaultdict(int)
                for u, w in rem_set:
                    deg[u] += 1
                    deg[w] += 1
                
                if not deg:
                    break
                    
                # Pick vertex with max degree
                max_deg_vertex = max(deg, key=deg.get)
                
                # Build maximal clique containing max_deg_vertex
                # Candidates must be connected to max_deg_vertex in rem_set
                # and form a clique among themselves in rem_set
                
                # Neighbors of max_deg_vertex in rem_set
                neighbors_in_rem = set()
                for u, w in rem_set:
                    if u == max_deg_vertex:
                        neighbors_in_rem.add(w)
                    elif w == max_deg_vertex:
                        neighbors_in_rem.add(u)
                
                # Start with {max_deg_vertex}
                clique = {max_deg_vertex}
                changed = True
                while changed:
                    changed = False
                    # Try to add vertices from neighbors_in_rem
                    # Sort by degree in the induced subgraph of neighbors_in_rem?
                    # Or just iterate
                    for x in list(neighbors_in_rem):
                        if x in clique:
                            continue
                        # Check if x is connected to all in clique in rem_set
                        is_connected = True
                        for y in clique:
                            edge = (min(x, y), max(x, y))
                            if edge not in rem_set:
                                is_connected = False
                                break
                        if is_connected:
                            clique.add(x)
                            changed = True
                            # Remove x from neighbors_in_rem? No, keep it, it's in clique now.
                
                # Add clique
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    # Remove edges covered by this clique
                    for a, b in combinations(sorted(clique), 2):
                        rem_set.discard((a, b))
                else:
                    # Should not happen if rem_set is not empty
                    # Pick any edge
                    e = next(iter(rem_set))
                    cliques.append(list(e))
                    rem_set.discard(e)
            
            # Validate
            clean = []
            for c in cliques:
                clean.append([int(x) for x in c])
            
            # Verify
            edge_count = defaultdict(int)
            for c in clean:
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        u, w = min(c[i], c[j]), max(c[i], c[j])
                        edge_count[(u, w)] += 1
            
            all_edges = set()
            for u in range(k):
                for w in range(u + 1, k):
                    all_edges.add((u, w))
            for v in range(k, n):
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            valid = True
            for e in all_edges:
                if edge_count[e] != 1:
                    valid = False
                    break
            for e, cnt in edge_count.items():
                if e not in all_edges:
                    valid = False
                    break
            
            if not valid:
                continue
            
            count = len(clean)
            if count < best_count:
                best_count = count
                best_result = clean
                
        if best_result is None:
            # Fallback
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            best_result = [[int(x) for x in c] for c in cliques]
            
        return best_result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    while len(results) < 3:
        results.append([])
        
    return results
