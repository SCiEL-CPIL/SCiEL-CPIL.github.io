import time
import random
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def max_bipartite_matching(adj_left, left_nodes, right_nodes):
        """
        Find maximum matching in bipartite graph.
        adj_left: dict mapping left node -> set of right nodes
        left_nodes, right_nodes: lists of nodes
        Returns list of (left, right) pairs
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Use Hopcroft-Karp for efficiency
        INF = float('inf')
        pair_u = {u: -1 for u in left_nodes}
        pair_v = {v: -1 for v in right_nodes}
        dist = {u: 0 for u in left_nodes}
        
        from collections import deque
        
        def bfs():
            queue = deque()
            for u in left_nodes:
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_left.get(u, set()):
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == INF:
                        dist[pu] = dist[u] + 1
                        queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj_left.get(u, set()):
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False
        
        matching = []
        while bfs():
            for u in left_nodes:
                if pair_u[u] == -1:
                    if dfs(u):
                        matching.append((u, pair_u[u]))
        
        return matching

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        I = list(range(k, n))
        K = list(range(k))
        
        # Total edges
        total_edges = k * (k - 1) // 2 + sum(len(neighbors[v]) for v in I)
        
        # For each v in I, N(v) is a subset of K
        # We want to partition all edges into cliques
        # Each clique is either:
        #   - A subset of K (covering K-K edges)
        #   - {v} ∪ S where S ⊆ N(v) (covering v-S edges and S-S edges if S is a clique in K)
        #   - {v, u} where u ∈ N(v) (covering just the v-u edge)
        
        # Key insight: if we make a clique {v} ∪ S where S ⊆ N(v) and S is a clique in K,
        # we cover |S| cross edges AND |S|(|S|-1)/2 K-K edges in one clique.
        # This is very efficient!
        
        # Strategy: For each v, find a large clique S ⊆ N(v) in K, and make {v} ∪ S.
        # Then cover remaining edges.
        
        # Since K is a clique, any subset of K is a clique. So any S ⊆ N(v) works!
        # The constraint is that each K-K edge can only be covered once.
        
        # So the problem reduces to: assign K-K edges to v's (where both endpoints are in N(v)),
        # maximizing the number of K-K edges covered by v-cliques (each such clique covers 
        # a set of K-K edges that form a clique in K, i.e., any subset).
        
        # Actually, if we assign a set S_v ⊆ N(v) to vertex v, the K-K edges covered are 
        # all pairs in S_v. We need these edge sets to be disjoint across v's.
        
        # This is equivalent to: find edge-disjoint cliques in K, each assigned to a v 
        # whose neighborhood contains all vertices of the clique.
        
        # Greedy approach: for each v (in some order), find the largest clique in the 
        # remaining subgraph of N(v) that can be assigned to v.
        
        # Better: use a matching-based approach.
        # For each v, we want to pair up vertices in N(v) to form K-K edges covered by v.
        # Each pair (u,w) in N(v) means v covers the edge (u,w).
        # The pairs must be edge-disjoint across all v's.
        
        # This is a b-matching / edge coloring type problem.
        
        # Let's use a simpler greedy with multiple strategies:
        
        best_result = None
        best_count = float('inf')
        
        for strategy in range(4):
            if time.time() > deadline - 5:
                break
            
            # Track which K-K edges are used
            used_kk_edges = set()
            # For each v, which vertices in K are consumed
            v_consumed = {v: set() for v in I}
            v_cliques = {v: [] for v in I}
            
            # Build edge_to_vs: for each K-K edge, which v's can use it
            k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
            edge_to_vs = {e: [] for e in k_edges}
            for v in I:
                N_v = set(neighbors[v])
                for e in k_edges:
                    if e[0] in N_v and e[1] in N_v:
                        edge_to_vs[e].append(v)
            
            if strategy == 0:
                # Scarcity-based: process edges with fewest candidate v's first
                edges_sorted = sorted(k_edges, key=lambda e: len(edge_to_vs[e]))
                for e in edges_sorted:
                    u, w = e
                    candidates = [v for v in edge_to_vs[e]
                                  if u not in v_consumed[v] and w not in v_consumed[v]]
                    if not candidates:
                        continue
                    # Pick v with largest remaining neighborhood
                    v = max(candidates, key=lambda v: len(set(neighbors[v]) - v_consumed[v]))
                    v_consumed[v].add(u)
                    v_consumed[v].add(w)
                    v_cliques[v].append([u, w, v])
                    used_kk_edges.add(e)
                    
            elif strategy == 1:
                # Process v's in order of decreasing |N(v)|, greedily match pairs
                vs_sorted = sorted(I, key=lambda v: -len(neighbors[v]))
                for v in vs_sorted:
                    N_v = set(neighbors[v]) - v_consumed[v]
                    if len(N_v) < 2:
                        continue
                    # Find max matching in the complete graph on N_v (which is just a matching)
                    # Since N_v is a clique in K, any pair works. We just need to match as many as possible.
                    # But we also need to ensure these edges aren't used by other v's.
                    # Greedy: sort by some heuristic and match
                    N_v_list = sorted(N_v)
                    # Try to match all possible pairs
                    matched = set()
                    remaining = list(N_v_list)
                    while len(remaining) >= 2:
                        # Pick two vertices that have the fewest other candidates
                        best_pair = None
                        best_score = float('inf')
                        for i in range(len(remaining)):
                            for j in range(i + 1, len(remaining)):
                                u, w = remaining[i], remaining[j]
                                # Score: number of other v's that could use this edge
                                score = len([vv for vv in edge_to_vs[(min(u,w), max(u,w))] 
                                            if vv != v and u not in v_consumed[vv] and w not in v_consumed[vv]])
                                if score < best_score:
                                    best_score = score
                                    best_pair = (u, w)
                        if best_pair is None:
                            break
                        u, w = best_pair
                        matched.add(u)
                        matched.add(w)
                        remaining.remove(u)
                        remaining.remove(w)
                        v_consumed[v].add(u)
                        v_consumed[v].add(w)
                        v_cliques[v].append([u, w, v])
                        used_kk_edges.add((min(u, w), max(u, w)))
                        
            elif strategy == 2:
                # Use bipartite matching per v
                vs_sorted = sorted(I, key=lambda v: -len(neighbors[v]))
                for v in vs_sorted:
                    N_v = set(neighbors[v]) - v_consumed[v]
                    if len(N_v) < 2:
                        continue
                    # Build a graph where we want to find a matching
                    # Since any two vertices in N_v form an edge in K, we just need a matching
                    # But we want to choose which edges to assign to v
                    # Use greedy: match vertices with fewest alternatives
                    N_v_list = sorted(N_v, key=lambda x: len([vv for vv in I if x in set(neighbors[vv]) and vv != v]))
                    matched = set()
                    remaining = list(N_v_list)
                    while len(remaining) >= 2:
                        # Pick the vertex with fewest remaining candidates
                        u = remaining[0]
                        # Find a partner
                        best_w = None
                        best_score = float('inf')
                        for w in remaining[1:]:
                            e = (min(u, w), max(u, w))
                            score = len([vv for vv in edge_to_vs[e] if vv != v and u not in v_consumed[vv] and w not in v_consumed[vv]])
                            if score < best_score:
                                best_score = score
                                best_w = w
                        if best_w is None:
                            break
                        matched.add(u)
                        matched.add(best_w)
                        remaining.remove(u)
                        remaining.remove(best_w)
                        v_consumed[v].add(u)
                        v_consumed[v].add(best_w)
                        v_cliques[v].append([u, best_w, v])
                        used_kk_edges.add((min(u, best_w), max(u, best_w)))
                        
            else:
                # Randomized with restarts
                rng = random.Random(seed + strategy * 1000)
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    used_kk_edges = set()
                    v_consumed = {v: set() for v in I}
                    v_cliques = {v: [] for v in I}
                    
                    edges_rand = k_edges[:]
                    rng.shuffle(edges_rand)
                    for e in edges_rand:
                        u, w = e
                        candidates = [v for v in edge_to_vs[e]
                                      if u not in v_consumed[v] and w not in v_consumed[v]]
                        if not candidates:
                            continue
                        v = rng.choice(candidates)
                        v_consumed[v].add(u)
                        v_consumed[v].add(w)
                        v_cliques[v].append([u, w, v])
                        used_kk_edges.add(e)
            
            # Build all cliques
            cliques = []
            
            # Add matching cliques {u, w, v}
            for v in I:
                cliques.extend(v_cliques[v])
            
            # Add individual cliques for unmatched neighbors
            for v in I:
                N_v = set(neighbors[v])
                consumed = v_consumed[v]
                for u in N_v - consumed:
                    cliques.append([u, v])
            
            # Cover remaining K-K edges
            remaining_edges = [e for e in k_edges if e not in used_kk_edges]
            if remaining_edges:
                # Greedy: find maximal cliques in the remaining subgraph
                rem_set = set(remaining_edges)
                while rem_set:
                    # Start with an edge
                    e = next(iter(rem_set))
                    clique = {e[0], e[1]}
                    changed = True
                    while changed:
                        changed = False
                        for x in range(k):
                            if x not in clique:
                                if all((min(x, y), max(x, y)) in rem_set for y in clique):
                                    clique.add(x)
                                    changed = True
                    if len(clique) >= 2:
                        cliques.append(sorted(clique))
                        for a, b in combinations(clique, 2):
                            rem_set.discard((min(a, b), max(a, b)))
                    else:
                        cliques.append(list(e))
                        rem_set.discard(e)
            
            # Validate
            clean = []
            for c in cliques:
                clean.append([int(x) for x in c])
            
            # Verify: every edge covered exactly once, no extra edges
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
            for v in I:
                for u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))
            
            valid = True
            for e in all_edges:
                if edge_count[e] != 1:
                    valid = False
                    break
            if valid:
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
            for v in I:
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
