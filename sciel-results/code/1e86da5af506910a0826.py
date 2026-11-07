"""Combined: greedy expansion from max clique + local search swaps + path-conflict graph clique."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    start_time = time.time()
    deadline = start_time + 80.0

    # Precompute common neighbours
    common = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c

    def find_max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if Pset:
                pivot = max(Pset, key=lambda v: len(adj[v] & Pset))
                cands = Pset - adj[pivot]
            else:
                cands = Pset
            for v in list(cands):
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.remove(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return best

    def check_feasibility(B):
        """Check if branch set B has a valid clique subdivision. Returns paths list or None."""
        B_set = set(B)
        non_adj = [(u, v) for u in B for v in B if u < v and v not in adj[u]]
        
        if len(non_adj) > n - len(B):
            return None

        cands = []
        for u, v in non_adj:
            ws = [w for w in common[u][v] if w not in B_set]
            cands.append(ws)
            if not ws:
                return None

        m = len(non_adj)
        right_verts = [v for v in range(n) if v not in B_set]
        right_idx = {v: i for i, v in enumerate(right_verts)}
        
        adj_left = []
        for ws in cands:
            adj_left.append([right_idx[w] for w in ws if w in right_idx])
            
        pair_u = [-1] * m
        pair_v = [-1] * len(right_verts)
        dist = [0] * m

        def bfs():
            q = deque()
            for u in range(m):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj_left[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(m):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1

        if matching < m:
            return None

        # Construct paths
        paths = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v in adj[u]:
                    paths.append([u, v])
        
        for i, (u, v) in enumerate(non_adj):
            w = right_verts[pair_u[i]]
            paths.append([u, w, v])
            
        return paths

    def greedy_expand(B, order, deadline):
        """Greedily add vertices from 'order' to B if feasible."""
        B_set = set(B)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for v in order:
                if v in B_set:
                    continue
                new_B = B + [v]
                paths = check_feasibility(new_B)
                if paths is not None:
                    B = new_B
                    B_set.add(v)
                    improved = True
                    break
        return B

    def local_search(B, deadline):
        """Try swaps: remove one branch, add one non-branch, if it increases size."""
        B_set = set(B)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for i in range(len(B)):
                for j in range(n):
                    if j in B_set:
                        continue
                    new_B = [B[k] for k in range(len(B)) if k != i] + [j]
                    paths = check_feasibility(new_B)
                    if paths is not None and len(new_B) > len(B):
                        B = new_B
                        B_set = set(B)
                        improved = True
                        break
                if improved:
                    break
        return B

    def path_conflict_clique(B, deadline):
        """
        Build a path-conflict graph where each node is a candidate path (u, w, v) 
        for a non-adjacent pair (u,v) in B, and edges connect internally-disjoint paths.
        Then find max clique in this conflict graph to select non-overlapping paths.
        """
        B_set = set(B)
        non_adj = [(u, v) for u in B for v in B if u < v and v not in adj[u]]
        
        if not non_adj:
            return None  # All adjacent, trivially feasible

        # For each non-adjacent pair, find all candidate internal vertices
        pair_paths = []
        for u, v in non_adj:
            ws = [w for w in common[u][v] if w not in B_set]
            if not ws:
                return None
            pair_paths.append((u, v, ws))

        # Build path-conflict graph
        # Each node: (pair_idx, w) where w is internal vertex for pair pair_idx
        all_nodes = []
        for pi, (u, v, ws) in enumerate(pair_paths):
            for w in ws:
                all_nodes.append((pi, w))

        num_nodes = len(all_nodes)
        if num_nodes == 0:
            return None

        # Build adjacency in conflict graph: two nodes are connected if they are 
        # for different pairs AND have different internal vertices
        conflict_adj = [set() for _ in range(num_nodes)]
        for i in range(num_nodes):
            pi_i, wi = all_nodes[i]
            for j in range(i + 1, num_nodes):
                pj_j, wj = all_nodes[j]
                if pi_i != pj_j and wi != wj:
                    conflict_adj[i].add(j)
                    conflict_adj[j].add(i)

        # Find max clique in conflict graph using Bron-Kerbosch with pivot
        best_clique = []
        
        def bk_pc(R, Pset, X):
            nonlocal best_clique
            if not Pset and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            if Pset:
                pivot = max(Pset, key=lambda v: len(conflict_adj[v] & Pset))
                cands = Pset - conflict_adj[pivot]
            else:
                cands = Pset
            for v in list(cands):
                bk_pc(R | {v}, Pset & conflict_adj[v], X & conflict_adj[v])
                Pset.remove(v)
                X.add(v)

        bk_pc(set(), set(range(num_nodes)), set())

        if len(best_clique) < len(non_adj):
            return None

        # Verify we have at least one path per pair
        selected = {}
        for node_idx in best_clique:
            pi, w = all_nodes[node_idx]
            if pi not in selected:
                selected[pi] = w

        if len(selected) < len(non_adj):
            return None

        # Construct paths
        paths = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v in adj[u]:
                    paths.append([u, v])
        
        for pi, (u, v, ws) in enumerate(pair_paths):
            w = selected[pi]
            paths.append([u, w, v])
            
        return paths

    def try_branch_set_with_conflict(B, deadline):
        """Try to find a valid subdivision using either matching or conflict-clique approach."""
        # First try the faster matching approach
        paths = check_feasibility(B)
        if paths is not None:
            return paths
        # Fall back to conflict-clique approach
        if time.time() < deadline:
            paths = path_conflict_clique(B, deadline)
            if paths is not None:
                return paths
        return None

    # Phase 1: Find max clique
    clique = find_max_clique()
    if len(clique) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Phase 2: Greedy expansion with multiple orderings
    best_B = list(clique)
    
    # Ordering 1: by degree descending
    order1 = sorted(range(n), key=lambda x: -degrees[x])
    B1 = greedy_expand(list(clique), order1, deadline)
    if len(B1) > len(best_B):
        best_B = B1

    # Ordering 2: by common-neighbor count (vertices that can serve as internal)
    def common_count(v):
        return sum(1 for u in range(n) if u != v and len(common[u][v]) > 0)
    order2 = sorted(range(n), key=lambda x: -common_count(x))
    B2 = greedy_expand(list(clique), order2, deadline)
    if len(B2) > len(best_B):
        best_B = B2

    # Phase 3: Local search from best_B
    best_B = local_search(best_B, deadline)

    # Phase 4: Try greedy from different starting points
    for k_start in range(2, min(6, n)):
        if time.time() > deadline:
            break
        top_k = sorted(range(n), key=lambda x: -degrees[x])[:k_start]
        B_k = greedy_expand(list(top_k), order1, deadline)
        if len(B_k) > len(best_B):
            best_B = B_k
            best_B = local_search(best_B, deadline)

    # Phase 5: Use conflict-clique approach to explore larger branch sets
    if time.time() < deadline - 10.0:
        pool = sorted(range(n), key=lambda x: -degrees[x])[:min(18, n)]
        for k in range(len(best_B) + 1, min(14, n + 1)):
            if time.time() > deadline - 3.0:
                break
            found = False
            for combo in combinations(pool, k):
                if time.time() > deadline - 2.0:
                    found = True
                    break
                B = list(combo)
                paths = try_branch_set_with_conflict(B, deadline)
                if paths is not None:
                    best_B = B
                    found = True
                    break
            if found:
                break

    # Phase 6: Local search with conflict-clique feasibility check
    if time.time() < deadline - 5.0:
        B_set = set(best_B)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(best_B)):
                for j in range(n):
                    if j in B_set:
                        continue
                    new_B = [best_B[k] for k in range(len(best_B)) if k != i] + [j]
                    paths = try_branch_set_with_conflict(new_B, deadline)
                    if paths is not None and len(new_B) > len(best_B):
                        best_B = new_B
                        B_set = set(best_B)
                        improved = True
                        break
                if improved:
                    break

    # Final verification
    final_paths = try_branch_set_with_conflict(best_B, deadline)
    if final_paths is None:
        # Fallback to clique
        best_B = list(clique)
        final_paths = check_feasibility(best_B)
    
    if len(best_B) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    return {'branches': [int(b) for b in best_B], 'paths': [[int(x) for x in p] for p in final_paths]}
