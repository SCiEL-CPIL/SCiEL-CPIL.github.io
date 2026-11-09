import time
import random
from collections import deque
from itertools import combinations
import networkx as nx
import numpy as np
import sys
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # ---------- Bron-Kerbosch with pivot to find max clique ----------
    best_clique = []

    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Pivot
        candidates_for_pivot = list(Pset) + list(X)
        u = None
        max_neighbors = -1
        for v in candidates_for_pivot:
            count = len(adj[v] & Pset)
            if count > max_neighbors:
                max_neighbors = count
                u = v
        if u is None:
            candidates = list(Pset)
        else:
            candidates = [v for v in Pset if v not in adj[u]]
        candidates.sort(key=lambda x: degrees[x], reverse=True)
        for v in candidates:
            if time.time() > deadline:
                return
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset = Pset - {v}
            X = X | {v}

    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())

    # Greedy + random restarts to find good cliques
    def greedy_clique(start_order):
        remaining = set(start_order)
        clique = set()
        while remaining:
            best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
            clique.add(best_v)
            remaining = remaining & adj[best_v]
        return clique

    for trial in range(100):
        if time.time() > deadline - 30:
            break
        shuffled = list(range(n))
        random.shuffle(shuffled)
        c = greedy_clique(shuffled)
        if len(c) > len(best_clique):
            best_clique = list(c)

    if not best_clique:
        best_clique = [0, 1]

    def maxflow_feasible(branches_list):
        """Check via max-flow whether a valid subdivision exists for branches_list.
        Returns True if feasible, False otherwise."""
        k = len(branches_list)
        if k < 2:
            return False
        branches_set = set(branches_list)
        
        # Build flow network
        S = 2 * n
        T = 2 * n + 1
        
        G = nx.DiGraph()
        INF = 10**9
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branches_set:
                cap = k - 1
            else:
                cap = 1
            G.add_edge(vin, vout, capacity=cap)
        
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            G.add_edge(uout, vin, capacity=INF)
            G.add_edge(vout, uin, capacity=INF)
        
        for b in branches_list:
            bin_ = 2 * b
            bout = 2 * b + 1
            G.add_edge(S, bin_, capacity=k - 1)
            G.add_edge(bout, T, capacity=k - 1)
        
        try:
            flow_value, _ = nx.maximum_flow(G, S, T)
        except Exception:
            return False
        
        return flow_value >= k * (k - 1)

    def extract_paths_from_flow(branches_list):
        """Given a feasible branch set, extract the actual paths by decomposing flow.
        Returns paths_dict mapping (i,j) -> path list, or None on failure."""
        k = len(branches_list)
        if k < 2:
            return None
        branches_set = set(branches_list)
        
        S = 2 * n
        T = 2 * n + 1
        
        G = nx.DiGraph()
        INF = 10**9
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branches_set:
                cap = k - 1
            else:
                cap = 1
            G.add_edge(vin, vout, capacity=cap)
        
        for u, v in edges:
            uout = 2 * u + 1
            vin = 2 * v
            vout = 2 * v + 1
            uin = 2 * u
            G.add_edge(uout, vin, capacity=INF)
            G.add_edge(vout, uin, capacity=INF)
        
        for b in branches_list:
            bin_ = 2 * b
            bout = 2 * b + 1
            G.add_edge(S, bin_, capacity=k - 1)
            G.add_edge(bout, T, capacity=k - 1)
        
        try:
            flow_value, flow_dict = nx.maximum_flow(G, S, T)
        except Exception:
            return None
        
        if flow_value < k * (k - 1):
            return None
        
        # Build directed multigraph of flow-carrying edges between original vertices
        # For each original edge u->v, track how many units of flow go u_out -> v_in
        # We'll decompose into individual paths.
        
        # Build adjacency of flow-carrying directed edges
        # flow_adj[u] = list of (v, count) for flow from u to v
        flow_adj = [[] for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                uout = 2 * u + 1
                vin = 2 * v
                if uout in flow_dict and vin in flow_dict[uout]:
                    f = flow_dict[uout][vin]
                    if f > 0:
                        flow_adj[u].append((v, int(f)))
        
        # Now decompose: we have k(k-1) units of flow total.
        # Each unit starts at some branch b (from S->b_in->b_out) and ends at some branch c (c_out->T).
        # We need to trace each unit from its source branch to its sink branch.
        
        # For each branch b, there are (k-1) units of flow leaving b_out.
        # We trace each unit by following the flow edges.
        
        paths_dict = {}
        # Track remaining flow on each directed edge
        remaining = {}
        for u in range(n):
            for v, cnt in flow_adj[u]:
                remaining[(u, v)] = cnt
        
        # For each branch b, trace (k-1) paths starting from b
        for b in branches_list:
            traced = 0
            while traced < k - 1:
                # Find next edge from b with remaining flow
                next_v = None
                for v, cnt in flow_adj[b]:
                    if remaining.get((b, v), 0) > 0:
                        next_v = v
                        break
                if next_v is None:
                    return None  # Should not happen if flow is valid
                
                remaining[(b, next_v)] -= 1
                traced += 1
                
                # Trace path from next_v
                path = [b, next_v]
                cur = next_v
                while cur not in branches_set:
                    next_w = None
                    for w, cnt in flow_adj[cur]:
                        if remaining.get((cur, w), 0) > 0:
                            next_w = w
                            break
                    if next_w is None:
                        return None
                    remaining[(cur, next_w)] -= 1
                    path.append(next_w)
                    cur = next_w
                
                # cur is now a branch vertex
                if cur == b:
                    return None  # Cycle back to start, shouldn't happen
                
                # Record the path
                i_idx = branches_list.index(b)
                j_idx = branches_list.index(cur)
                key = (min(i_idx, j_idx), max(i_idx, j_idx))
                if key in paths_dict:
                    return None  # Duplicate pair
                paths_dict[key] = path
        
        if len(paths_dict) != k * (k - 1) // 2:
            return None
        
        return paths_dict

    # ---------- Path finding (shortest path avoiding forbidden vertices) ----------
    def shortest_path(s, t, forbidden):
        """BFS for shortest path from s to t avoiding forbidden set (interior only)."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if nb == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nb in forbidden:
                    continue
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_k_disjoint_paths(branches_list, max_attempts=200):
        """Try to find internally disjoint paths for all pairs using backtracking.
        Uses max-flow feasibility check first, then tries multiple orderings."""
        k = len(branches_list)
        if k < 2:
            return None
        branches_set = set(branches_list)
        pairs = list(combinations(range(k), 2))
        
        # First check feasibility with max-flow
        if not maxflow_feasible(branches_list):
            return None
        
        # Try to extract from flow directly
        paths_dict = extract_paths_from_flow(branches_list)
        if paths_dict is not None:
            return paths_dict
        
        # If flow extraction failed, try BFS with different orderings
        # Sort pairs by shortest path length (easier ones first)
        def pair_difficulty(pair):
            i, j = pair
            s, t = branches_list[i], branches_list[j]
            forbidden = branches_set - {s, t}
            p = shortest_path(s, t, forbidden)
            if p is None:
                return float('inf')
            return len(p)
        
        sorted_pairs = sorted(pairs, key=pair_difficulty)
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            if attempt == 0:
                order = sorted_pairs
            else:
                order = list(pairs)
                random.shuffle(order)
            
            assigned = {}
            used_interior = set()
            success = True
            for i, j in order:
                s, t = branches_list[i], branches_list[j]
                forbidden = (branches_set - {s, t}) | used_interior
                p = shortest_path(s, t, forbidden)
                if p is None:
                    success = False
                    break
                assigned[(i, j)] = p
                used_interior |= set(p[1:-1])
            if success and len(assigned) == len(pairs):
                return assigned
        
        # Try backtracking for small k
        if k <= 5:
            result = backtrack_paths(branches_list, branches_set, pairs, 0, set(), {})
            if result is not None:
                return result
        
        return None

    def backtrack_paths(branches_list, branches_set, pairs, idx, used_interior, assigned):
        """Backtracking search for path assignment (for small k)."""
        if time.time() > deadline:
            return None
        if idx == len(pairs):
            return assigned
        
        i, j = pairs[idx]
        s, t = branches_list[i], branches_list[j]
        forbidden = (branches_set - {s, t}) | used_interior
        
        # Try to find a path
        p = shortest_path(s, t, forbidden)
        if p is not None:
            new_used = used_interior | set(p[1:-1])
            new_assigned = dict(assigned)
            new_assigned[(i, j)] = p
            result = backtrack_paths(branches_list, branches_set, pairs, idx + 1, new_used, new_assigned)
            if result is not None:
                return result
        
        # Try alternative: find a longer path by removing some used vertices temporarily
        # This is expensive; only try a few alternatives
        if len(used_interior) > 0 and idx < 3:
            # Try removing one used vertex at a time to find alternative paths
            for rem_v in list(used_interior)[:5]:
                alt_forbidden = (branches_set - {s, t}) | (used_interior - {rem_v})
                p_alt = shortest_path(s, t, alt_forbidden)
                if p_alt is not None:
                    new_used = (used_interior - {rem_v}) | set(p_alt[1:-1])
                    new_assigned = dict(assigned)
                    new_assigned[(i, j)] = p_alt
                    result = backtrack_paths(branches_list, branches_set, pairs, idx + 1, new_used, new_assigned)
                    if result is not None:
                        return result
        
        return None

    # ---------- Verify a clique subdivision ----------
    def verify_clique_subdivision(clique):
        """Try to find internally disjoint paths for all pairs in clique.
        Returns (True, paths_dict) or (False, None)."""
        k = len(clique)
        if k < 2:
            return False, None
        
        paths_dict = find_k_disjoint_paths(clique)
        if paths_dict is not None:
            return True, paths_dict
        return False, None

    # ---------- Search for maximum valid subdivision ----------
    best_valid = None
    best_valid_size = 0

    # Try the max clique
    if len(best_clique) >= 2:
        valid, paths = verify_clique_subdivision(best_clique)
        if valid:
            best_valid = (best_clique, paths)
            best_valid_size = len(best_clique)

    # Try subsets of the clique
    if best_valid is None:
        for size in range(len(best_clique) - 1, 1, -1):
            if time.time() > deadline:
                break
            found = False
            for subset in combinations(best_clique, size):
                valid, paths = verify_clique_subdivision(list(subset))
                if valid:
                    best_valid = (list(subset), paths)
                    best_valid_size = size
                    found = True
                    break
            if found:
                break

    # Try to grow: add vertices one by one (including non-clique vertices)
    if best_valid is not None:
        current_branches = list(best_valid[0])
        current_paths = best_valid[1]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort candidates by degree (higher degree first) for better chances
            candidates = [v for v in range(n) if v not in set(current_branches)]
            candidates.sort(key=lambda x: degrees[x], reverse=True)
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = current_branches + [v]
                valid, new_paths = verify_clique_subdivision(new_branches)
                if valid:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_valid = (current_branches, current_paths)
                    best_valid_size = len(current_branches)
                    improved = True
                    break

    # Try random non-clique sets to find larger subdivisions
    if time.time() < deadline - 5:
        for trial in range(200):
            if time.time() > deadline - 3:
                break
            # Random set of size best_valid_size + 1
            target_size = best_valid_size + 1
            if target_size > n:
                break
            random_set = random.sample(range(n), target_size)
            valid, paths = verify_clique_subdivision(random_set)
            if valid:
                best_valid = (random_set, paths)
                best_valid_size = target_size

    # Fallback
    if best_valid is None or len(best_valid[0]) < 2:
        # Try all edges
        for u, v in edges:
            if u != v:
                best_valid = ([u, v], {(0, 1): [u, v]})
                best_valid_size = 2
                break

    if best_valid is None:
        u, v = edges[0]
        best_valid = ([u, v], {(0, 1): [u, v]})

    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    branch_to_idx = {b: i for i, b in enumerate(branches)}

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j))
            if p is None:
                p = paths_dict.get((j, i))
            if p is None:
                # Fallback: direct edge or shortest path
                if branches[i] in adj[branches[j]]:
                    p = [branches[i], branches[j]]
                else:
                    p = shortest_path(branches[i], branches[j], set(branches) - {branches[i], branches[j]})
                    if p is None:
                        p = [branches[i], branches[j]]
            # Ensure correct orientation
            if p[0] != branches[i]:
                p = list(reversed(p))
            output_paths.append(list(p))

    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
