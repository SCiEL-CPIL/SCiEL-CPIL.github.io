"""Structural change: model as minimum cardinality independent dominating set on P_24,
then add edges between S-vertices at path-distance 2. Verify and optimize."""
import time
from itertools import combinations, product

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    colors = [-1] * n
    for i in range(n):
        if colors[i] == -1:
            colors[i] = 0
            q = [i]
            while q:
                u = q.pop(0)
                for v in adj[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        return False
    return True

def _valid_solution(n, base_edges, added_edges):
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    diam = _bfs_diameter(n, adj)
    return diam != -1 and diam <= 4

def _path_dist(u, v):
    return abs(u - v)

def _dominates(S, n):
    """Check if S is a dominating set on path P_n."""
    covered = set()
    for s in S:
        covered.add(s)
        if s - 1 >= 0:
            covered.add(s - 1)
        if s + 1 < n:
            covered.add(s + 1)
    return len(covered) == n

def _is_independent(S):
    """Check no two vertices in S are adjacent on the path."""
    S_sorted = sorted(S)
    for i in range(len(S_sorted) - 1):
        if S_sorted[i + 1] - S_sorted[i] == 1:
            return False
    return True

def _edges_from_S(S, base_edge_set):
    """Generate edges between S-vertices at path-distance 2, not already in base."""
    edges = []
    S_set = set(S)
    for u, v in combinations(sorted(S), 2):
        if _path_dist(u, v) == 2:
            a, b = min(u, v), max(u, v)
            if (a, b) not in base_edge_set:
                edges.append((a, b))
    return edges

def _try_all_S_of_size(k, n, base_edge_set):
    """Try all independent dominating sets of size k, return best valid solution."""
    best = None
    for S in combinations(range(n), k):
        if not _is_independent(S):
            continue
        if not _dominates(S, n):
            continue
        added = _edges_from_S(S, base_edge_set)
        if _valid_solution(n, P['edges'], added):
            if best is None or len(added) < len(best):
                best = added
    return best

def _greedy_dominating_set(n, max_iter=100000):
    """Greedy: pick vertex covering most uncovered, maintain independence."""
    S = []
    covered = set()
    for _ in range(n):
        best_v = -1
        best_gain = 0
        for v in range(n):
            if v in covered:
                continue
            # Check independence
            independent = True
            for s in S:
                if abs(v - s) == 1:
                    independent = False
                    break
            if not independent:
                continue
            gain = 0
            for x in [v - 1, v, v + 1]:
                if 0 <= x < n and x not in covered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v == -1:
            break
        S.append(best_v)
        for x in [best_v - 1, best_v, best_v + 1]:
            if 0 <= x < n:
                covered.add(x)
    return S

def construct():
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    deadline = time.time() + 80
    
    # First, try known small solutions directly
    # A 3-edge solution: if we can find S of size 3 that's independent dominating
    # On P_24, minimum dominating set has size ceil(24/3) = 8
    # But we need INDEPENDENT dominating set, which is larger
    
    # Actually, let me think differently. The added edges form a structure on top of the path.
    # For diameter 4, we need every pair of vertices within distance 4.
    
    # Let's try direct construction: add edges that create shortcuts
    # Key insight: with k added edges, we can create at most k "jumps"
    # Path has 23 edges, diameter 23. Need to reduce to 4.
    
    # Try various hand-crafted and systematic solutions
    
    test_solutions = [
        # 4-edge solutions
        [(0, 6), (6, 12), (12, 18), (18, 23)],
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)],  # 5 edges
        [(1, 7), (7, 13), (13, 19), (19, 23)],
        [(0, 7), (7, 14), (14, 21), (21, 23)],
        [(2, 8), (8, 14), (14, 20), (20, 23)],
        [(0, 6), (6, 12), (12, 18), (18, 22)],
        [(1, 6), (6, 12), (12, 18), (18, 23)],
        [(0, 5), (5, 11), (11, 17), (17, 23)],
        [(0, 7), (7, 14), (14, 20), (20, 23)],
        [(2, 9), (9, 16), (16, 23), (0, 7)],
        
        # Try 3-edge solutions
        [(0, 8), (8, 16), (16, 23)],
        [(0, 7), (7, 15), (15, 23)],
        [(1, 8), (8, 16), (16, 23)],
        [(0, 9), (9, 18), (18, 23)],
        [(3, 10), (10, 17), (17, 23)],
        [(0, 8), (8, 16), (16, 22)],
        [(2, 10), (10, 18), (18, 23)],
        [(0, 7), (7, 14), (14, 21)],
        
        # 5-edge solutions for safety
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)],
        [(1, 5), (5, 10), (10, 15), (15, 20), (20, 23)],
        [(0, 4), (4, 9), (9, 14), (14, 19), (19, 23)],
        [(0, 5), (5, 11), (11, 17), (17, 23), (0, 10)],
    ]
    
    best_sol = None
    for sol in test_solutions:
        if time.time() > deadline:
            break
        # Check for duplicates with base
        sol_set = set((min(u, v), max(u, v)) for u, v in sol)
        if sol_set & base_edge_set:
            continue
        # Check for duplicates within sol
        if len(sol_set) != len(sol):
            continue
        if _valid_solution(n, base_edges, sol):
            if best_sol is None or len(sol) < len(best_sol):
                best_sol = [tuple(e) for e in sol]
    
    if best_sol is not None and len(best_sol) <= 3:
        return [[u, v] for u, v in best_sol]
    
    # Now try the IDS approach: find independent dominating sets and derive edges
    # For the path, an independent dominating set S means:
    # - No two elements of S are adjacent
    # - Every vertex is within distance 1 of some element of S
    
    # Minimum independent dominating set on P_n has size about n/4
    # But we don't need S to be minimal; we need the resulting graph to have diam <= 4
    
    # Let's try S of various sizes and see what edges they produce
    # The edges are between S-vertices at distance 2 on the path
    
    # Actually, re-reading the strategy: "adding edges between each pair of vertices in S 
    # if their path distance is exactly 2"
    # This creates a graph where S-vertices form a subgraph with edges at distance-2
    
    # Let me try a different approach: systematic search for small solutions
    
    # For diameter 4 on 24 vertices, we need good coverage
    # Let's try all 3-edge combinations from a reduced candidate set
    
    # Good candidates: edges that jump far
    far_candidates = []
    for u in range(n):
        for v in range(u + 3, n):  # distance at least 3
            if (u, v) not in base_edge_set:
                far_candidates.append((u, v))
    
    # Try all 3-edge subsets from a curated set
    curated = [
        (0, 6), (0, 7), (0, 8), (1, 7), (1, 8), (2, 8), (2, 9),
        (5, 11), (5, 12), (6, 12), (6, 13), (7, 13), (7, 14),
        (11, 17), (11, 18), (12, 18), (12, 19), (13, 19), (13, 20),
        (17, 23), (18, 23), (19, 23), (20, 23), (16, 22), (16, 23),
    ]
    
    # Filter: only opposite parity (ensures triangle-free with path)
    curated = [(u, v) for u, v in curated if u % 2 != v % 2]
    
    # Try all 3-edge combinations
    if time.time() < deadline - 5:
        for combo in combinations(curated, 3):
            if time.time() > deadline:
                break
            sol = list(combo)
            sol_set = set((min(u, v), max(u, v)) for u, v in sol)
            if len(sol_set) != 3:
                continue
            if _valid_solution(n, base_edges, sol):
                if best_sol is None or 3 < len(best_sol):
                    best_sol = sol
                break  # Found a 3-edge solution
    
    if best_sol is not None and len(best_sol) == 3:
        return [[u, v] for u, v in best_sol]
    
    # Try 4-edge combinations from curated
    if best_sol is None or len(best_sol) > 4:
        if time.time() < deadline - 10:
            for combo in combinations(curated, 4):
                if time.time() > deadline:
                    break
                sol = list(combo)
                sol_set = set((min(u, v), max(u, v)) for u, v in sol)
                if len(sol_set) != 4:
                    continue
                if _valid_solution(n, base_edges, sol):
                    if best_sol is None or 4 < len(best_sol):
                        best_sol = sol
                    break
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    # Fallback: greedy construction
    added = []
    current_edges = list(base_edges)
    added_set = set()
    
    while time.time() < deadline - 5:
        adj = _build_adj(n, current_edges)
        diam = _bfs_diameter(n, adj)
        
        if diam <= 4:
            break
        if diam == -1:
            break
        
        # Find worst pair
        worst_u, worst_v = -1, -1
        max_dist_found = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                        if dist[v] > max_dist_found:
                            max_dist_found = dist[v]
                            worst_u = s
                            worst_v = v
        
        if worst_u == -1:
            break
        
        # Find best candidate edge
        best_candidate = None
        best_diam = max_dist_found + 1
        
        for a, b in far_candidates:
            if (min(a, b), max(a, b)) in added_set:
                continue
            if a % 2 == b % 2:  # skip same parity
                continue
            adj[a].add(b)
            adj[b].add(a)
            new_diam = _bfs_diameter(n, adj)
            adj[a].remove(b)
            adj[b].remove(a)
            
            if new_diam != -1 and new_diam < best_diam:
                best_diam = new_diam
                best_candidate = (a, b)
        
        if best_candidate is None:
            break
        
        added.append(best_candidate)
        added_set.add((min(best_candidate[0], best_candidate[1]), max(best_candidate[0], best_candidate[1])))
        current_edges.append(best_candidate)
    
    if _valid_solution(n, base_edges, added):
        if best_sol is None or len(added) < len(best_sol):
            best_sol = added
    
    # Try to reduce
    if best_sol is not None:
        for i in range(len(best_sol) - 1, -1, -1):
            if time.time() > deadline:
                break
            reduced = best_sol[:i] + best_sol[i+1:]
            if _valid_solution(n, base_edges, reduced):
                best_sol = reduced
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    return []
