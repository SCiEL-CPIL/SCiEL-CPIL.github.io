"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

import time
import random
from collections import deque

def _build_adj():
    a = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        a[u].add(v)
        a[v].add(u)
    return a

def _find_largest_component(adj):
    visited = set()
    components = []
    for start in range(P['n']):
        if start not in visited:
            comp = []
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in adj[v]:
                    if u not in visited:
                        visited.add(u)
                        stack.append(u)
            components.append(comp)
    return max(components, key=len) if components else []

def _bipartite_matching(adj, vertices):
    """
    Compute a maximum matching in a bipartite graph where:
    - Left side: each vertex v has two 'slots' (v_L, v_R) representing its two path-neighbor positions
    - Right side: actual vertices
    - Edge (v_L, u) exists if u is adjacent to v
    This helps identify which vertices can serve as endpoints (only one slot matched)
    vs internal vertices (both slots matched).
    """
    if not vertices:
        return {}, {}
    
    vset = set(vertices)
    # Build bipartite graph: left nodes are (v, slot) for slot in {0,1}, right nodes are vertices
    # We want to find a matching that maximizes the number of matched slots
    # A vertex v is an endpoint candidate if only one of its slots is matched
    
    left_nodes = []
    left_to_right = {}
    for v in vertices:
        left_nodes.append((v, 0))
        left_nodes.append((v, 1))
        left_to_right[(v, 0)] = list(adj[v] & vset)
        left_to_right[(v, 1)] = list(adj[v] & vset)
    
    # Hopcroft-Karp style matching
    match_r = {}  # right node -> left node
    match_l = {}  # left node -> right node
    
    def bfs():
        dist = {}
        queue = deque()
        for ln in left_nodes:
            if ln not in match_l:
                dist[ln] = 0
                queue.append(ln)
            else:
                dist[ln] = float('inf')
        found = False
        while queue:
            ln = queue.popleft()
            for rn in left_to_right[ln]:
                if rn not in match_r:
                    found = True
                elif (match_r[rn] not in dist) or dist.get(match_r[rn], float('inf')) > dist[ln] + 1:
                    dist[match_r[rn]] = dist[ln] + 1
                    queue.append(match_r[rn])
        return dist, found
    
    def dfs(ln, dist):
        for rn in left_to_right[ln]:
            if rn not in match_r or (match_r[rn] in dist and dist[match_r[rn]] == dist[ln] + 1 and dfs(match_r[rn], dist)):
                match_l[ln] = rn
                match_r[rn] = ln
                return True
        dist[ln] = float('inf')
        return False
    
    while True:
        dist, found = bfs()
        if not found:
            break
        for ln in left_nodes:
            if ln not in match_l:
                dfs(ln, dist)
    
    # Determine endpoint candidates: vertices with only one slot matched
    endpoint_candidates = set()
    for v in vertices:
        matched_count = sum(1 for s in [0, 1] if (v, s) in match_l)
        if matched_count == 1:
            endpoint_candidates.add(v)
    
    return match_l, endpoint_candidates

def _matching_slack(adj, vertex, seen):
    """
    Compute a measure of 'slack' for a vertex: how many unvisited neighbors it has.
    Lower slack means more constrained, should be visited earlier.
    """
    return len(adj[vertex] - seen)

def _greedy_path(adj, start, strategy, rng=None, deadline=None):
    """Greedy path extension with different neighbor-selection strategies."""
    path = [start]
    seen = {start}
    while True:
        if deadline and time.time() > deadline:
            break
        cands = adj[path[-1]] - seen
        if not cands:
            break
        if strategy == 'min_deg':
            v = min(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(cands, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(cands))
        elif strategy == 'slack_aware':
            # Prefer vertices with fewer unvisited neighbors (more constrained)
            # but also consider degree in the remaining graph
            def slack_score(w):
                slack = len(adj[w] - seen)
                # Penalize if w has very few options
                return (slack, len(adj[w]), w)
            v = min(cands, key=slack_score)
        elif strategy == 'endpoint_aware':
            # Try to leave high-degree vertices as potential endpoints
            # Visit low-degree vertices first
            v = min(cands, key=lambda w: (len(adj[w]), len(adj[w] - seen), w))
        else:
            v = min(cands)
        path.append(v)
        seen.add(v)
    return path

def _backtrack_search(adj, start, max_depth, rng, deadline, endpoint_hints=None):
    """
    DFS with backtracking, using matching-based endpoint hints and lookahead to guide choices.
    """
    best_path = [start]
    n = P['n']
    endpoint_set = set(endpoint_hints) if endpoint_hints else set()
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = path[:]
        if depth >= max_depth:
            return
        cands = list(adj[path[-1]] - seen)
        if not cands:
            return
        # Lookahead: estimate how far we can extend from each candidate
        def score(w):
            slack = len(adj[w] - seen)
            is_endpoint = 1 if w in endpoint_set else 0
            progress = depth / max_depth
            endpoint_penalty = is_endpoint * (1.0 - progress) * 2
            # Lookahead: count unvisited neighbors of w's unvisited neighbors
            lookahead = 0
            for u in adj[w] - seen:
                lookahead += len(adj[u] - seen)
            # Prefer vertices that lead to more unvisited territory
            return (slack + endpoint_penalty - lookahead * 0.1, w)
        cands.sort(key=score)
        for v in cands:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def _extend_from_both_ends(adj, path, deadline):
    """Try to extend a path from both ends using min-degree heuristic."""
    if not path:
        return path
    result = list(path)
    seen = set(result)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try extending the tail
        cands = adj[result[-1]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.append(v)
            seen.add(v)
            improved = True
        # Try extending the head
        cands = adj[result[0]] - seen
        if cands:
            v = min(cands, key=lambda w: len(adj[w] - seen))
            result.insert(0, v)
            seen.add(v)
            improved = True
    return result

def _splice_paths(adj, path1, path2, deadline):
    """
    Try to splice two paths together by finding a vertex that connects an endpoint
    of path1 to an endpoint of path2.
    """
    if not path1 or not path2:
        return path1 if path1 else path2
    
    seen1 = set(path1)
    seen2 = set(path2)
    
    # Try all combinations of endpoints
    for e1 in [path1[0], path1[-1]]:
        for e2 in [path2[0], path2[-1]]:
            if e1 in seen2 or e2 in seen1:
                continue
            if e2 in adj[e1]:
                # Direct connection
                if path1[0] == e1:
                    combined = path2 + path1
                else:
                    combined = path1 + path2
                # Remove duplicates
                seen_r = set()
                unique = []
                for x in combined:
                    if x not in seen_r:
                        unique.append(x)
                        seen_r.add(x)
                return unique
    
    # Try to find a bridge vertex w such that e1-w-e2 forms a path
    for e1 in [path1[0], path1[-1]]:
        for e2 in [path2[0], path2[-1]]:
            if e1 in seen2 or e2 in seen1:
                continue
            common = (adj[e1] & adj[e2]) - seen1 - seen2
            if common:
                w = min(common, key=lambda x: len(adj[x]))
                if path1[0] == e1:
                    combined = path2 + [w] + path1
                else:
                    combined = path1 + [w] + path2
                seen_r = set()
                unique = []
                for x in combined:
                    if x not in seen_r:
                        unique.append(x)
                        seen_r.add(x)
                return unique
    
    return path1 if len(path1) > len(path2) else path2

def _two_end_grow(adj, start1, start2, rng, deadline):
    """Grow path from two ends inward."""
    if start1 == start2:
        return [start1]
    path1 = [start1]
    path2 = [start2]
    seen = {start1, start2}
    
    while True:
        if time.time() > deadline:
            break
        # Try to extend end 1
        cands1 = adj[path1[-1]] - seen
        if cands1:
            v = min(cands1, key=lambda w: len(adj[w] - seen))
            path1.append(v)
            seen.add(v)
        else:
            # Try to extend end 2
            cands2 = adj[path2[-1]] - seen
            if cands2:
                v = min(cands2, key=lambda w: len(adj[w] - seen))
                path2.append(v)
                seen.add(v)
            else:
                break
    
    # Combine: path1 forward + path2 reversed
    result = path1 + path2[::-1]
    seen_r = set()
    unique = []
    for x in result:
        if x not in seen_r:
            unique.append(x)
            seen_r.add(x)
    return unique

def _two_opt_improve(adj, path, deadline):
    """
    Try 2-opt style improvements: remove an internal vertex and try to reinsert it
    elsewhere in the path to extend it.
    """
    if len(path) < 3:
        return path
    
    best_path = path[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, len(best_path) - 1):
            if time.time() > deadline:
                return best_path
            v = best_path[i]
            # Remove v from position i
            candidate = best_path[:i] + best_path[i+1:]
            seen = set(candidate)
            # Try to insert v at various positions
            for j in range(len(candidate) + 1):
                if j == 0:
                    # Insert at front
                    if j < len(candidate) and candidate[j] in adj[v]:
                        new_path = [v] + candidate
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            improved = True
                            break
                elif j == len(candidate):
                    # Insert at end
                    if candidate[-1] in adj[v]:
                        new_path = candidate + [v]
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            improved = True
                            break
                else:
                    # Insert in middle: need candidate[j-1] - v - candidate[j]
                    if candidate[j-1] in adj[v] and candidate[j] in adj[v]:
                        new_path = candidate[:j] + [v] + candidate[j:]
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            improved = True
                            break
    return best_path

def _insert_vertex(adj, path, v, deadline):
    """Try to insert vertex v into path at the best position."""
    if v in path:
        return path
    seen = set(path)
    if not (adj[v] - seen):
        return path
    
    best_path = path[:]
    best_len = len(path)
    
    for i in range(len(path) + 1):
        if time.time() > deadline:
            break
        if i == 0:
            if path[0] in adj[v]:
                candidate = [v] + path
                if len(candidate) > best_len:
                    best_path = candidate
                    best_len = len(candidate)
        elif i == len(path):
            if path[-1] in adj[v]:
                candidate = path + [v]
                if len(candidate) > best_len:
                    best_path = candidate
                    best_len = len(candidate)
        else:
            if path[i-1] in adj[v] and path[i] in adj[v]:
                candidate = path[:i] + [v] + path[i:]
                if len(candidate) > best_len:
                    best_path = candidate
                    best_len = len(candidate)
    
    return best_path

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    adj = _build_adj()
    best = []
    
    # Find the largest component
    largest_comp = _find_largest_component(adj)
    if not largest_comp:
        return list(range(P['n']))[:1]
    comp_set = set(largest_comp)
    
    # Step 1: Bipartite matching to identify endpoint candidates
    match_l, endpoint_candidates = _bipartite_matching(adj, largest_comp)
    
    # Step 2: Strategy 1 - Deterministic greedy from every start in largest component
    for start in largest_comp:
        path = _greedy_path(adj, start, 'min_deg')
        if len(path) > len(best):
            best = path
    
    # Step 3: Strategy 2 - Randomized restarts with different heuristics
    rng = random.Random(42)
    strategies = ['min_deg', 'max_deg', 'random', 'slack_aware', 'endpoint_aware']
    
    for strat in strategies:
        for start in largest_comp:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
    
    # Step 4: Strategy 3 - Backtracking search with endpoint hints (increased depth)
    degs = [(len(adj[i]), i) for i in largest_comp]
    degs.sort(reverse=True)
    
    # Try backtracking from high-degree vertices with deeper search
    for deg, start in degs[:25]:
        if time.time() > deadline:
            break
        max_depth = min(len(largest_comp), 50)
        path = _backtrack_search(adj, start, max_depth, rng, deadline, endpoint_hints=endpoint_candidates)
        if len(path) > len(best):
            best = path
    
    # Step 5: Strategy 4 - Try to extend the best path found so far
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
    
    # Step 5.5: 2-opt improvement
    if best and time.time() < deadline:
        improved = _two_opt_improve(adj, best[:], deadline)
        if len(improved) > len(best):
            best = improved
    
    # Step 6: Strategy 5 - Two-end growth from pairs of endpoint candidates
    ep_list = list(endpoint_candidates) if endpoint_candidates else [d for _, d in degs[:10]]
    if len(ep_list) >= 2:
        for i in range(len(ep_list)):
            for j in range(i + 1, min(i + 10, len(ep_list))):
                if time.time() > deadline:
                    break
                path = _two_end_grow(adj, ep_list[i], ep_list[j], rng, deadline)
                if len(path) > len(best):
                    best = path
                # Try 2-opt on the grown path
                if len(path) > len(best):
                    improved = _two_opt_improve(adj, path[:], deadline)
                    if len(improved) > len(best):
                        best = improved
    
    # Step 7: Strategy 6 - More randomized searches with splicing
    if time.time() < deadline:
        partial_paths = []
        for _ in range(50):
            if time.time() > deadline:
                break
            start = rng.choice(largest_comp)
            strat = rng.choice(strategies)
            path = _greedy_path(adj, start, strat, rng)
            if len(path) > len(best):
                best = path
            if len(path) >= 5:
                partial_paths.append(path)
        
        # Try to splice partial paths
        for i in range(len(partial_paths)):
            for j in range(i + 1, len(partial_paths)):
                if time.time() > deadline:
                    break
                spliced = _splice_paths(adj, partial_paths[i], partial_paths[j], deadline)
                if len(spliced) > len(best):
                    best = spliced
                    # Try 2-opt on spliced path
                    if time.time() < deadline:
                        improved = _two_opt_improve(adj, best[:], deadline)
                        if len(improved) > len(best):
                            best = improved
    
    # Step 8: Final backtracking attempts with different seeds and deeper search
    for seed in [123, 456, 789, 101, 202, 303]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        for deg, start in degs[:15]:
            if time.time() > deadline:
                break
            max_depth = min(len(largest_comp), 45)
            path = _backtrack_search(adj, start, max_depth, rng2, deadline, endpoint_hints=endpoint_candidates)
            if len(path) > len(best):
                best = path
                # Try 2-opt
                if time.time() < deadline:
                    improved = _two_opt_improve(adj, best[:], deadline)
                    if len(improved) > len(best):
                        best = improved
    
    # Step 9: Final extension attempt
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
        # 2-opt after extension
        if time.time() < deadline:
            improved = _two_opt_improve(adj, best[:], deadline)
            if len(improved) > len(best):
                best = improved
    
    # Step 10: Try all starts with random strategy one more time if time permits
    for start in largest_comp:
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, 'random', rng)
            if len(path) > len(best):
                best = path
    
    # Step 11: Try to insert unvisited vertices into the best path
    if best and time.time() < deadline:
        visited = set(best)
        for v in largest_comp:
            if time.time() > deadline:
                break
            if v not in visited:
                new_path = _insert_vertex(adj, best[:], v, deadline)
                if len(new_path) > len(best):
                    best = new_path
    
    # Final extension and 2-opt
    if best and time.time() < deadline:
        extended = _extend_from_both_ends(adj, best[:], deadline)
        if len(extended) > len(best):
            best = extended
        if time.time() < deadline:
            improved = _two_opt_improve(adj, best[:], deadline)
            if len(improved) > len(best):
                best = improved
    
    return best
# EVOLVE-BLOCK-END
