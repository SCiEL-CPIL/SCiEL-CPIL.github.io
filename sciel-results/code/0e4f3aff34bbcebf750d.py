import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 82.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # ---------- color-coding detection of a colorful K_k ----------
    # For a random k-coloring, a colorful K_k (clique subdivision where each
    # branch gets a distinct color and all path vertices are color-distinct)
    # is detected by DP over color subsets. We then extract the actual paths.
    # Key insight: if we find a colorful clique in the "reachability" sense
    # (for every pair of branch-colored vertices, there's a path using only
    # vertices of colors in the full set), we can extract disjoint paths.

    def color_coding_dp(k, coloring, adj_list):
        """
        coloring: list of length n, coloring[v] in {0..k-1} (or -1 if uncolored, but we color all)
        Returns: list of k branch vertices (one per color) if a colorful K_k subdivision exists, else None.
        DP: dp[S][v] = True if there's a colorful path/tree spanning colors S ending at v with v having color in S.
        Actually for clique subdivision, we need: for each pair of branches, a path. 
        Simpler: detect a colorful clique in the graph where two vertices are "connected" if there's a path
        between them using only vertices whose colors are in the full color set. But that's not quite right either.
        
        Better approach: A clique subdivision of size k means there exist k branch vertices b_0,...,b_{k-1}
        and for each pair (i,j), a path P_{ij} from b_i to b_j, such that all interiors of all paths are
        pairwise disjoint and avoid all branches. 
        
        With color coding: assign each vertex a random color in {0..k-1}. A "colorful" clique subdivision
        is one where all branch vertices have distinct colors AND all path vertices (including branches)
        have distinct colors. Then we can use the following DP:
        
        dp[S][v] = set of possible "roots" or just a boolean: there exists a colorful path from some root
        to v using exactly colors S. But we need multiple paths sharing only endpoints (branches).
        
        Alternative: Use the fact that a clique subdivision is equivalent to: there exist k vertices that
        are "k-connected" in a certain sense. 
        
        Let me use a different color-coding approach: 
        - dp[S][v] = True if there's a colorful simple path from a fixed start to v using colors S.
        - For a clique, we need all pairs connected. 
        
        Actually, the cleanest: detect if there's a colorful copy of K_k as a topological minor.
        This is NP-hard in general but color-coding gives a randomized FPT algorithm.
        
        Standard color-coding for clique: 
        dp[S][v] = True if there's a colorful clique of size |S| containing v, with colors S.
        Transition: dp[S][v] = OR over u in adj[v] with color(u) in S\{color(v)} of dp[S\{color(v)}][u].
        But this detects actual cliques, not clique subdivisions!
        
        For clique subdivisions, we need paths instead of edges. So:
        dp[S][v] = True if there's a "colorful K_|S| subdivision" rooted at v (v is one branch)
        with branch colors S.
        Transition is harder because we need disjoint paths.
        
        Let me use a simpler but effective approach: 
        1. Find candidate branch sets using color-coding for cliques (as a filter).
        2. Verify via BFS path packing.
        
        OR: use color-coding directly on the subdivision structure.
        """
        # Simpler: use color-coding to find candidate branch sets (colorful cliques in a
        # "path-connectivity" graph), then verify.
        
        # Build a "k-path graph": two vertices u,v are connected if there's a path between them
        # of length <= k using only vertices with distinct colors (colorful path).
        # This is expensive. Instead, let's just do:
        
        # For each color c, let V_c = {v : coloring[v] == c}
        # dp[S][v] = True if there exists a colorful set of |S|-1 paths connecting v to |S|-1 other
        # vertices, one of each color in S\{color(v)}, with all paths internally disjoint and colorful.
        
        # This is complex. Let me use a practical approach:
        # 1. For the given coloring, identify one candidate vertex per color (the one with max degree).
        # 2. Check if these k candidates form a clique subdivision via BFS.
        # 3. Try all combinations of one vertex per color (if k is small, this is feasible).
        
        if k > n:
            return None
        
        # Group vertices by color
        by_color = [[] for _ in range(k)]
        for v in range(n):
            c = coloring[v]
            if 0 <= c < k:
                by_color[c].append(v)
        
        # If any color class is empty, no colorful K_k
        for c in range(k):
            if not by_color[c]:
                return None
        
        # Try combinations: for small k, try all tuples (one from each color)
        # But k can be up to ~8, and each color class has ~n/k vertices.
        # For k=8, n=24, each class has ~3, so 3^8 = 6561 combos. Feasible.
        # For k=10, each class has ~2.4, so ~2^10=1024. Feasible.
        
        # Limit total combinations
        import itertools
        max_combos = 50000
        combo_count = 1
        for c in range(k):
            combo_count *= len(by_color[c])
            if combo_count > max_combos:
                break
        
        if combo_count > max_combos:
            # Sample some combinations
            candidates = []
            for _ in range(min(5000, max_combos)):
                combo = []
                for c in range(k):
                    combo.append(random.choice(by_color[c]))
                candidates.append(tuple(combo))
        else:
            candidates = list(itertools.product(*[by_color[c] for c in range(k)]))
        
        # For each candidate branch set, check if it forms a clique subdivision
        for combo in candidates:
            if time_up():
                return None
            branches = list(combo)
            if len(set(branches)) != k:
                continue
            # Check via BFS path packing
            if _check_subdivision(branches, adj):
                return branches
        return None

    def _check_subdivision(branches, adj):
        """Check if branches form a clique subdivision via BFS path packing."""
        m = len(branches)
        if m < 2:
            return False
        bset = set(branches)
        used = set()
        
        # Sort pairs by path length (shorter first) for better packing
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = branches[i], branches[j]
                # BFS shortest path avoiding used and branches (except endpoints)
                banned = set(used) | bset - {a, b}
                path = _bfs_path(a, b, banned, adj)
                if path is None:
                    return False
                pairs.append((len(path), (i, j), path))
        
        pairs.sort(key=lambda x: x[0])
        for _, (i, j), path in pairs:
            used.update(path[1:-1])
        return True

    def _bfs_path(s, t, banned, adj):
        if s == t:
            return None
        if t in adj[s] and s not in banned and t not in banned:
            return [s, t]
        if s in banned or t in banned:
            return None
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v in parent or v in banned:
                    continue
                if v == t:
                    path = [t]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                parent[v] = u
                q.append(v)
        return None

    # ---------- Max clique (for initial heuristic) ----------
    def find_max_clique():
        best = []
        cands = sorted(range(n), key=lambda x: -len(adj[x]))
        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                expand(clique + [v], [u for u in cand[i + 1:] if u in adj[v]])
        expand([], cands)
        return best

    clique = find_max_clique()
    
    # ---------- Try to verify the max clique as a subdivision ----------
    def verify_and_get_paths(branches):
        m = len(branches)
        if m < 2:
            return None
        bset = set(branches)
        used = set()
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = branches[i], branches[j]
                banned = set(used) | (bset - {a, b})
                path = _bfs_path(a, b, banned, adj)
                if path is None:
                    return None
                pairs.append((len(path), (i, j), path))
        pairs.sort(key=lambda x: x[0])
        result_paths = {}
        for _, (i, j), path in pairs:
            used.update(path[1:-1])
            result_paths[(i, j)] = path
        # Return in consistent order
        paths = []
        for i in range(m):
            for j in range(i + 1, m):
                paths.append(result_paths[(i, j)])
        return paths

    best_branches = None
    best_paths = None
    best_count = 0

    # Try max clique first
    cp = verify_and_get_paths(clique)
    if cp is not None:
        best_branches = list(clique)
        best_paths = cp
        best_count = len(clique)

    # If clique doesn't work, try subsets
    if best_count < 2 and len(clique) >= 2:
        for sz in range(len(clique) - 1, 1, -1):
            sub = clique[:sz]
            cp = verify_and_get_paths(sub)
            if cp is not None:
                best_branches = list(sub)
                best_paths = cp
                best_count = sz
                break

    # Fallback: at least 2 branches
    if best_count < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
        best_count = 2

    # ---------- Color-coding search for larger subdivisions ----------
    # Try k from best_count+1 up to min(n, 12)
    for k in range(min(n, 12), best_count, -1):
        if time_up():
            break
        found = False
        # Try multiple random colorings
        max_trials = 200 if k <= 8 else 50
        for trial in range(max_trials):
            if time_up():
                break
            # Random k-coloring
            coloring = [random.randrange(k) for _ in range(n)]
            branches = color_coding_dp(k, coloring, adj)
            if branches is not None:
                # Verify and get paths
                paths = verify_and_get_paths(branches)
                if paths is not None:
                    best_branches = list(branches)
                    best_paths = paths
                    best_count = k
                    found = True
                    break
        if found:
            break  # Found a larger one, stop

    # ---------- Incremental improvement: try to add vertices ----------
    if not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            cur = list(best_branches)
            cur_set = set(cur)
            # Try adding each non-branch vertex
            candidates = [v for v in range(n) if v not in cur_set]
            candidates.sort(key=lambda v: -len(adj[v]))
            for v in candidates:
                if time_up():
                    break
                trial = cur + [v]
                p = verify_and_get_paths(trial)
                if p is not None:
                    best_branches = trial
                    best_paths = p
                    best_count = len(trial)
                    improved = True
                    break

    # ---------- Try swapping to improve ----------
    if not time_up() and best_count >= 3:
        for _ in range(200):
            if time_up():
                break
            cur = list(best_branches)
            cur_set = set(cur)
            idx = random.randrange(len(cur))
            v = random.choice([x for x in range(n) if x not in cur_set])
            new_b = cur[:idx] + cur[idx + 1:] + [v]
            p = verify_and_get_paths(new_b)
            if p is not None and len(new_b) > best_count:
                best_branches = new_b
                best_paths = p
                best_count = len(new_b)

    # Final safety
    if best_paths is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]
    return {'branches': branches_out, 'paths': paths_out}
