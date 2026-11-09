"""Find longest simple path in pinned graph using multi-strategy search with 2-opt refinement."""
import time
import random
import sys
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)


def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps


def warnsdorff_score(v, seen):
    return (len(ADJ[v] - seen), v)


def greedy_path(start):
    """Greedy Warnsdorff path from start."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: warnsdorff_score(w, seen))
        path.append(v)
        seen.add(v)
    return path


def greedy_path_max(start):
    """Greedy path preferring high-degree unvisited neighbors."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (len(ADJ[w] - seen), -w))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path):
    """Extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = ADJ[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: warnsdorff_score(w, seen))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = ADJ[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: warnsdorff_score(w, seen))
                path.append(v)
                seen.add(v)
                changed = True
    return path


def try_insert(path):
    """Try to insert unvisited vertices into the path."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in ADJ[u] and w in ADJ[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path


def two_opt_refine(path, deadline):
    """Try to extend path by reversing segments to create new endpoint opportunities."""
    path = list(path)
    best = list(path)
    n = len(path)
    for i in range(n - 2):
        if time.time() > deadline:
            break
        for j in range(i + 2, n):
            if time.time() > deadline:
                break
            new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
            if i > 0 and new_path[i] not in ADJ[new_path[i-1]]:
                continue
            if j < n - 1 and new_path[j] not in ADJ[new_path[j+1]]:
                continue
            valid = True
            for k in range(i+1, j+1):
                if new_path[k] not in ADJ[new_path[k-1]]:
                    valid = False
                    break
            if not valid:
                continue
            extended = extend_both_ends(new_path)
            if len(extended) > len(best):
                best = extended
    return best


def refine(path, deadline):
    """Full refinement pipeline."""
    path = extend_both_ends(path)
    path = try_insert(path)
    path = extend_both_ends(path)
    path2 = two_opt_refine(path, deadline)
    if len(path2) > len(path):
        path = path2
    path = extend_both_ends(path)
    path = try_insert(path)
    path = extend_both_ends(path)
    return path


def iddfs_path(start, deadline, target_len=None):
    """Iterative deepening DFS with Warnsdorff heuristic."""
    best_path = [start]
    best_len = 1

    def dfs(cur, path, seen, remaining):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        if target_len and best_len >= target_len:
            return
        if remaining <= 0:
            return
        candidates = ADJ[cur] - seen
        if not candidates:
            return
        candidates = sorted(candidates, key=lambda w: warnsdorff_score(w, seen))
        for v in candidates:
            path.append(v)
            seen.add(v)
            dfs(v, path, seen, remaining - 1)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return

    depths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 99]
    for depth in depths:
        if time.time() > deadline:
            break
        seen = {start}
        dfs(start, [start], seen, depth)
        if best_len >= N:
            break
    return best_path


def limited_backtracking(start, target_len, deadline):
    """Time-limited DFS trying to find a path of target_len or longer."""
    best_path = [start]
    best_len = 1

    def dfs(path, seen):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        if best_len >= target_len:
            return
        cand = ADJ[path[-1]] - seen
        if not cand:
            return
        cand_sorted = sorted(cand, key=lambda w: warnsdorff_score(w, seen))
        for v in cand_sorted:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return

    dfs([start], {start})
    return best_path


def forced_move_path(start, rng, mode):
    """Path construction with forced moves and heuristic choices."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = ADJ[cur] - seen
        if not candidates:
            break
        if len(candidates) == 1:
            v = list(candidates)[0]
        elif mode == 'warns':
            v = min(candidates, key=lambda w: warnsdorff_score(w, seen))
        elif mode == 'max':
            v = max(candidates, key=lambda w: (len(ADJ[w] - seen), -w))
        else:
            v = rng.choice(list(candidates))
        path.append(v)
        seen.add(v)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]

    best_path = []
    best_len = 0

    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)

        # Phase 1: Greedy from every vertex with multiple strategies
        for start in comp_list:
            if time.time() > deadline - 20:
                break
            # Try Warnsdorff greedy
            p = greedy_path(start)
            p = refine(p, deadline)
            if len(p) > best_len:
                best_path, best_len = p, len(p)
            # Try max-degree greedy
            p = greedy_path_max(start)
            p = refine(p, deadline)
            if len(p) > best_len:
                best_path, best_len = p, len(p)
            if best_len >= comp_size:
                break

        if best_len >= comp_size:
            break

        # Phase 2: IDDFS from promising starts
        if time.time() < deadline - 15:
            deg_sorted = sorted(comp_list, key=lambda v: len(ADJ[v]), reverse=True)
            # Focus on high-degree vertices and endpoints of current best
            promising = list(deg_sorted[:15])
            if best_path:
                for ep in [best_path[0], best_path[-1], best_path[len(best_path)//2]]:
                    if ep not in promising and ep in comp:
                        promising.append(ep)

            for start in promising:
                if time.time() > deadline - 15:
                    break
                if best_len >= comp_size:
                    break
                dl = min(deadline - 15, time.time() + 3.0)
                p = iddfs_path(start, dl, comp_size)
                p = refine(p, deadline)
                if len(p) > best_len:
                    best_path, best_len = p, len(p)

        # Phase 3: Limited backtracking from most promising starts
        if time.time() < deadline - 5:
            deg_sorted = sorted(comp_list, key=lambda v: len(ADJ[v]), reverse=True)
            for start in deg_sorted[:10]:
                if time.time() > deadline - 5:
                    break
                if best_len >= comp_size:
                    break
                remaining = deadline - 5 - time.time()
                if remaining < 0.5:
                    break
                budget = min(2.5, remaining * 0.4)
                dl = time.time() + budget
                p = limited_backtracking(start, comp_size, dl)
                p = refine(p, deadline)
                if len(p) > best_len:
                    best_path, best_len = p, len(p)

        # Phase 4: Randomized restarts with forced moves
        if time.time() < deadline - 2:
            modes = ['warns', 'max', 'rand']
            rng = random.Random(748)
            for _ in range(20000):
                if time.time() > deadline - 2:
                    break
                start = rng.choice(comp_list)
                mode = modes[rng.randrange(len(modes))]
                p = forced_move_path(start, rng, mode)
                p = refine(p, deadline)
                if len(p) > best_len:
                    best_path, best_len = p, len(p)
                if best_len >= comp_size:
                    break

        # Phase 5: Final intensive refinement
        if best_path and time.time() < deadline - 1:
            for _ in range(3):
                if time.time() > deadline - 1:
                    break
                extended = extend_both_ends(best_path)
                if len(extended) > best_len:
                    best_path, best_len = extended, len(extended)
                extended = try_insert(best_path)
                if len(extended) > best_len:
                    best_path, best_len = extended, len(extended)
                extended = two_opt_refine(best_path, deadline)
                if len(extended) > best_len:
                    best_path, best_len = extended, len(extended)
                extended = extend_both_ends(best_path)
                if len(extended) > best_len:
                    best_path, best_len = extended, len(extended)

    if not best_path:
        best_path = [0]

    # Validate
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            best_path = [0]
            break
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        best_path = fixed

    return best_path
