"""Structural approach: multi-start path construction + path merging + detour insertion."""
import random
import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)


def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps


def bfs_farthest(start, comp):
    dist = {start: 0}
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in ADJ[u]:
            if v in comp and v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    farthest = max(dist, key=lambda x: dist[x])
    return farthest, parent, dist


def reconstruct_path(parent, target):
    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def diameter_path(comp):
    comp_list = sorted(comp)
    if not comp_list:
        return []
    if len(comp_list) == 1:
        return comp_list
    f1, _, _ = bfs_farthest(comp_list[0], comp)
    f2, parent2, _ = bfs_farthest(f1, comp)
    return reconstruct_path(parent2, f2)


def greedy_extend(path, comp):
    path = list(path)
    seen = set(path)
    improved = True
    while improved:
        improved = False
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def detour_insert(path, comp, deadline):
    path = list(path)
    seen = set(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return path
            a, b = path[i], path[i + 1]
            forbidden = seen - {a, b}
            parent = {a: None}
            queue = deque([a])
            found = False
            while queue and not found:
                u = queue.popleft()
                if u == b:
                    found = True
                    break
                for v in ADJ[u]:
                    if v not in parent and v not in forbidden:
                        parent[v] = u
                        queue.append(v)
            if found:
                detour = [b]
                while parent[detour[-1]] is not None:
                    detour.append(parent[detour[-1]])
                detour.reverse()
                if len(detour) > 2:
                    new_path = path[:i] + detour + path[i + 2:]
                    if len(set(new_path)) == len(new_path):
                        valid = all(new_path[k+1] in ADJ[new_path[k]] for k in range(len(new_path)-1))
                        if valid:
                            path = new_path
                            seen = set(path)
                            improved = True
                            break
    return path


def merge_paths(paths, comp, deadline):
    paths = sorted(paths, key=len, reverse=True)
    if not paths:
        return []
    best = paths[0][:]
    best_seen = set(best)
    for other in paths[1:]:
        if time.time() > deadline:
            break
        other_seen = set(other)
        if other_seen & best_seen:
            continue
        for b_end in [best[0], best[-1]]:
            if time.time() > deadline:
                break
            for o_end in [other[0], other[-1]]:
                if o_end in ADJ[b_end]:
                    new_path = best + other
                    if len(set(new_path)) == len(new_path):
                        valid = all(new_path[k+1] in ADJ[new_path[k]] for k in range(len(new_path)-1))
                        if valid:
                            best = new_path
                            best_seen = set(best)
                            break
            if len(best) > len(paths[0]):
                break
        if len(best) >= len(comp):
            break
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 75

    comps = find_components()
    comps.sort(key=len, reverse=True)

    best_path = []

    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= len(best_path):
            continue

        candidates = []

        diam = diameter_path(comp)
        candidates.append(diam)

        for start in sorted(comp):
            if time.time() > deadline:
                break
            seen = {start}
            p = [start]
            while True:
                cand = ADJ[p[-1]] - seen
                if not cand:
                    break
                v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
                p.append(v)
                seen.add(v)
            candidates.append(p)

        extended = []
        for p in candidates:
            if time.time() > deadline:
                break
            ep = greedy_extend(p, comp)
            extended.append(ep)

        merged = merge_paths(extended, comp, deadline)
        if len(merged) > 0:
            extended.append(merged)

        p_best = max(extended, key=len) if extended else []

        for _ in range(5):
            if time.time() > deadline:
                break
            p_new = detour_insert(p_best, comp, deadline)
            if len(p_new) > len(p_best):
                p_best = p_new
            else:
                break

        if len(p_best) > len(best_path):
            best_path = p_best

        if len(best_path) >= comp_size:
            break

    if not best_path:
        best_path = [0]

    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed

    return best_path
