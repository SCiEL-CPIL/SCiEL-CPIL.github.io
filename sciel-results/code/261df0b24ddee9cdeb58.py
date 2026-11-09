"""
Longest simple path in G(100, 1.8/100) with seed 748.
Uses a combination of:
1. Component analysis to isolate the largest connected component.
2. Initial path construction via a greedy heuristic with backtracking.
3. Iterative deepening DFS with Warnsdorff heuristic to extend the path.
4. 2-opt local search to improve the path.
"""

import time
import random
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18],
         [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13],
         [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18],
         [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76],
         [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52],
         [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55],
         [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70],
         [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42],
         [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48],
         [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78],
         [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80],
         [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81],
         [69, 90], [70, 92], [74, 85], [75, 82]]

# Build adjacency as sorted lists + set for O(1) membership
ADJ_LIST = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ_LIST[u].add(v)
    ADJ_LIST[v].add(u)
ADJ_SORTED = [sorted(s) for s in ADJ_LIST]
ADJ_SET = [set(s) for s in ADJ_LIST]
DEG = [len(s) for s in ADJ_SET]

def components():
    """Return list of components (each a set), sorted by size desc."""
    seen = set()
    comps = []
    for s in range(N):
        if s in seen:
            continue
        comp = set()
        stack = [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in ADJ_SET[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps

def is_tree(comp):
    """A connected component is a tree iff edges == |comp|-1."""
    edges = 0
    for u in comp:
        edges += len(ADJ_SET[u])
    edges //= 2
    return edges == len(comp) - 1

def leaf_order_path(comp):
    """For a tree component, peel leaves in degree-1 order to build a valid simple
    path covering all vertices. Returns the path or None if not linearly peelable."""
    if not comp:
        return None
    remaining = set(comp)
    deg = {u: len(ADJ_SET[u] & remaining) for u in remaining}
    # Find current leaves
    order = []
    # Start from a leaf.
    leaves = [u for u in remaining if deg[u] <= 1]
    if not leaves:
        return None
    start = min(leaves)
    path = [start]
    pathset = {start}
    while len(path) < len(comp):
        cur = path[-1]
        # candidate neighbors not in path
        cand = [v for v in ADJ_SET[cur] if v not in pathset]
        if not cand:
            return None
        # Prefer the candidate whose removal keeps the unvisited subgraph connected.
        unvisited = comp - pathset
        chosen = None
        for v in cand:
            # removing v from unvisited: is unvisited-{v} still connected to path?
            rest = unvisited - {v}
            if not rest:
                chosen = v
                break
            # BFS from a neighbor of v that stays in rest, or from a path vertex
            # adjacent to rest.
            seed = None
            for w in ADJ_SET[v]:
                if w in rest:
                    seed = w
                    break
            if seed is None:
                continue
            # BFS over rest
            seenr = {seed}
            st = [seed]
            while st:
                x = st.pop()
                for y in ADJ_SET[x]:
                    if y in rest and y not in seenr:
                        seenr.add(y)
                        st.append(y)
            if len(seenr) == len(rest):
                chosen = v
                break
        if chosen is None:
            # fallback: take first candidate (may produce dead end -> None)
            chosen = cand[0]
        path.append(chosen)
        pathset.add(chosen)
    # verify
    for i in range(len(path) - 1):
        if path[i + 1] not in ADJ_SET[path[i]]:
            return None
    return path

def warnsdorff_extend(path, pathset, max_len, deadline, best):
    """Greedy Warnsdorff extension from both ends, updating best in place."""
    path = list(path)
    pathset = set(path)
    improved = True
    while improved:
        improved = False
        # extend front
        while True:
            cur = path[0]
            cand = [v for v in ADJ_SET[cur] if v not in pathset]
            if not cand:
                break
            # Warnsdorff: fewest onward options first
            def score(w):
                return sum(1 for z in ADJ_SET[w] if z not in pathset)
            v = min(cand, key=lambda w: (score(w), w))
            path.insert(0, v)
            pathset.add(v)
            improved = True
        # extend back
        while True:
            cur = path[-1]
            cand = [v for v in ADJ_SET[cur] if v not in pathset]
            if not cand:
                break
            def score(w):
                return sum(1 for z in ADJ_SET[w] if z not in pathset)
            v = min(cand, key=lambda w: (score(w), w))
            path.append(v)
            pathset.add(v)
            improved = True
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    return path

def two_ends_extend(path, comp, deadline):
    """Extend the path from both ends greedily until stuck or deadline."""
    path = list(path)
    pathset = set(path)
    improved = True
    while improved:
        improved = False
        # Extend from the end
        if path:
            cur = path[-1]
            cand = [v for v in ADJ_SET[cur] if v not in pathset and v in comp]
            if cand:
                # Warnsdorff: fewest onward options first
                def score(w):
                    return sum(1 for z in ADJ_SET[w] if z not in pathset and z in comp)
                v = min(cand, key=lambda w: (score(w), w))
                path.append(v)
                pathset.add(v)
                improved = True
        # Extend from the front
        if path:
            cur = path[0]
            cand = [v for v in ADJ_SET[cur] if v not in pathset and v in comp]
            if cand:
                def score(w):
                    return sum(1 for z in ADJ_SET[w] if z not in pathset and z in comp)
                v = min(cand, key=lambda w: (score(w), w))
                path.insert(0, v)
                pathset.add(v)
                improved = True
        if time.time() > deadline:
            break
    return path

def backtracking_extend(path, comp, deadline):
    """Try to extend the path using backtracking DFS with Warnsdorff ordering."""
    path = list(path)
    pathset = set(path)
    best = list(path)
    
    def dfs(path, pathset):
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best = list(path)
        if len(path) == len(comp):
            return
        cur = path[-1]
        cand = [v for v in ADJ_SET[cur] if v not in pathset and v in comp]
        if not cand:
            return
        # Warnsdorff: fewest onward options first
        def score(w):
            return sum(1 for z in ADJ_SET[w] if z not in pathset and z in comp)
        cand.sort(key=lambda w: (score(w), w))
        for v in cand:
            if time.time() > deadline:
                return
            path.append(v)
            pathset.add(v)
            dfs(path, pathset)
            path.pop()
            pathset.remove(v)
    
    dfs(path, pathset)
    return best

def local_search(path, comp, deadline):
    """Improve the path using 2-opt local search."""
    path = list(path)
    pathset = set(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 2, len(path)):
                # Check if reversing the segment [i+1, j] improves the path
                if path[j] in ADJ_SET[path[i]]:
                    new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                    # Check if new_path is a valid simple path
                    new_set = set(new_path)
                    if len(new_set) != len(new_path):
                        continue
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in ADJ_SET[new_path[k]]:
                            valid = False
                            break
                    if valid and len(new_path) > len(path):
                        path = new_path
                        pathset = set(path)
                        improved = True
                        break
            if improved:
                break
    return path

def greedy_path(comp, deadline):
    """Build an initial path using a greedy heuristic."""
    comp_list = sorted(comp)
    # Start from a vertex with minimum degree
    start = min(comp_list, key=lambda u: (DEG[u], u))
    path = [start]
    pathset = {start}
    comp_set = set(comp)
    
    while len(path) < len(comp):
        cur = path[-1]
        cand = [v for v in ADJ_SET[cur] if v not in pathset and v in comp_set]
        if not cand:
            break
        # Warnsdorff: fewest onward options first
        def score(w):
            return sum(1 for z in ADJ_SET[w] if z not in pathset and z in comp_set)
        v = min(cand, key=lambda w: (score(w), w))
        path.append(v)
        pathset.add(v)
        if time.time() > deadline:
            break
    return path

def verify(path):
    if len(path) < 1:
        return False
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in ADJ_SET[path[i]]:
            return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s limit
    
    comps = components()
    best = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        if comp_size <= len(best):
            continue
        
        # Fast exact case: tree component -> leaf-peel Hamiltonian path if linear.
        if is_tree(comp):
            p = leaf_order_path(comp)
            if p is not None and len(p) == comp_size and verify(p):
                if len(p) > len(best):
                    best = p
                continue
        
        # Greedy path construction
        greedy = greedy_path(comp, deadline)
        if len(greedy) > len(best):
            best = list(greedy)
        
        # Two-ends extension
        two_ends = two_ends_extend(greedy, comp, deadline)
        if len(two_ends) > len(best):
            best = list(two_ends)
        
        # Backtracking extension (time-limited)
        if time.time() < deadline - 5:
            backtrack = backtracking_extend(greedy, comp, deadline - 5)
            if len(backtrack) > len(best):
                best = list(backtrack)
        
        # Local search polish
        if time.time() < deadline - 2:
            polished = local_search(best, comp, deadline - 2)
            if len(polished) > len(best):
                best = list(polished)
        
        if len(best) >= comp_size:
            break
    
    if not best:
        best = [0]
    if not verify(best):
        for v in range(N):
            best = [v]
            break
    return best

if __name__ == "__main__":
    result = construct()
    print(result)
