"""Longest simple path in pinned G(100, 1.8/100) graph using multi-strategy search."""
import random
import time
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Precompute adjacency lists sorted by degree (ascending) for Warnsdorff ordering
ADJ_LIST = [sorted(ADJ[i]) for i in range(N)]
DEG = [len(ADJ[i]) for i in range(N)]


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


def warnsdorff_extend(path, seen, deadline):
    """Extend path using Warnsdorff rule (pick neighbor with fewest unvisited neighbors)."""
    while time.time() < deadline:
        last = path[-1]
        cand = [v for v in ADJ[last] if v not in seen]
        if not cand:
            break
        # Warnsdorff: pick neighbor with fewest unvisited neighbors
        v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path, seen, deadline):
    """Extend path from both ends greedily."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Extend front
        while time.time() < deadline:
            first = path[0]
            cand = [v for v in ADJ[first] if v not in seen]
            if not cand:
                break
            v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        # Extend back
        while time.time() < deadline:
            last = path[-1]
            cand = [v for v in ADJ[last] if v not in seen]
            if not cand:
                break
            v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def forced_move_extend(path, seen, deadline):
    """Extend path preferring forced moves (neighbors with only one unvisited neighbor)."""
    while time.time() < deadline:
        last = path[-1]
        cand = [v for v in ADJ[last] if v not in seen]
        if not cand:
            break
        # Prefer forced moves: neighbor with exactly 1 unvisited neighbor
        forced = [v for v in cand if len([x for x in ADJ[v] if x not in seen]) == 1]
        if forced:
            v = forced[0]
        else:
            v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path


def max_degree_extend(path, seen, deadline):
    """Extend path preferring neighbors with highest degree (more options)."""
    while time.time() < deadline:
        last = path[-1]
        cand = [v for v in ADJ[last] if v not in seen]
        if not cand:
            break
        v = max(cand, key=lambda w: (len(ADJ[w]), -w))
        path.append(v)
        seen.add(v)
    return path


def randomized_greedy(start, comp, strategy, rng, deadline):
    """Run a randomized greedy path from start within comp."""
    path = [start]
    seen = {start}
    comp_set = set(comp)
    
    while time.time() < deadline:
        last = path[-1]
        cand = [v for v in ADJ[last] if v not in seen and v in comp_set]
        if not cand:
            break
        
        if strategy == 'warnsdorff':
            v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen and x in comp_set]), w))
        elif strategy == 'forced':
            forced = [v for v in cand if len([x for x in ADJ[v] if x not in seen and x in comp_set]) == 1]
            if forced:
                v = forced[0]
            else:
                v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen and x in comp_set]), w))
        elif strategy == 'maxdeg':
            v = max(cand, key=lambda w: (len(ADJ[w]), -w))
        elif strategy == 'random':
            v = rng.choice(cand)
        elif strategy == 'mixed':
            # 50% Warnsdorff, 30% random, 20% max degree
            r = rng.random()
            if r < 0.5:
                v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen and x in comp_set]), w))
            elif r < 0.8:
                v = rng.choice(cand)
            else:
                v = max(cand, key=lambda w: (len(ADJ[w]), -w))
        else:
            v = min(cand, key=lambda w: (len([x for x in ADJ[w] if x not in seen and x in comp_set]), w))
        
        path.append(v)
        seen.add(v)
    
    return path


def bdfs_search(start, comp, target_len, deadline):
    """Backtracking DFS with pruning to find path of at least target_len."""
    comp_set = set(comp)
    best = []
    
    def dfs(path, seen, depth):
        nonlocal best
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best = list(path)
        if len(path) >= target_len:
            return
        last = path[-1]
        cand = [v for v in ADJ[last] if v not in seen and v in comp_set]
        if not cand:
            return
        # Sort by Warnsdorff
        cand.sort(key=lambda w: (len([x for x in ADJ[w] if x not in seen and x in comp_set]), w))
        for v in cand:
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
            if len(path) >= target_len:
                return
    
    dfs([start], {start}, 0)
    return best


def path_to_cycle_check(path):
    """Check if path endpoints have an edge (would form a cycle)."""
    if len(path) < 3:
        return False
    return path[0] in ADJ[path[-1]]


def try_insert_detour(path, seen, comp, deadline):
    """Try to insert a detour: for each edge in path, check if we can go around."""
    path = list(path)
    comp_set = set(comp)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            # Look for a vertex w not in path that connects to both a and b (or to a and something)
            # Actually, look for a path from a to b that doesn't use the direct edge
            # Simpler: find w adjacent to a, not in seen, then see if we can reach b from w
            # This is complex; let's try a simpler detour:
            # Find w adjacent to a (not in path) and w adjacent to some x, then x adjacent to b
            for w in ADJ[a]:
                if w in seen or w not in comp_set:
                    continue
                # Check if w connects back to b or to path[i+1]
                if w in ADJ[b]:
                    # Insert w between a and b
                    path.insert(i + 1, w)
                    seen.add(w)
                    improved = True
                    break
                # Check if there's a 2-step detour: w -> x -> b
                for x in ADJ[w]:
                    if x in seen or x not in comp_set or x == a or x == b:
                        continue
                    if x in ADJ[b]:
                        path[i+1:i+1] = [w, x]
                        seen.add(w)
                        seen.add(x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return path


def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin before 90s limit
    
    # Build components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Strategy 1: Multi-start randomized greedy with multiple heuristics
    strategies = ['warnsdorff', 'forced', 'maxdeg', 'random', 'mixed']
    rng = random.Random(42)
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        
        # Multi-start with multiple strategies
        for start in comp_list:
            if time.time() > deadline:
                break
            for strategy in strategies:
                if time.time() > deadline:
                    break
                p = randomized_greedy(start, comp, strategy, rng, deadline)
                p = extend_both_ends(p, set(p), deadline)
                if len(p) > best_len:
                    best_path = p
                    best_len = len(p)
        
        # If we haven't found a Hamiltonian path in this component, try more
        if best_len < comp_size:
            # Try backtracking DFS from a few promising starts
            for start in comp_list[:min(10, len(comp_list))]:
                if time.time() > deadline:
                    break
                dl = min(deadline, time.time() + 5.0)
                p = bdfs_search(start, comp, best_len + 1, dl)
                if len(p) > best_len:
                    best_path = p
                    best_len = len(p)
        
        # Try detour insertion on the best path found so far
        if len(best_path) > 2:
            seen = set(best_path)
            best_path = try_insert_detour(best_path, seen, comp, deadline)
            if len(best_path) > best_len:
                best_len = len(best_path)
    
    # Additional randomized restarts if time permits
    if time.time() < deadline - 5 and best_len < max(len(c) for c in comps):
        main_comp = max(comps, key=len)
        for _ in range(50):
            if time.time() > deadline:
                break
            start = rng.choice(sorted(main_comp))
            strategy = rng.choice(strategies)
            p = randomized_greedy(start, main_comp, strategy, rng, deadline)
            p = extend_both_ends(p, set(p), deadline)
            if len(p) > best_len:
                best_path = p
                best_len = len(p)
    
    if not best_path:
        best_path = [0]
    
    # Final safety check: verify path validity
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
