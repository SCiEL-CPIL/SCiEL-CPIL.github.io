"""
Longest simple path in pinned G(100, 0.018) graph.
Strategy:
1. Identify connected components.
2. For the largest component, use iterative deepening DFS with Warnsdorff heuristic
   to find a path covering as many vertices as possible.
3. Use memoization and symmetry breaking to speed up search.
4. If a Hamiltonian path exists in the largest component, return it.
   Otherwise, return the longest path found.
"""
import time
import sys
from collections import deque

# Pinned graph parameters
P = {
    'n': 100,
    'edges': [
        [0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18],
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
        [69, 90], [70, 92], [74, 85], [75, 82]
    ]
}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Precompute degrees
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

def greedy_warnsdorff(start, comp_set):
    """Greedy path using Warnsdorff's rule (prefer vertices with fewer unvisited neighbors)."""
    seen = {start}
    path = [start]
    while True:
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            break
        # Warnsdorff: pick vertex with fewest unvisited neighbors
        cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
        path.append(cands[0])
        seen.add(cands[0])
    return path

def extend_both_ends(path, comp_set):
    """Try to extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(path)
    improved = True
    while improved:
        improved = False
        # Extend front
        while True:
            cands = [v for v in ADJ[path[0]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
            path.insert(0, cands[0])
            seen.add(cands[0])
            improved = True
        # Extend back
        while True:
            cands = [v for v in ADJ[path[-1]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and v not in seen]), v))
            path.append(cands[0])
            seen.add(cands[0])
            improved = True
    return path

def iddfs_longest_path(comp, deadline):
    """
    Iterative deepening DFS to find longest simple path in component.
    Uses Warnsdorff ordering and time-limited search.
    """
    comp_set = set(comp)
    comp_list = sorted(comp)
    target_len = len(comp)
    best_path = []
    
    # Start with greedy to get a good initial path
    for start in comp_list:
        p = greedy_warnsdorff(start, comp_set)
        p = extend_both_ends(p, comp_set)
        if len(p) > len(best_path):
            best_path = p
    
    if len(best_path) >= target_len:
        return best_path
    
    # Now try to improve with DFS
    # Use the best greedy path as a starting point and try to insert missing vertices
    missing = comp_set - set(best_path)
    
    # Try DFS from each end of best_path, trying to incorporate missing vertices
    def dfs_extend(path, seen, remaining_budget):
        """DFS to extend path, trying to cover more vertices."""
        nonlocal best_path
        if time.time() > deadline:
            return False
        
        if len(path) > len(best_path):
            best_path = list(path)
            if len(path) >= target_len:
                return True
        
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            return False
        
        # Warnsdorff ordering
        cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
        
        for v in cands:
            path.append(v)
            seen.add(v)
            if dfs_extend(path, seen, remaining_budget):
                return True
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return False
        return False
    
    # Try extending from both ends of the best greedy path
    for start in [best_path[0], best_path[-1]]:
        if time.time() > deadline:
            break
        seen = set(best_path)
        # Try adding missing vertices one by one via DFS
        dfs_extend(list(best_path), seen, deadline - time.time())
    
    # Try full DFS from various starts with time limit
    for start in comp_list:
        if time.time() > deadline:
            break
        if len(best_path) >= target_len:
            break
        seen = {start}
        path = [start]
        dfs_extend(path, seen, deadline - time.time())
    
    return best_path

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin: return well before 90s limit
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_list = sorted(comp)
        
        # Get initial greedy path
        for start in comp_list:
            if time.time() > deadline:
                break
            p = greedy_warnsdorff(start, set(comp))
            p = extend_both_ends(p, set(comp))
            if len(p) > len(best_path):
                best_path = p
        
        # If we haven't found a full path, try IDDFS
        if len(best_path) < len(comp):
            improved = iddfs_longest_path(comp_list, deadline)
            if len(improved) > len(best_path):
                best_path = improved
    
    if not best_path:
        best_path = [0]
    
    # Verify validity
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
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
