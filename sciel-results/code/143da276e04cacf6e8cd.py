"""
Longest simple path in pinned G(100, 0.018) graph.
Structural change: Use bidirectional path growth with 2-opt local search,
component-aware pruning, and multi-start diversification.
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
    """Greedy path using Warnsdorff's rule."""
    seen = {start}
    path = [start]
    while True:
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            break
        cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
        path.append(cands[0])
        seen.add(cands[0])
    return path

def greedy_reverse_warnsdorff(start, comp_set):
    """Greedy path preferring vertices with MORE unvisited neighbors (diversity)."""
    seen = {start}
    path = [start]
    while True:
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            break
        cands.sort(key=lambda v: (-len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
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
        while True:
            cands = [v for v in ADJ[path[0]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
            path.insert(0, cands[0])
            seen.add(cands[0])
            improved = True
        while True:
            cands = [v for v in ADJ[path[-1]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
            path.append(cands[0])
            seen.add(cands[0])
            improved = True
    return path

def two_opt_improve(path, comp_set, deadline):
    """
    2-opt style local search: try to insert missing vertices into the path.
    For each missing vertex, find a position in the path where it can be inserted
    (i.e., its neighbors on both sides in the path are adjacent in the graph).
    """
    path = list(path)
    seen = set(path)
    missing = comp_set - seen
    
    if not missing:
        return path
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in list(missing):
            if time.time() > deadline:
                break
            # Try to insert m at each position i (between path[i-1] and path[i])
            # Position 0: before path[0], need m adjacent to path[0]
            # Position i: between path[i-1] and path[i], need m adjacent to both
            # Position len(path): after path[-1], need m adjacent to path[-1]
            
            best_pos = -1
            best_gain = 0
            
            # Check insertion at each position
            for i in range(len(path) + 1):
                if i == 0:
                    # Insert before path[0]
                    if path[0] in ADJ[m]:
                        # Check if this creates a valid path
                        new_path = [m] + path
                        # Verify all edges
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in ADJ[new_path[j]]:
                                valid = False
                                break
                        if valid:
                            best_pos = 0
                            break
                elif i == len(path):
                    # Insert after path[-1]
                    if path[-1] in ADJ[m]:
                        new_path = path + [m]
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in ADJ[new_path[j]]:
                                valid = False
                                break
                        if valid:
                            best_pos = i
                            break
                else:
                    # Insert between path[i-1] and path[i]
                    if path[i-1] in ADJ[m] and m in ADJ[path[i]]:
                        # Remove edge path[i-1]-path[i], add path[i-1]-m and m-path[i]
                        new_path = path[:i] + [m] + path[i:]
                        # Verify all edges
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in ADJ[new_path[j]]:
                                valid = False
                                break
                        if valid:
                            best_pos = i
                            break
            
            if best_pos >= 0:
                # Insert m at best_pos
                if best_pos == 0:
                    path.insert(0, m)
                else:
                    path.insert(best_pos, m)
                seen.add(m)
                missing.discard(m)
                improved = True
    
    return path

def bidirectional_grow(start, comp_set, deadline):
    """
    Grow path bidirectionally from start, alternating between front and back.
    """
    seen = {start}
    path = [start]
    
    while time.time() < deadline:
        # Try to extend front
        front_cands = [v for v in ADJ[path[0]] if v in comp_set and v not in seen]
        # Try to extend back
        back_cands = [v for v in ADJ[path[-1]] if v in comp_set and v not in seen]
        
        if not front_cands and not back_cands:
            break
        
        # Choose which end to extend based on fewer options (Warnsdorff)
        if front_cands and (not back_cands or 
            min(len(ADJ[v] - seen) for v in front_cands) <= min(len(ADJ[v] - seen) for v in back_cands)):
            # Extend front
            v = min(front_cands, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
        else:
            # Extend back
            v = min(back_cands, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
    
    return path

def dfs_with_pruning(path, seen, comp_set, target_len, deadline, best_path):
    """
    DFS with component-aware pruning: only continue if the remaining unvisited
    vertices in the current connected component can potentially lead to a longer path.
    """
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
    
    # Pruning: check if we can still reach target_len
    # Simple bound: len(path) + len(cands) < target_len -> prune
    if len(path) + len(cands) < len(best_path):
        return False
    
    # Warnsdorff ordering
    cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
    
    for v in cands:
        path.append(v)
        seen.add(v)
        if dfs_with_pruning(path, seen, comp_set, target_len, deadline, best_path):
            return True
        path.pop()
        seen.remove(v)
        if time.time() > deadline:
            return False
    
    return False

def construct():
    t0 = time.time()
    deadline = t0 + 80  # Safety margin
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_set = set(comp)
        comp_list = sorted(comp)
        comp_size = len(comp)
        
        # Phase 1: Multi-start greedy with different heuristics
        for start in comp_list:
            if time.time() > deadline:
                break
            
            # Standard Warnsdorff
            p1 = greedy_warnsdorff(start, comp_set)
            p1 = extend_both_ends(p1, comp_set)
            if len(p1) > len(best_path):
                best_path = p1
            
            # Reverse Warnsdorff (diversity)
            p2 = greedy_reverse_warnsdorff(start, comp_set)
            p2 = extend_both_ends(p2, comp_set)
            if len(p2) > len(best_path):
                best_path = p2
            
            # Bidirectional grow
            p3 = bidirectional_grow(start, comp_set, deadline)
            p3 = extend_both_ends(p3, comp_set)
            if len(p3) > len(best_path):
                best_path = p3
        
        if len(best_path) >= comp_size:
            break
        
        # Phase 2: 2-opt improvement on best path
        if time.time() < deadline:
            improved = two_opt_improve(best_path, comp_set, deadline)
            if len(improved) > len(best_path):
                best_path = improved
        
        if len(best_path) >= comp_size:
            break
        
        # Phase 3: DFS with pruning from promising starts
        if time.time() < deadline:
            # Try DFS from each end of best path
            for start in [best_path[0], best_path[-1]]:
                if time.time() > deadline:
                    break
                seen = set(best_path)
                path = list(best_path)
                dfs_with_pruning(path, seen, comp_set, comp_size, deadline, best_path)
            
            # Try DFS from other starts with time limit
            for start in comp_list:
                if time.time() > deadline:
                    break
                if len(best_path) >= comp_size:
                    break
                seen = {start}
                path = [start]
                dfs_with_pruning(path, seen, comp_set, comp_size, deadline, best_path)
        
        if len(best_path) >= comp_size:
            break
    
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
