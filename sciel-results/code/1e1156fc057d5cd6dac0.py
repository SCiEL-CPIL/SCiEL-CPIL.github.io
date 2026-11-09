"""Find longest simple path in pinned graph using component analysis + Warnsdorff + time-bounded DFS."""
import time
from collections import deque
import heapq

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

def warnsdorff_score(v, seen):
    """Prefer vertices with fewer unseen neighbors (Warnsdorff's rule)."""
    return (len(ADJ[v] - seen), v)

def greedy_path_from(start, seen=None):
    """Greedy Warnsdorff path from start."""
    if seen is None:
        seen = {start}
    else:
        seen = set(seen)
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: warnsdorff_score(w, seen))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path greedily from both ends."""
    path = list(path)
    seen = set(seen)
    changed = True
    while changed:
        changed = False
        # Extend front
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def time_bounded_dfs(start, comp_set, deadline, best_path, best_len):
    """Iterative time-bounded DFS to find longest path starting from start within comp."""
    best = [best_path[:], best_len]
    comp_size = len(comp_set)
    
    # Iterative DFS with explicit stack
    # Stack entries: (path, seen_frozen, next_candidate_index, candidates)
    # To save memory, we track path and seen, and recompute candidates each time
    # But for speed, we pre-sort candidates at each level
    
    path = [start]
    seen = {start}
    if len(path) > best[1]:
        best[0] = path[:]
        best[1] = len(path)
    
    # Use a stack of (path_copy, seen_copy, candidate_list, candidate_index)
    # But copying seen is expensive. Instead, use backtracking with undo.
    # Simpler: recursive-like iterative with explicit backtracking stack.
    
    # Stack: list of (vertex_added, candidate_list_for_next, candidate_index)
    # We maintain path and seen incrementally.
    stack = []  # each entry: (candidates, next_idx)
    
    def get_sorted_candidates(v, seen):
        cand = ADJ[v] & comp_set - seen
        return sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    
    # Initial candidates for start
    cands = get_sorted_candidates(start, seen)
    if cands:
        stack.append((cands, 0))
    
    steps = 0
    while stack:
        steps += 1
        if steps % 5000 == 0 and time.time() > deadline:
            break
        if time.time() > deadline:
            break
        
        cands, idx = stack[-1]
        if idx >= len(cands):
            # Backtrack
            stack.pop()
            if stack:
                # Remove last vertex from path
                v = path.pop()
                seen.remove(v)
                # Advance parent's candidate index
                parent_cands, parent_idx = stack[-1]
                stack[-1] = (parent_cands, parent_idx + 1)
            continue
        
        v = cands[idx]
        # Advance this level's index
        stack[-1] = (cands, idx + 1)
        
        # Visit v
        path.append(v)
        seen.add(v)
        
        if len(path) > best[1]:
            best[0] = path[:]
            best[1] = len(path)
        
        if len(path) == comp_size:
            # Hamiltonian path found
            path.pop()
            seen.remove(v)
            if stack:
                parent_cands, parent_idx = stack[-1]
                stack[-1] = (parent_cands, parent_idx + 1)
            continue
        
        next_cands = get_sorted_candidates(v, seen)
        if next_cands:
            stack.append((next_cands, 0))
        else:
            # Dead end, backtrack immediately
            path.pop()
            seen.remove(v)
    
    return best[0]

def local_search_improve(path, comp_set, deadline):
    """Try to insert unvisited vertices into the path at various positions."""
    path = list(path)
    seen = set(path)
    unvisited = comp_set - seen
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in unvisited:
            if time.time() > deadline:
                break
            # Try inserting u between path[i] and path[i+1]
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    # Can insert u here
                    new_path = path[:i+1] + [u] + path[i+1:]
                    # Validate
                    valid = True
                    for j in range(len(new_path) - 1):
                        if new_path[j+1] not in ADJ[new_path[j]]:
                            valid = False
                            break
                    if valid:
                        path = new_path
                        seen.add(u)
                        unvisited.remove(u)
                        improved = True
                        break
            if improved:
                break
        # Also try extending ends
        path = extend_both_ends(path, set(path))
        path = [v for v in path if v in comp_set]
        seen = set(path)
        unvisited = comp_set - seen
    
    return path

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    if not comps:
        return [0]
    
    largest_comp = comps[0]
    comp_size = len(largest_comp)
    comp_list = sorted(largest_comp)
    
    best_path = []
    
    # Phase 1: Greedy + extend from every vertex in largest component
    for start in comp_list:
        if time.time() > deadline:
            break
        p = greedy_path_from(start)
        p = extend_both_ends(p, set(p))
        # Only keep vertices in the component
        p = [v for v in p if v in largest_comp]
        if len(p) > len(best_path):
            best_path = list(p)
    
    # If we found Hamiltonian path in largest component, done
    if len(best_path) >= comp_size:
        # Validate
        for i in range(len(best_path) - 1):
            if best_path[i+1] not in ADJ[best_path[i]]:
                break
        else:
            return best_path
    
    # Phase 2: Time-bounded DFS from promising starts
    # Sort starts by degree descending
    promising_starts = sorted(comp_list, key=lambda x: (-len(ADJ[x] & largest_comp), x))
    
    # Also try endpoints and middle of current best path
    if best_path:
        for v in [best_path[0], best_path[-1], best_path[len(best_path)//2]]:
            if v not in promising_starts:
                promising_starts.insert(0, v)
    
    # Limit number of DFS starts to avoid spending too much time
    max_dfs_starts = min(len(promising_starts), 15)
    
    for start in promising_starts[:max_dfs_starts]:
        if time.time() > deadline:
            break
        if len(best_path) >= comp_size:
            break
        
        # Give each start a time budget
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        budget = min(4.0, remaining * 0.3)
        dl = time.time() + budget
        
        local_best = time_bounded_dfs(start, largest_comp, dl, best_path, len(best_path))
        if len(local_best) > len(best_path):
            best_path = local_best
    
    # Phase 3: Try to extend the best path found
    if best_path:
        best_path = extend_both_ends(best_path, set(best_path))
        best_path = [v for v in best_path if v in largest_comp]
    
    # Phase 4: Local search - try to insert unvisited vertices into the path
    if best_path and len(best_path) < comp_size:
        best_path = local_search_improve(best_path, largest_comp, deadline)
    
    # Final validation
    if not best_path:
        best_path = [comp_list[0]]
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [comp_list[0]]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path
