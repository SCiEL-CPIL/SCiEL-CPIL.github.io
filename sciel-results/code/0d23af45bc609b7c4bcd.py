import random
import time
from itertools import combinations

_DEADLINE = 140.0
_start_time = time.time()

def _check_time():
    return time.time() - _start_time > _DEADLINE

def is_2_colorable(n, edges):
    if not edges:
        return True
        
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    if n <= 20:
        for m in range(1 << n):
            valid = True
            for em in edge_masks:
                if (m & em) == 0 or (m & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # Simple backtracking for larger n
        # This is a fallback, we prefer n <= 20
        colors = [-1] * n
        def propagate():
            changed = True
            while changed:
                changed = False
                for e in edges:
                    c0 = 0
                    c1 = 0
                    unassigned = []
                    for v in e:
                        if colors[v] == 0: c0 += 1
                        elif colors[v] == 1: c1 += 1
                        else: unassigned.append(v)
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if colors[v] == -1:
                            colors[v] = 1
                            changed = True
                        elif colors[v] == 0:
                            return False
                    elif c1 == 4 and len(unassigned) == 1:
                        v = unassigned[0]
                        if colors[v] == -1:
                            colors[v] = 0
                            changed = True
                        elif colors[v] == 1:
                            return False
            return True

        def backtrack(idx):
            if idx >= n:
                for e in edges:
                    c0 = sum(1 for v in e if colors[v] == 0)
                    if c0 == 5 or c0 == 0:
                        return False
                return True
            
            for c in (0, 1):
                colors[idx] = c
                saved = colors[idx:]
                if propagate():
                    if backtrack(idx + 1):
                        return True
                colors[idx:] = saved
                if _check_time():
                    return True
            return False
        
        return backtrack(0)

def construct():
    best_n = 9
    best_edges = None
    best_count = float('inf')
    
    # Search n from 9 to 15
    for n in range(9, 16):
        if _check_time():
            break
            
        all_possible_edges = list(combinations(range(n), 5))
        num_possible = len(all_possible_edges)
        
        max_restarts = 10
        for restart in range(max_restarts):
            if _check_time():
                break
            
            # Initialize
            if restart == 0:
                edges = set(all_possible_edges)
            else:
                # Random dense graph
                density = random.uniform(0.2, 0.8)
                num_edges = int(num_possible * density)
                edges = set(random.sample(all_possible_edges, num_edges))
            
            # Ensure non-2-colorable
            if is_2_colorable(n, list(edges)):
                remaining = [e for e in all_possible_edges if e not in edges]
                random.shuffle(remaining)
                for e in remaining:
                    if _check_time():
                        break
                    edges.add(e)
                    if not is_2_colorable(n, list(edges)):
                        break
                else:
                    continue
            
            # Local search: remove edges
            improved = True
            while improved and not _check_time():
                improved = False
                edge_list = list(edges)
                random.shuffle(edge_list)
                for e in edge_list:
                    if _check_time():
                        break
                    if e not in edges:
                        continue
                    candidate = edges - {e}
                    if not is_2_colorable(n, list(candidate)):
                        edges = candidate
                        improved = True
                        break
            
            # Update best
            if len(edges) < best_count:
                best_count = len(edges)
                best_n = n
                best_edges = list(edges)
                
            # Try swaps to escape local minima
            if not _check_time():
                for _ in range(3):
                    if _check_time():
                        break
                    if not edges:
                        continue
                    e_remove = random.choice(list(edges))
                    candidate_edges = edges - {e_remove}
                    possible_adds = [e for e in all_possible_edges if e not in candidate_edges]
                    if not possible_adds:
                        continue
                    e_add = random.choice(possible_adds)
                    candidate_edges.add(e_add)
                    
                    if not is_2_colorable(n, list(candidate_edges)):
                        # Minimize candidate_edges
                        improved = True
                        while improved and not _check_time():
                            improved = False
                            edge_list = list(candidate_edges)
                            random.shuffle(edge_list)
                            for e in edge_list:
                                if _check_time():
                                    break
                                if e not in candidate_edges:
                                    continue
                                c = candidate_edges - {e}
                                if not is_2_colorable(n, list(c)):
                                    candidate_edges = c
                                    improved = True
                                    break
                        
                        if len(candidate_edges) < best_count:
                            best_count = len(candidate_edges)
                            best_n = n
                            best_edges = list(candidate_edges)

    if best_edges is None:
        # Fallback
        best_n = 9
        best_edges = [list(e) for e in combinations(range(best_n), 5)]
        
    return {'n': best_n, 'edges': best_edges}
