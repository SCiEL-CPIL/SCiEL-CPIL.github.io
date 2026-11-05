import time
import random
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 140.0


def _check_non_2_colorable(n, edges, timeout=3.0):
    """
    Return True if the hypergraph is NOT 2-colorable (no proper 2-coloring exists).
    Proper 2-coloring: every edge has at least one vertex of each color.
    """
    start = time.time()
    edges_set = [frozenset(e) for e in edges]
    
    vertex_edges = [[] for _ in range(n)]
    for idx, e in enumerate(edges_set):
        for v in e:
            vertex_edges[v].append(idx)
    
    color = [-1] * n
    
    def propagate(v):
        if time.time() - start > timeout:
            return 'timeout'
        if color[v] == -1:
            return True
        queue = [v]
        while queue:
            u = queue.pop()
            for eidx in vertex_edges[u]:
                edge = edges_set[eidx]
                cu = color[u]
                same = 0
                opp = 0
                unassigned = None
                for w in edge:
                    if w == u:
                        continue
                    if color[w] == cu:
                        same += 1
                    elif color[w] == 1 - cu:
                        opp += 1
                    else:
                        unassigned = w
                        break
                if unassigned is None:
                    # All assigned
                    if same == 4 or opp == 4:
                        # Check: if all 5 same color, contradiction
                        all_same = all(color[w] == color[list(edge)[0]] for w in edge)
                        if all_same:
                            return False
                else:
                    if same == 4:
                        if color[unassigned] == -1:
                            color[unassigned] = 1 - cu
                            queue.append(unassigned)
                        elif color[unassigned] == cu:
                            return False
                    elif opp == 4:
                        if color[unassigned] == -1:
                            color[unassigned] = cu
                            queue.append(unassigned)
                        elif color[unassigned] != cu:
                            return False
        return True
    
    def all_edges_mixed():
        for e in edges_set:
            ce = [color[v] for v in e]
            if all(c == ce[0] for c in ce):
                return False
        return True
    
    def backtrack(pos):
        if time.time() - start > timeout:
            return 'timeout'
        if pos == n:
            return all_edges_mixed()
        for c in [0, 1]:
            color[pos] = c
            res = propagate(pos)
            if res == 'timeout':
                color[pos] = -1
                return 'timeout'
            if res:
                r = backtrack(pos + 1)
                if r == 'timeout':
                    color[pos] = -1
                    return 'timeout'
                if r:
                    color[pos] = -1
                    return True
            color[pos] = -1
        return False
    
    # Try to find a valid 2-coloring
    r = backtrack(0)
    if r == 'timeout':
        return None
    # r=True means we found a valid 2-coloring, so hypergraph IS 2-colorable
    # r=False means no valid 2-coloring, so hypergraph is NOT 2-colorable
    return not r


def construct():
    elapsed = time.time() - _START_TIME
    
    # Use a known construction: complete 5-uniform hypergraph on 9 vertices
    # This has 126 edges and is definitely non-2-colorable.
    # Try to find something smaller within time budget.
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Verify it's non-2-colorable
    if time.time() - _START_TIME > _TIME_LIMIT - 5:
        return {'n': n, 'edges': all_edges}
    
    result = _check_non_2_colorable(n, all_edges, timeout=3.0)
    if result != True:
        return {'n': n, 'edges': all_edges}
    
    # Greedy edge removal
    current_edges = list(all_edges)
    random.seed(12345)
    
    for _ in range(200):  # Max 200 removal attempts
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        if not current_edges:
            break
        
        # Try removing a random edge
        idx = random.randint(0, len(current_edges) - 1)
        candidate = current_edges[:idx] + current_edges[idx+1:]
        
        if not candidate:
            continue
            
        r = _check_non_2_colorable(n, candidate, timeout=2.0)
        if r == True:
            current_edges = candidate
    
    best_n = n
    best_edges = current_edges
    best_count = len(current_edges)
    
    # Try larger vertex sets with random sparse hypergraphs
    for trial_n in [10, 11, 12, 13, 14]:
        if time.time() - _START_TIME > _TIME_LIMIT - 15:
            break
        
        for trial in range(3):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            all_possible = list(combinations(range(trial_n), 5))
            random.shuffle(all_possible)
            
            for edge_count in [35, 40, 45, 50]:
                if time.time() - _START_TIME > _TIME_LIMIT - 8:
                    break
                if edge_count >= len(all_possible):
                    continue
                
                candidate_edges = [list(e) for e in all_possible[:edge_count]]
                r = _check_non_2_colorable(trial_n, candidate_edges, timeout=2.0)
                if r == True and edge_count < best_count:
                    best_n = trial_n
                    best_edges = candidate_edges
                    best_count = edge_count
    
    if time.time() - _START_TIME > _TIME_LIMIT - 3:
        return {'n': best_n, 'edges': best_edges}
    
    return {'n': best_n, 'edges': best_edges}
