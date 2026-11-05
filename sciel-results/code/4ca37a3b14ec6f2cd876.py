"""Hamilton cycle construction via greedy + 2-opt refinement (polynomial time)."""
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Edge position lookup: O(1) access
edge_pos = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(min(u, v), max(u, v))] = idx

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def cycle_max_edge(cycle):
    """Return the maximum arrival index among cycle edges."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        pos = edge_pos[_edge_key(u, v)]
        if pos > mx:
            mx = pos
    return mx

def _greedy_cycle(start=0):
    """Build a Hamilton cycle greedily: always pick the unused neighbor with smallest edge index."""
    path = [start]
    used = [False] * N
    used[start] = True
    for _ in range(N - 1):
        cur = path[-1]
        best_nbr = None
        best_pos = float('inf')
        for w in range(N):
            if not used[w]:
                pos = edge_pos[_edge_key(cur, w)]
                if pos < best_pos:
                    best_pos = pos
                    best_nbr = w
        path.append(best_nbr)
        used[best_nbr] = True
    # Close the cycle
    return path

def _two_opt_refine(cycle, deadline):
    """Improve cycle by 2-opt moves: reverse segments to reduce max edge index."""
    best_cycle = list(cycle)
    best_score = cycle_max_edge(best_cycle)
    start_time = time.time()
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i + 2, N):
                if i == 0 and j == N - 1:
                    continue  # would just reverse the whole cycle
                # Consider reversing segment [i+1..j]
                # Current edges: (cycle[i], cycle[i+1]) and (cycle[j], cycle[(j+1)%N])
                # New edges: (cycle[i], cycle[j]) and (cycle[i+1], cycle[(j+1)%N])
                u1 = best_cycle[i]
                u2 = best_cycle[i + 1]
                u3 = best_cycle[j]
                u4 = best_cycle[(j + 1) % N]
                
                old_e1 = edge_pos[_edge_key(u1, u2)]
                old_e2 = edge_pos[_edge_key(u3, u4)]
                new_e1 = edge_pos[_edge_key(u1, u3)]
                new_e2 = edge_pos[_edge_key(u2, u4)]
                
                # Check if this improves the max
                # Remove old edges, add new edges
                # The new max is max of all edges except the two removed, plus the two new
                # Quick check: if both new edges are <= current best and at least one old edge was the max
                old_max = max(old_e1, old_e2)
                new_max_pair = max(new_e1, new_e2)
                
                # Full check: compute new cycle's max
                if new_max_pair >= best_score:
                    # Could still help if old edges contained the unique max
                    # But let's do a more careful check
                    pass
                
                # Perform the reversal and check
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_max_edge(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
    
    return best_cycle, best_score

def _try_multiple_starts(deadline):
    """Try greedy from multiple starting vertices, refine each, return best."""
    best_cycle = None
    best_score = float('inf')
    
    for start in range(N):
        if time.time() > deadline:
            break
        cycle = _greedy_cycle(start)
        refined, score = _two_opt_refine(cycle, deadline)
        if score < best_score:
            best_score = score
            best_cycle = refined
            if best_score <= 30:  # very good, stop early
                break
    
    return best_cycle, best_score

def _hill_climb_refine(cycle, deadline):
    """Additional local search: try swapping non-adjacent vertices."""
    best_cycle = list(cycle)
    best_score = cycle_max_edge(best_cycle)
    start_time = time.time()
    
    for _ in range(50):  # limited iterations
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for j in range(i + 1, N):
                if abs(i - j) == 1 or (i == 0 and j == N - 1):
                    continue  # adjacent or wrap-around adjacent
                # Swap positions i and j
                new_cycle = best_cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                new_score = cycle_max_edge(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
        if not improved:
            break
    
    return best_cycle, best_score

_cached_result = None

def construct():
    global _cached_result
    if _cached_result is not None:
        return _cached_result
    
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Phase 1: Greedy + 2-opt from multiple starts
    best_cycle, best_score = _try_multiple_starts(deadline)
    
    # Phase 2: Hill climbing refinement
    if best_cycle is not None and time.time() < deadline:
        refined, score = _hill_climb_refine(best_cycle, deadline)
        if score < best_score:
            best_cycle = refined
            best_score = score
    
    # Fallback: ensure we have a valid cycle
    if best_cycle is None or len(best_cycle) != N:
        best_cycle = _greedy_cycle(0)
    
    _cached_result = best_cycle
    return best_cycle
