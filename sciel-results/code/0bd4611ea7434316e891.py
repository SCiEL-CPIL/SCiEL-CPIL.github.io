"""Rewritten program for the suite task; stronger search within the same family."""
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEADLINE = 85.0  # safety margin before 90s kill

def _greedy_extend(path, seen):
    """Extend path greedily using a score that balances remaining options."""
    while True:
        cur = path[-1]
        cands = [v for v in ADJ[cur] if v not in seen]
        if not cands:
            break
        # Prefer vertices with fewer unvisited neighbors (dead-end first) to save options
        # But also consider connectivity: prefer vertices that lead to larger components
        # Use a simple heuristic: sort by (number of unvisited neighbors, vertex_id)
        cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
        nxt = cands[0]
        path.append(nxt)
        seen.add(nxt)
    return path

def _try_all_starts():
    """Try greedy extension from every start vertex."""
    best = []
    for start in range(N):
        path = [start]
        seen = {start}
        path = _greedy_extend(path, seen)
        if len(path) > len(best):
            best = path[:]
    return best

def _backtrack_search(start, time_limit=10.0):
    """Backtracking search from a given start vertex with time limit."""
    best = []
    start_time = time.time()
    
    def dfs(path, seen):
        nonlocal best
        if len(path) > len(best):
            best = path[:]
        cur = path[-1]
        cands = [v for v in ADJ[cur] if v not in seen]
        if not cands:
            return
        # Sort candidates by heuristic
        cands.sort(key=lambda v: (len(ADJ[v] - seen), v))
        for nxt in cands:
            if time.time() - start_time > time_limit:
                return
            path.append(nxt)
            seen.add(nxt)
            dfs(path, seen)
            path.pop()
            seen.remove(nxt)
    
    dfs([start], {start})
    return best

def _multi_start_search():
    """Try multiple start vertices with backtracking, allocating time based on remaining budget."""
    global DEADLINE
    remaining = DEADLINE - time.time()
    if remaining < 1.0:
        return _try_all_starts()
    
    # First, do greedy to get a baseline
    greedy_best = _try_all_starts()
    best = greedy_best[:]
    
    # Allocate remaining time for backtracking from promising starts
    # Sort starts by greedy path length (descending) to prioritize
    start_scores = []
    for s in range(N):
        path = [s]
        seen = {s}
        path = _greedy_extend(path, seen)
        start_scores.append((len(path), s))
    start_scores.sort(reverse=True)
    
    # Try backtracking from top starts
    per_start_time = min(2.0, max(0.5, (DEADLINE - time.time() - 0.5) / max(1, len(start_scores))))
    
    for score, s in start_scores:
        if time.time() > DEADLINE - 0.5:
            break
        # Skip if greedy already found a full path
        if score >= N:
            break
        bt_best = _backtrack_search(s, time_limit=per_start_time)
        if len(bt_best) > len(best):
            best = bt_best[:]
        # If we've found a Hamiltonian path, we're done
        if len(best) == N:
            break
    
    return best

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # 85s wall-clock budget with 5s safety margin
    
    # Phase 1: Greedy from all starts
    best = _try_all_starts()
    
    # If greedy already found a Hamiltonian path, return immediately
    if len(best) == N:
        return best
    
    # Phase 2: Backtracking from promising starts
    remaining = DEADLINE - time.time()
    if remaining > 2.0:
        # Sort starts by greedy path length
        start_scores = []
        for s in range(N):
            path = [s]
            seen = {s}
            path = _greedy_extend(path, seen)
            start_scores.append((len(path), s))
        start_scores.sort(reverse=True)
        
        # Allocate time for backtracking
        n_starts = min(len(start_scores), max(1, int(remaining / 2.0)))
        per_start_time = min(3.0, max(0.3, (remaining - 1.0) / max(1, n_starts)))
        
        for i in range(n_starts):
            if time.time() > DEADLINE - 0.5:
                break
            score, s = start_scores[i]
            # Skip if greedy already covered all vertices
            if score >= N:
                break
            bt_best = _backtrack_search(s, time_limit=per_start_time)
            if len(bt_best) > len(best):
                best = bt_best[:]
            if len(best) == N:
                break
    
    return best

# EVOLVE-BLOCK-END
