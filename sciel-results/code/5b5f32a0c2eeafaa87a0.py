"""Priority-queue branch and bound for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product
import heapq

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and per-pair blocked sets."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    pair_blocked = {}
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            mask1 = (dist[i] == dij)
            mask1[j] = False
            mask2 = (dist[j] == dij)
            mask2[i] = False
            mask3 = (dist[i] == dist[j])
            mask3[i] = False
            mask3[j] = False
            combined = mask1 | mask2 | mask3
            ks = np.where(combined)[0]
            blocked_set = frozenset(ks.tolist())
            pair_blocked[(i, j)] = blocked_set
            pair_blocked[(j, i)] = blocked_set
    
    return points, dist, pair_blocked

def _is_valid_set(S, pair_blocked):
    """Check if S contains no isosceles triangle."""
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked:
                if pair_blocked[key] & S_set:
                    return False
    return True

def _is_maximal(S, pair_blocked):
    """Check if S is maximal: no point outside S can be added."""
    if not S:
        return False
    S_set = set(S)
    all_blocked = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked:
                all_blocked |= pair_blocked[key]
    for p in range(V):
        if p in S_set:
            continue
        if p not in all_blocked:
            return False
    return True

def _can_add(p, S, pair_blocked):
    """Check if point p can be added to S without creating an isosceles triangle."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            key = (S[i], S[j])
            if key in pair_blocked and p in pair_blocked[key]:
                return False
    return True

def _blocking_power(p, S, pair_blocked, U):
    """Compute how many points in U would be blocked by adding p to S."""
    nb = set()
    for a in S:
        key = (a, p)
        if key in pair_blocked:
            nb |= pair_blocked[key]
    return len(nb & U)

def _construct_direct(dist, points, pair_blocked, time_budget=280.0):
    """
    Priority-queue branch and bound: maintain a frontier of partial solutions,
    always expanding the most promising (smallest size, highest blocking potential).
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V
    best_set = None
    
    # Phase 1: Greedy initial solution
    S_greedy = []
    U = set(range(V))
    for _ in range(500):
        if time.time() > deadline - 40:
            break
        candidates = [p for p in U if _can_add(p, S_greedy, pair_blocked)]
        if not candidates:
            break
        best_p = None
        best_bp = -1
        for p in candidates:
            bp = _blocking_power(p, S_greedy, pair_blocked, U)
            if bp > best_bp:
                best_bp = bp
                best_p = p
        if best_p is None:
            break
        S_greedy.append(best_p)
        nb = set()
        for a in S_greedy[:-1]:
            key = (a, best_p)
            if key in pair_blocked:
                nb |= pair_blocked[key]
        U -= {best_p} | nb
    
    if _is_maximal(S_greedy, pair_blocked):
        best_size = len(S_greedy)
        best_set = list(S_greedy)
    
    # Phase 2: Priority-queue B&B
    # State: (size, S_tuple, U_frozen)
    frontier = []
    if S_greedy:
        heapq.heappush(frontier, (len(S_greedy), tuple(sorted(S_greedy)), frozenset(U)))
    
    max_expansions = 5000
    expansions = 0
    
    while frontier and expansions < max_expansions:
        if time.time() > deadline - 5:
            break
        
        size, S_tuple, U_frozen = heapq.heappop(frontier)
        S = list(S_tuple)
        U = set(U_frozen)
        expansions += 1
        
        if size >= best_size:
            continue
        
        # Check if complete
        if not U:
            if _is_maximal(S, pair_blocked):
                if size < best_size:
                    best_size = size
                    best_set = list(S)
            continue
        
        # Find addable points
        candidates = [p for p in U if _can_add(p, S, pair_blocked)]
        
        if not candidates:
            if _is_maximal(S, pair_blocked):
                if size < best_size:
                    best_size = size
                    best_set = list(S)
            continue
        
        # Score and branch on top candidates
        scored = []
        for p in candidates:
            bp = _blocking_power(p, S, pair_blocked, U)
            scored.append((bp, p))
        scored.sort(key=lambda x: -x[0])
        
        # Branch on top 3
        for bp, p in scored[:3]:
            if time.time() > deadline - 5:
                break
            
            new_S = S + [p]
            nb = set()
            for a in S:
                key = (a, p)
                if key in pair_blocked:
                    nb |= pair_blocked[key]
            new_U = U - {p} - nb
            
            new_size = len(new_S)
            if new_size < best_size:
                heapq.heappush(frontier, (new_size, tuple(sorted(new_S)), frozenset(new_U)))
    
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_direct(dist, points, pair_blocked)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
