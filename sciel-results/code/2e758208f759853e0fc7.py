"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
NUM_POINTS = N * N

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Precompute squared distances
def _get_dist_matrix():
    dist = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
    for i in range(NUM_POINTS):
        x1, y1 = divmod(i, N)
        for j in range(i + 1, NUM_POINTS):
            x2, y2 = divmod(j, N)
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            dist[i][j] = d
            dist[j][i] = d
    return dist

DIST = _get_dist_matrix()

def _is_valid(S):
    """Check if S is a no-isosceles set."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        p = S_list[i]
        seen_dists = set()
        for j in range(n):
            if i == j:
                continue
            q = S_list[j]
            d = DIST[p][q]
            if d in seen_dists:
                return False
            seen_dists.add(d)
    return True

def _is_maximal(S):
    """Check if S is maximal."""
    S_list = list(S)
    n = len(S_list)
    S_set = set(S_list)
    
    for p in range(NUM_POINTS):
        if p in S_set:
            continue
        
        blocked = False
        
        # Case 1: Apex at p
        dists_from_p = [DIST[p][q] for q in S_list]
        if len(dists_from_p) != len(set(dists_from_p)):
            blocked = True
        
        if not blocked:
            # Case 2: Apex at i in S
            for i in S_list:
                d_ip = DIST[i][p]
                for j in S_list:
                    if i == j:
                        continue
                    if DIST[i][j] == d_ip:
                        blocked = True
                        break
                if blocked:
                    break
        
        if not blocked:
            # Case 3: Apex at j in S
            for j in S_list:
                d_jp = DIST[j][p]
                for i in S_list:
                    if j == i:
                        continue
                    if DIST[j][i] == d_jp:
                        blocked = True
                        break
                if blocked:
                    break

        if not blocked:
            return False
            
    return True

def _construct_greedy():
    """Greedy construction of a maximal no-isosceles set."""
    S = set()
    points = list(range(NUM_POINTS))
    random.shuffle(points)
    
    for p in points:
        # Check if p can be added to S
        new_S = S | {p}
        if _is_valid(new_S):
            S = new_S
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_S = None
    best_size = NUM_POINTS + 1
    
    while time.time() < deadline:
        S = _construct_greedy()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for p in S_list:
                if time.time() > deadline:
                    break
                
                new_S = S - {p}
                
                if _is_maximal(new_S):
                    S = new_S
                    improved = True
                    break
            
        if len(S) < best_size:
            best_size = len(S)
            best_S = S

    if best_S is None:
        best_S = _construct_greedy()
        
    return [(p // N, p % N) for p in best_S]
# EVOLVE-BLOCK-END
