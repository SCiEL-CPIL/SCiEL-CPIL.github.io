"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def _hopcroft_karp(adj, n_left, n_right):
    """Hopcroft-Karp bipartite matching. adj[i] = list of right nodes adjacent to left node i (0-indexed)."""
    from collections import deque
    
    match_l = [-1] * n_left
    match_r = [-1] * n_right
    dist = [0] * n_left
    
    def bfs():
        queue = deque()
        found = False
        for u in range(n_left):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1:
                    found = True
                elif dist[mu] == -1:
                    dist[mu] = dist[u] + 1
                    queue.append(mu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            mu = match_r[v]
            if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _compute_f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, 200):
        # Build adjacency: left nodes 0..n-1 represent i=1..n
        # Right nodes 0..L-1 represent numbers m+1, m+2, ..., m+L
        adj = [[] for _ in range(n)]
        for i in range(1, n + 1):
            for j in range(L):
                if (m + 1 + j) % i == 0:
                    adj[i - 1].append(j)
        if _hopcroft_karp(adj, n, L) == n:
            return L
    return 144  # should not reach here

LCM_12 = 27720  # LCM(1,2,...,12)

def _search_optimal_m(deadline):
    """Search for m in [0, LCM_12) that maximizes f(12, m)."""
    best_m = 0
    best_f = 0
    
    # Phase 1: Sample residue classes at multiples of 1, 2, 3, ..., 12
    # and their offsets
    candidates = set()
    for step in range(1, 13):
        for offset in range(step):
            candidates.add(offset * LCM_12 // step + offset)
    
    # Also add some random-ish structured candidates
    for base in range(0, LCM_12, 100):
        for delta in range(-5, 6):
            c = (base + delta) % LCM_12
            candidates.add(c)
    
    # Add specific interesting residues: near multiples of small numbers
    for i in range(1, 13):
        for k in range(LCM_12 // i):
            pos = k * i
            candidates.add(pos % LCM_12)
            candidates.add((pos + 1) % LCM_12)
            candidates.add((pos - 1) % LCM_12)
    
    # Sort and evaluate
    candidates = sorted(candidates)
    
    for m in candidates:
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            # If we found a high value, do local search around it
            if best_f >= 20:
                for dm in range(-10, 11):
                    if time.time() > deadline:
                        break
                    m2 = (m + dm) % LCM_12
                    if m2 not in candidates:
                        f2 = _compute_f(m2)
                        if f2 > best_f:
                            best_f = f2
                            best_m = m2
    
    return best_m, best_f

def construct():
    start_time = time.time()
    deadline = start_time + 70  # Safety margin before 90s kill
    
    m, f_val = _search_optimal_m(deadline)
    
    # Verify it's a proper int
    if not isinstance(m, int) or isinstance(m, bool):
        m = int(m)
    
    return m
# EVOLVE-BLOCK-END
