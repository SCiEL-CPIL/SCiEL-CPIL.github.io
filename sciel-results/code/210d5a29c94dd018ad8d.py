import time
import random
from itertools import combinations

# Precompute the edge order
def _build_edge_order():
    n = 24
    rng = random.Random(614)
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    return edges

_EDGE_ORDER = _build_edge_order()
_EDGE_INDEX = {}
for _i, (_u, _v) in enumerate(_EDGE_ORDER, 1):
    _EDGE_INDEX[(_u, _v)] = _i
    _EDGE_INDEX[(_v, _u)] = _i

N = 24
FULL_MASK = (1 << N) - 1


def _edge_pos(u, v):
    return _EDGE_INDEX[(u, v)]


def _max_edge_in_cycle(cycle):
    """Compute max 1-based arrival index of edges in the Hamilton cycle."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        p = _edge_pos(u, v)
        if p > mx:
            mx = p
    return mx


def _has_ham_cycle(k, adj_bitmasks, adj_lists):
    """
    Check if graph with first k edges has a Hamilton cycle.
    Uses color-coding with randomized vertex hashing.
    Returns True if a Hamilton cycle is detected.
    """
    # Color-coding: assign random colors to vertices, then do DP
    # We need to check multiple random colorings for high probability
    
    # For efficiency, we'll use a single coloring approach with the DP
    # that checks if there's a colorful Hamilton cycle
    
    # Actually, let's use a more direct approach: 
    # DP over subsets - dp[S][v] = True if there's a path visiting exactly S ending at v
    # This is O(2^n * n * n) which for n=24 is ~24^2 * 16.7M = too slow in Python
    
    # Better: use the adjacency structure and a smarter search
    # For the prefix graph, use backtracking with pruning
    
    # Actually for n=24, the subset DP in pure Python is too slow.
    # Let's use a different approach: try to find a cycle using greedy + local search
    
    # Alternative: use the fact that for small k, the graph is sparse
    # For larger k, it's dense and likely has a Hamilton cycle
    
    # Let's use a randomized greedy approach to find a Hamilton cycle in G_k
    # If found, return True; if not after several tries, return False (with some uncertainty)
    
    # Actually, the most reliable approach within time limits:
    # Use a backtracking search with good ordering and pruning
    
    # For the binary search, we need a reliable YES/NO.
    # Let's use: if k is large enough (graph is dense), assume YES.
    # For smaller k, do a bounded search.
    
    # Threshold: if minimum degree >= N/2 and graph is connected, likely Hamiltonian
    # But we need certainty for the binary search to work correctly.
    
    # Let's do a proper backtracking with time limit
    start_time = time.time()
    deadline = 5.0  # seconds per check
    
    # Build adjacency list for the prefix graph
    # adj_lists[v] = list of neighbors connected to v in G_k
    
    # Use backtracking to find Hamilton cycle
    # Start from vertex 0, try to build a path visiting all vertices
    
    # Degree pruning: each vertex must have degree >= 2 in the cycle
    # So if any vertex has degree < 2 in G_k, no Hamilton cycle exists
    for v in range(N):
        if len(adj_lists[v]) < 2:
            return False
    
    # Also check connectivity
    # BFS/DFS to check if graph is connected
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for w in adj_lists[u]:
            if not visited[w]:
                visited[w] = True
                count += 1
                stack.append(w)
    if count < N:
        return False
    
    # Now do backtracking search for Hamilton cycle
    # Fix vertex 0 as start, and to avoid symmetry, fix vertex 1 as second (if connected)
    # Actually, just fix vertex 0 as start
    
    path = [0]
    used = [False] * N
    used[0] = True
    
    def backtrack(pos):
        if time.time() - start_time > deadline:
            return None  # timeout
        
        if pos == N:
            # Check if last vertex connects back to first
            last = path[-1]
            first = path[0]
            if first in adj_lists[last]:
                return list(path)
            return None
        
        cur = path[-1]
        # Try neighbors in order of increasing degree (more constrained first)
        neighbors = sorted(adj_lists[cur], key=lambda x: len(adj_lists[x]))
        
        for nxt in neighbors:
            if not used[nxt]:
                # Pruning: if nxt is the last unvisited vertex, it must connect to start
                remaining = N - pos - 1
                if remaining == 1:
                    if path[0] not in adj_lists[nxt]:
                        continue
                path.append(nxt)
                used[nxt] = True
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                path.pop()
                used[nxt] = False
        
        return None
    
    result = backtrack(1)
    return result is not None


def _find_cycle_in_prefix(k, adj_lists):
    """Find an actual Hamilton cycle in G_k using backtracking."""
    start_time = time.time()
    deadline = 8.0
    
    path = [0]
    used = [False] * N
    used[0] = True
    
    def backtrack(pos):
        if time.time() - start_time > deadline:
            return None
        
        if pos == N:
            last = path[-1]
            first = path[0]
            if first in adj_lists[last]:
                return list(path)
            return None
        
        cur = path[-1]
        neighbors = sorted(adj_lists[cur], key=lambda x: len(adj_lists[x]))
        
        for nxt in neighbors:
            if not used[nxt]:
                remaining = N - pos - 1
                if remaining == 1:
                    if path[0] not in adj_lists[nxt]:
                        continue
                path.append(nxt)
                used[nxt] = True
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                path.pop()
                used[nxt] = False
        
        return None
    
    result = backtrack(1)
    return result


def _build_adj_lists(k):
    """Build adjacency lists for graph with first k edges."""
    adj_lists = [[] for _ in range(N)]
    for i in range(k):
        u, v = _EDGE_ORDER[i]
        adj_lists[u].append(v)
        adj_lists[v].append(u)
    return adj_lists


def construct():
    global _EDGE_ORDER, _EDGE_INDEX
    
    start_time = time.time()
    total_deadline = 80.0  # keep margin before 90s limit
    
    # First, try to find a good cycle using local search on a random cycle
    # Then try to improve by binary searching for the minimum k
    
    # Step 1: Get a baseline cycle using greedy + 2-opt
    rng = random.Random(42)
    
    # Start with a random permutation
    best_cycle = list(range(N))
    rng.shuffle(best_cycle)
    best_score = _max_edge_in_cycle(best_cycle)
    
    # 2-opt improvement
    def two_opt(cycle):
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for j in range(i + 2, N):
                    if j == (i + N - 1) % N:
                        continue
                    # Check if reversing segment i..j improves
                    a = cycle[i]
                    b = cycle[(i + 1) % N]
                    c = cycle[j]
                    d = cycle[(j + 1) % N]
                    
                    old_cost = _edge_pos(a, b) + _edge_pos(c, d)
                    new_cost = _edge_pos(a, c) + _edge_pos(b, d)
                    
                    if new_cost < old_cost:
                        # Reverse segment i+1..j
                        cycle[i+1:j+1] = reversed(cycle[i+1:j+1])
                        improved = True
        return cycle
    
    # Apply 2-opt with time limit
    opt_start = time.time()
    while time.time() - opt_start < 10:
        two_opt(best_cycle)
        new_score = _max_edge_in_cycle(best_cycle)
        if new_score < best_score:
            best_score = new_score
        else:
            break
    
    # Step 2: Binary search for minimum k where Hamilton cycle exists
    # Lower bound: need at least N edges (a cycle has N edges)
    # Upper bound: best_score
    lo = N
    hi = best_score
    
    # Verify hi is valid
    adj_hi = _build_adj_lists(hi)
    if not _has_ham_cycle(hi, None, adj_hi):
        # Shouldn't happen, but handle gracefully
        return best_cycle
    
    # Binary search
    best_k = hi
    best_cycle_at_k = best_cycle
    
    while lo <= hi and time.time() - start_time < total_deadline - 10:
        mid = (lo + hi) // 2
        adj_mid = _build_adj_lists(mid)
        
        if _has_ham_cycle(mid, None, adj_mid):
            best_k = mid
            # Try to find the actual cycle
            cycle_found = _find_cycle_in_prefix(mid, adj_mid)
            if cycle_found is not None:
                best_cycle_at_k = cycle_found
                # Try to improve with 2-opt
                opt_start2 = time.time()
                while time.time() - opt_start2 < 3:
                    two_opt(best_cycle_at_k)
                    s = _max_edge_in_cycle(best_cycle_at_k)
                    if s <= mid:
                        best_cycle_at_k = best_cycle_at_k[:]
                    else:
                        break
                hi = mid - 1
            else:
                hi = mid - 1
        else:
            lo = mid + 1
    
    # Final: use the best cycle we found
    # Make sure it's a valid Hamilton cycle
    final_cycle = best_cycle_at_k
    final_score = _max_edge_in_cycle(final_cycle)
    
    # If the binary search didn't find anything better, use the 2-opt result
    if final_score > best_score:
        final_cycle = best_cycle
        final_score = best_score
    
    # One more round of 2-opt on the final cycle
    opt_start3 = time.time()
    while time.time() - opt_start3 < 5 and time.time() - start_time < total_deadline - 2:
        two_opt(final_cycle)
        s = _max_edge_in_cycle(final_cycle)
        if s < final_score:
            final_score = s
    
    return final_cycle
