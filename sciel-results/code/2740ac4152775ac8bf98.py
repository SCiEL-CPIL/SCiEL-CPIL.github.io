import time
import random
import numpy as np

N = 24
TOTAL_EDGES = N * (N - 1) // 2  # 276

def _build_edge_order():
    rng = random.Random(614)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(edges)
    return edges

_EDGE_ORDER = _build_edge_order()
_EDGE_INDEX = {}
for _i, (_u, _v) in enumerate(_EDGE_ORDER, 1):
    _EDGE_INDEX[(_u, _v)] = _i
    _EDGE_INDEX[(_v, _u)] = _i


def _edge_pos(u, v):
    return _EDGE_INDEX[(u, v)]


def _cycle_score(cycle):
    """Return the max 1-based arrival index of edges in the cycle."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        p = _edge_pos(u, v)
        if p > mx:
            mx = p
    return mx


def _build_adj_lists(k):
    """Build adjacency lists for graph with first k edges."""
    adj = [[] for _ in range(N)]
    for i in range(k):
        u, v = _EDGE_ORDER[i]
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _has_ham_cycle(k, adj_lists):
    """
    Check if graph with first k edges has a Hamilton cycle.
    Uses Dirac's theorem for dense graphs, backtracking for sparse.
    """
    # Quick check: Dirac's condition - if min degree >= N/2, Hamiltonian
    min_deg = min(len(adj) for adj in adj_lists)
    if min_deg >= N // 2:
        return True
    
    # Check connectivity
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
    
    # Check min degree >= 2
    if min_deg < 2:
        return False
    
    # Use backtracking with time limit
    start_time = time.time()
    deadline = 3.0
    
    # Convert adj_lists to sets for fast lookup
    adj_sets = [set(adj) for adj in adj_lists]
    
    path = [0]
    used = [False] * N
    used[0] = True
    found = [False]
    
    def backtrack(pos):
        if found[0]:
            return
        if time.time() - start_time > deadline:
            return
        
        if pos == N:
            last = path[-1]
            first = path[0]
            if first in adj_sets[last]:
                found[0] = True
            return
        
        cur = path[-1]
        # Try neighbors in order of increasing degree (more constrained first)
        neighbors = sorted(adj_sets[cur], key=lambda x: len(adj_sets[x]))
        
        for nxt in neighbors:
            if not used[nxt]:
                remaining = N - pos - 1
                if remaining == 1:
                    if path[0] not in adj_sets[nxt]:
                        continue
                path.append(nxt)
                used[nxt] = True
                backtrack(pos + 1)
                if found[0]:
                    return
                path.pop()
                used[nxt] = False
    
    backtrack(1)
    return found[0]


def _find_cycle_in_prefix(k, adj_lists, time_limit=5.0):
    """Find an actual Hamilton cycle in G_k using backtracking."""
    start_time = time.time()
    
    adj_sets = [set(adj) for adj in adj_lists]
    
    path = [0]
    used = [False] * N
    used[0] = True
    result = [None]
    
    def backtrack(pos):
        if result[0] is not None:
            return
        if time.time() - start_time > time_limit:
            return
        
        if pos == N:
            last = path[-1]
            first = path[0]
            if first in adj_sets[last]:
                result[0] = list(path)
            return
        
        cur = path[-1]
        neighbors = sorted(adj_sets[cur], key=lambda x: len(adj_sets[x]))
        
        for nxt in neighbors:
            if not used[nxt]:
                remaining = N - pos - 1
                if remaining == 1:
                    if path[0] not in adj_sets[nxt]:
                        continue
                path.append(nxt)
                used[nxt] = True
                backtrack(pos + 1)
                if result[0] is not None:
                    return
                path.pop()
                used[nxt] = False
    
    backtrack(1)
    return result[0]


def _local_search_cycle(cycle, time_limit=10.0):
    """
    Local search to minimize the maximum edge position in the cycle.
    Uses 2-opt and single vertex repositioning.
    """
    start = time.time()
    best = cycle[:]
    best_score = _cycle_score(best)
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try 2-opt moves
        for i in range(N):
            if time.time() - start > time_limit:
                break
            for j in range(i + 2, N):
                if j == (i + N - 1) % N:
                    continue
                a = best[i]
                b = best[(i + 1) % N]
                c = best[j]
                d = best[(j + 1) % N]
                
                old_cost = _edge_pos(a, b) + _edge_pos(c, d)
                new_cost = _edge_pos(a, c) + _edge_pos(b, d)
                
                if new_cost < old_cost:
                    best[i+1:j+1] = reversed(best[i+1:j+1])
                    new_score = _cycle_score(best)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
        
        # Try single vertex repositioning
        if not improved and time.time() - start < time_limit:
            for pos in range(N):
                if time.time() - start > time_limit:
                    break
                v = best[pos]
                # Remove v from position pos
                candidate = best[:pos] + best[pos+1:]
                # Try inserting v at each other position
                for new_pos in range(len(candidate)):
                    trial = candidate[:new_pos] + [v] + candidate[new_pos:]
                    s = _cycle_score(trial)
                    if s < best_score:
                        best = trial
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best, best_score


def construct():
    global _EDGE_ORDER, _EDGE_INDEX
    
    start_time = time.time()
    total_deadline = 85.0  # keep margin before 90s limit
    
    # Step 1: Get a baseline cycle using 2-opt
    rng = random.Random(42)
    best_cycle = list(range(N))
    rng.shuffle(best_cycle)
    best_score = _cycle_score(best_cycle)
    
    # Apply local search
    opt_start = time.time()
    best_cycle, best_score = _local_search_cycle(best_cycle, time_limit=15.0)
    
    # Step 2: Binary search for minimum k where Hamilton cycle exists
    lo = N
    hi = best_score
    
    best_k = hi
    best_cycle_at_k = best_cycle[:]
    
    # Binary search
    while lo <= hi and time.time() - start_time < total_deadline - 15:
        mid = (lo + hi) // 2
        adj_mid = _build_adj_lists(mid)
        
        if _has_ham_cycle(mid, adj_mid):
            best_k = mid
            cycle_found = _find_cycle_in_prefix(mid, adj_mid, time_limit=5.0)
            if cycle_found is not None:
                # Improve with local search
                improved_cycle, improved_score = _local_search_cycle(cycle_found, time_limit=5.0)
                if improved_score <= mid:
                    best_cycle_at_k = improved_cycle
                else:
                    best_cycle_at_k = cycle_found
                hi = mid - 1
            else:
                hi = mid - 1
        else:
            lo = mid + 1
    
    # Final: use the best cycle we found
    final_cycle = best_cycle_at_k
    final_score = _cycle_score(final_cycle)
    
    # If the binary search didn't find anything better, use the local search result
    if final_score > best_score:
        final_cycle = best_cycle
        final_score = best_score
    
    # One more round of local search on the final cycle
    remaining_time = total_deadline - (time.time() - start_time) - 3.0
    if remaining_time > 1.0:
        final_cycle, final_score = _local_search_cycle(final_cycle, time_limit=remaining_time)
    
    return final_cycle
