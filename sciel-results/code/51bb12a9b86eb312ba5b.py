import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
# Fixed edge order from the task description
EDGE_ORDER = [
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]

N = 24
TIME_LIMIT = 85.0  # Safety margin before 90s limit

# Precompute edge index lookup: (u,v) -> 1-based arrival index
EDGE_INDEX = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(_u, _v)] = _i
    EDGE_INDEX[(_v, _u)] = _i

def cycle_score(cycle):
    """Compute the max edge arrival index for a Hamilton cycle."""
    max_idx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        idx = EDGE_INDEX[(u, v)]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def find_cycle_dfs(K, start_node=0, time_budget=2.0):
    """
    Finds a Hamilton cycle using only edges from the first K in EDGE_ORDER.
    Uses optimized DFS with bitmasks and Warnsdorff's heuristic.
    """
    start_time = time.time()
    # Build adjacency bitmasks for the first K edges
    adj = [0] * N
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Check min degree
    for i in range(N):
        if adj[i].bit_count() < 2:
            return None

    path = [start_node]
    v_mask = 1 << start_node
    
    def dfs_opt(current, depth):
        if time.time() - start_time > time_budget:
            return False
            
        if depth == N:
            if adj[current] & (1 << start_node):
                return True
            return False
        
        candidates = []
        m = adj[current] & ~v_mask
        while m:
            bit = m & (-m)
            v = bit.bit_length() - 1
            unvisited_count = (adj[v] & ~v_mask).bit_count()
            candidates.append((unvisited_count, v))
            m ^= bit
        
        if not candidates:
            return False
            
        candidates.sort()
        
        for _, v in candidates:
            path.append(v)
            if dfs_opt(v, depth + 1):
                return True
            path.pop()
            
        return False

    if dfs_opt(start_node, 1):
        return path
    return None

def polish_cycle(cycle, time_budget=10.0):
    """
    Try to reduce the max edge index by 2-opt style swaps.
    """
    start = time.time()
    best_cycle = list(cycle)
    best_score = cycle_score(best_cycle)
    
    # Precompute edge indices for the current cycle for faster access?
    # Or just compute on the fly.
    
    while time.time() - start < time_budget:
        improved = False
        current_score = best_score
        
        # Identify edges with index == current_score
        max_edges = []
        for i in range(N):
            u = best_cycle[i]
            v = best_cycle[(i + 1) % N]
            idx = EDGE_INDEX[(u, v)]
            if idx == current_score:
                max_edges.append(i)
        
        # Try to remove one of the max edges
        for i in max_edges:
            u = best_cycle[i]
            v = best_cycle[(i + 1) % N]
            
            # Try 2-opt with any other edge j
            for j in range(N):
                if j == i or j == (i - 1) % N:
                    continue
                
                x = best_cycle[j]
                y = best_cycle[(j + 1) % N]
                
                # Option 1: (u, x) and (v, y)
                # Check if edges exist and have index < current_score
                idx1 = EDGE_INDEX.get((u, x))
                idx2 = EDGE_INDEX.get((v, y))
                if idx1 is not None and idx2 is not None and idx1 < current_score and idx2 < current_score:
                    # Perform 2-opt
                    # Cycle: ... u, v ... x, y ...
                    # New: ... u, x ... y, v ... (reversing segment between v and x? No, standard 2-opt)
                    # Standard 2-opt on (u, v) and (x, y) removes (u, v) and (x, y)
                    # Adds (u, x) and (v, y).
                    # This reverses the segment from v to x.
                    # Let's construct the new cycle.
                    # Path: u, v, ..., x, y
                    # New Path: u, x, ..., v, y
                    # Wait, the cycle is circular.
                    # Let's use the reverse segment method.
                    # Segment from i+1 to j (if j > i) or i+1 to N+j (if j < i).
                    # Simpler:
                    # Create new cycle by reversing the segment between v and x?
                    # Actually, just build the new list.
                    # Indices: i is (u, v). j is (x, y).
                    # We want to connect u->x and v->y.
                    # This implies the order is u, x, ..., v, y?
                    # No, the segment between v and x is reversed.
                    # Let's just use the `local_search_polish` logic from the original code which was correct for 2-opt.
                    
                    # Re-implementing 2-opt carefully
                    # Original: ... u, v, ..., x, y ...
                    # New: ... u, x, ..., v, y ...
                    # This is valid if we reverse the segment from v to x.
                    
                    # Let's just use the generic 2-opt swap.
                    new_cycle = list(best_cycle)
                    # Reverse segment between i+1 and j
                    # If j > i:
                    # Segment is best_cycle[i+1 : j+1]
                    # Reverse it.
                    # This changes (u, v) to (u, rev_v) and (x, y) to (rev_x, y)?
                    # Let's stick to the simple reverse in the original code.
                    
                    # The original code:
                    # new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                    # This reverses the segment starting at i.
                    # Edge (i-1, i) becomes (i-1, rev_i).
                    # Edge (j, j+1) becomes (rev_j, j+1).
                    # This is a valid 2-opt.
                    
                    # We want to remove edge (u, v) which is at index i.
                    # So we reverse segment starting at i.
                    # New edge at i is (u, rev_v).
                    # New edge at j is (rev_j, y).
                    # We need (u, rev_v) and (rev_j, y) to be low index.
                    # rev_v is the vertex at position j in the reversed segment?
                    # If we reverse [i..j], the vertex at i becomes the old vertex at j.
                    # So new edge (u, old_j).
                    # And new edge (old_i, y)? No.
                    # Let's trace:
                    # Cycle: A, B, C, D, E
                    # i=1 (B, C). j=3 (D, E).
                    # Reverse [1..3]: B, C, D -> D, C, B.
                    # New Cycle: A, D, C, B, E.
                    # Edges: (A, D), (D, C), (C, B), (B, E), (E, A).
                    # Removed: (B, C) and (D, E).
                    # Added: (A, D) and (B, E).
                    # So we are checking if (A, D) and (B, E) are good.
                    # A = u, D = x, B = v, E = y.
                    # So we check (u, x) and (v, y).
                    
                    # So the check `idx1 = (u, x)` and `idx2 = (v, y)` is correct for reversing [i..j].
                    
                    new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                    new_score = cycle_score(new_cycle)
                    if new_score < best_score:
                        best_cycle = new_cycle
                        best_score = new_score
                        improved = True
                        break # Restart loop with new cycle
                else:
                    # Check Option 2: (u, y) and (v, x)
                    # This corresponds to reversing [i..j] but in a different way?
                    # No, standard 2-opt only adds (u, x) and (v, y) for the reversal.
                    # The other combination (u, y) and (v, x) is not a simple 2-opt reversal of a contiguous segment.
                    # It would require crossing edges, which might not be a single 2-opt.
                    # So we only check Option 1.
                    pass
        
        if not improved:
            # Try random 2-opt moves that don't increase score
            for _ in range(100):
                if time.time() - start > time_budget:
                    break
                i = random.randint(0, N-1)
                j = random.randint(0, N-1)
                if i == j or i == (j+1)%N or j == (i+1)%N:
                    continue
                new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score <= best_score:
                    best_cycle = new_cycle
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                # If no improvement in a while, break
                # To avoid infinite loop, we can break if no improvement in last X iterations
                # But for now, just break if no improvement in this pass
                break
                
    return best_cycle

def construct():
    start_time = time.time()
    
    # Binary search for minimal K
    lo = N
    hi = N * (N - 1) // 2
    best_cycle = None
    best_K = hi
    
    while lo <= hi:
        if time.time() - start_time > TIME_LIMIT - 15: # Reserve 15s for polishing
            break
            
        mid = (lo + hi) // 2
        cycle = find_cycle_dfs(mid, time_budget=2.0)
        
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            # Try a second, more aggressive search if it failed?
            # For now, just move on.
            lo = mid + 1

    if best_cycle:
        # Polish the cycle
        if time.time() - start_time < TIME_LIMIT - 2:
            budget = TIME_LIMIT - 2 - (time.time() - start_time)
            best_cycle = polish_cycle(best_cycle, time_budget=budget)
        return best_cycle
    else:
        return list(range(N))

# EVOLVE-BLOCK-END
