"""Find a Hamilton cycle with minimized max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import random
import sys

_N = 24

# Build the edge arrival lookup from the pinned edge order
_EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# edge_arrival[u][v] = 1-based index in edge_order
_edge_arrival = [[0] * _N for _ in range(_N)]
for idx, (u, v) in enumerate(_EDGE_ORDER, 1):
    _edge_arrival[u][v] = idx
    _edge_arrival[v][u] = idx


def _cycle_score(cycle):
    """Score = max one-based arrival index of edges in the cycle (with closing edge)."""
    n = len(cycle)
    mx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if _edge_arrival[u][v] > mx:
            mx = _edge_arrival[u][v]
    return mx


def _build_adj(k):
    """Build adjacency list and mask using only edges with arrival index <= k."""
    adj = [[] for _ in range(_N)]
    adj_mask = [0] * _N
    for i in range(_N):
        for j in range(i + 1, _N):
            if _edge_arrival[i][j] <= k:
                adj[i].append(j)
                adj[j].append(i)
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj, adj_mask


def _find_cycle_warnsdorff(k, rng=None, timeout=1.0):
    """
    Find a Hamilton cycle using only edges with arrival index <= k.
    Uses iterative Warnsdorff's rule with backtracking and pruning.
    """
    adj, adj_mask = _build_adj(k)
    
    # Check minimum degree
    for i in range(_N):
        if len(adj[i]) < 2:
            return None
    
    # Check connectivity
    visited = [False] * _N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    if count < _N:
        return None
        
    if rng is None:
        rng = random.Random()

    start_time = time.time()
    FULL = (1 << _N) - 1
    
    # We use a stack for backtracking.
    # State: (current_node, visited_mask, path_list)
    # To avoid copying the path list on every step, we maintain a single list and pop/append.
    # However, Warnsdorff often works greedily. We will implement a greedy approach first,
    # and if it fails, we will try a limited backtracking search.
    
    def attempt(start_node, visited_mask, path, max_steps=5000):
        """Greedy Warnsdorff with light backtracking."""
        steps = 0
        stack = [(start_node, visited_mask, path[:])]
        
        while stack:
            if time.time() - start_time > timeout:
                return None
            if steps > max_steps:
                return None
            steps += 1
            
            current, v_mask, p = stack[-1]
            
            if len(p) == _N:
                if _edge_arrival[current][0] <= k:
                    return p
                stack.pop()
                continue
            
            # Get candidates
            candidates = []
            for v in adj[current]:
                if v_mask & (1 << v):
                    continue
                # If this is the last vertex to visit, it must connect to 0
                if len(p) == _N - 1:
                    if _edge_arrival[v][0] <= k:
                        candidates.append(v)
                else:
                    candidates.append(v)
            
            if not candidates:
                stack.pop()
                continue
            
            # Calculate Warnsdorff scores (degree in remaining graph)
            # Remaining unvisited vertices
            remaining = FULL & ~v_mask
            
            scored_candidates = []
            for v in candidates:
                # Count unvisited neighbors of v
                unvisited_neighbors = adj_mask[v] & remaining
                # Exclude v itself from count? No, adj_mask doesn't include self.
                # But we are adding v to visited, so its "available" neighbors for future steps
                # are those in (remaining ^ (1<<v)).
                # Actually, Warnsdorff usually looks at the degree in the *remaining* graph.
                # When we move to v, v becomes visited. The neighbors that matter are those
                # that are still unvisited.
                next_remaining = remaining & ~(1 << v)
                deg = bin(adj_mask[v] & next_remaining).count('1')
                
                # Tie-breaking: add random noise or just use index
                # For deterministic first pass, we can sort by (deg, v)
                # For randomized, we can shuffle ties
                scored_candidates.append((deg, rng.random(), v))
            
            # Sort by degree (ascending), then random
            scored_candidates.sort(key=lambda x: (x[0], x[1]))
            
            # Try candidates in order
            # Optimization: Pruning
            # If we pick v, do any other unvisited vertices become isolated?
            # A vertex u is isolated if it has no neighbors in (path + v) AND no neighbors in (remaining \ v).
            # Since u is in remaining, it has no neighbors in path (by definition of remaining? No, 
            # remaining is just unvisited. Path vertices are visited).
            # So u must have at least one neighbor in (remaining \ v) OR at least two neighbors in (path + v)
            # to potentially close the cycle? 
            # Actually, for a Hamiltonian path extension, u just needs to be reachable.
            # A stronger prune: if u has 0 neighbors in (remaining \ v), it MUST connect to the path.
            # It needs at least 2 connections to the path eventually? No, just 1 to enter, 1 to exit?
            # In a path, internal nodes have 2, endpoints have 1.
            # If u is isolated from the rest of the unvisited graph, it must be an endpoint of the 
            # remaining path segment.
            
            found_next = False
            for deg, _, v in scored_candidates:
                new_mask = v_mask | (1 << v)
                new_remaining = FULL & ~new_mask
                
                # Pruning check
                valid = True
                # Check all vertices in new_remaining
                r = new_remaining
                while r:
                    bit = r & -r
                    u = bit.bit_length() - 1
                    r -= bit
                    # u is unvisited.
                    # Neighbors of u in new_remaining
                    u_unvisited_neighbors = adj_mask[u] & new_remaining
                    if u_unvisited_neighbors == 0:
                        # u is isolated from other unvisited vertices.
                        # It must connect to the path.
                        # Path includes start (0) and current.
                        # In the final cycle, u needs 2 neighbors.
                        # One will be from the path (entry/exit).
                        # If u has no unvisited neighbors, it must be the *last* vertex added to the path 
                        # before closing, OR it must connect to two path vertices?
                        # Actually, if u has no unvisited neighbors, it can only connect to visited vertices.
                        # In a Hamiltonian cycle, every vertex has degree 2.
                        # If u has 0 unvisited neighbors, both its cycle neighbors must be in the visited set.
                        # So u must have at least 2 neighbors in the visited set (new_mask).
                        # Note: The visited set includes the path.
                        # Let's check if u has >= 2 neighbors in new_mask.
                        u_visited_neighbors = adj_mask[u] & new_mask
                        if bin(u_visited_neighbors).count('1') < 2:
                            valid = False
                            break
                if not valid:
                    continue
                    
                # Also check if current vertex's other unvisited neighbors are safe?
                # This is covered by the general check above for all unvisited vertices.
                
                # Commit to v
                p.append(v)
                stack.append((v, new_mask, p))
                found_next = True
                break
            
            if not found_next:
                # Backtrack
                stack.pop()
                
        return None

    # Try multiple starts? No, 0 is fixed for cycle symmetry.
    # Try multiple random restarts for tie-breaking
    for i in range(20):
        result = attempt(0, 1, [0])
        if result:
            return result
            
    return None


def _local_search_2opt(cycle, max_iter=2000, deadline=None):
    """Improve a Hamilton cycle using 2-opt moves to minimize max edge arrival."""
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best)
    
    if deadline is None:
        deadline = time.time() + 5.0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment i+1..j
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s limit
    
    # Get baseline
    baseline = list(range(_N))
    best_cycle = baseline
    best_score = _cycle_score(baseline)
    
    # Binary search for the minimum k where a Hamilton cycle exists
    # Edge indices range from 1 to C(24,2) = 276
    lo, hi = 1, 276
    found_k = None
    
    # First, check if the full graph has a Hamilton cycle (it should, it's K24)
    # Binary search with time budget
    search_budget = 30.0  # seconds for binary search
    search_deadline = start_time + search_budget
    
    while lo <= hi and time.time() < search_deadline:
        mid = (lo + hi) // 2
        
        # Quick check: minimum degree in prefix
        min_deg = _N
        for i in range(_N):
            deg = 0
            for j in range(_N):
                if i != j and _edge_arrival[i][j] <= mid:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
        if min_deg < 2:
            lo = mid + 1
            continue
        
        # Try to find a cycle
        # Use a short timeout for the check in binary search
        timeout = min(2.0, max(0.5, search_deadline - time.time()))
        result = _find_cycle_warnsdorff(mid, timeout=timeout)
        
        if result:
            found_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search found a k, try to find an actual cycle
    if found_k is not None:
        # Try to find a cycle at found_k with more time
        cycle_deadline = min(deadline - 10.0, start_time + 60.0)
        
        for attempt in range(5):
            if time.time() > cycle_deadline:
                break
            timeout = min(10.0, max(1.0, cycle_deadline - time.time()))
            rng = random.Random(attempt * 100 + 42)
            cycle = _find_cycle_warnsdorff(found_k, rng=rng, timeout=timeout)
            if cycle is not None:
                score = _cycle_score(cycle)
                if score < best_score:
                    best_cycle = cycle
                    best_score = score
                    break
        
    # If we haven't made good progress, try local search from random starts
    # This helps if the binary search picked a slightly larger K than optimal
    # or if we want to polish the solution.
    if best_score > 100 and time.time() < deadline - 5.0:
        rng = random.Random(42)
        ls_deadline = deadline - 2.0
        
        for restart in range(10):
            if time.time() > ls_deadline:
                break
            
            # Random permutation
            perm = list(range(_N))
            rng.shuffle(perm)
            
            # Local search
            improved_cycle, improved_score = _local_search_2opt(
                perm, max_iter=500, deadline=min(ls_deadline, time.time() + 2.0)
            )
            
            if improved_score < best_score:
                best_cycle = improved_cycle
                best_score = improved_score
    
    # Final local search on the best cycle found
    if time.time() < deadline - 1.0:
        final_cycle, final_score = _local_search_2opt(
            best_cycle, max_iter=1000, deadline=deadline - 0.5
        )
        if final_score < best_score:
            best_cycle = final_cycle
            best_score = final_score
    
    return best_cycle

# EVOLVE-BLOCK-END
