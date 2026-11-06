"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from itertools import permutations

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = 24
# Build arrival index lookup: arrival[u][v] = 1-based index of edge (u,v) in the shuffled list
arrival = [[0] * N for _ in range(N)]
for idx, (u, v) in enumerate(P['edge_order'], 1):
    arrival[u][v] = idx
    arrival[v][u] = idx

# Build weight matrix for Hungarian: w[i][j] = arrival index of edge (i,j), infinity if i==j
INF = float('inf')
W = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            W[i][j] = INF
        else:
            W[i][j] = arrival[i][j]


def hungarian(cost):
    """Solve the assignment problem using the Hungarian algorithm (O(n^3)).
    Returns (assignment, total_cost) where assignment[i] = j means row i assigned to col j."""
    n = len(cost)
    # u[i], v[j] are potentials
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row assigned to column j
    way = [0] * (n + 1)
    
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augmenting path
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total_cost = sum(cost[assignment[i]][i] for i in range(n))
    return assignment, total_cost


def extract_cycles(assignment):
    """Extract cycles from a permutation (assignment).
    Returns list of cycles, each cycle is a list of vertices in order."""
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = assignment[j]
            cycles.append(cycle)
    return cycles


def cycle_bottleneck(cycle, arrival):
    """Compute the bottleneck (max arrival index) of a cycle."""
    max_idx = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        max_idx = max(max_idx, arrival[u][v])
    return max_idx


def merge_cycles(cycles, arrival):
    """Merge multiple cycles into a single Hamilton cycle by swapping edges.
    Returns the merged cycle as a list of vertices."""
    if len(cycles) == 1:
        return cycles[0]
    
    # Sort cycles by length
    cycles = sorted(cycles, key=len)
    best_cycle = None
    best_bottleneck = INF
    
    # Try all pairs of cycles to merge
    while len(cycles) > 1:
        merged = False
        best_pair = None
        best_new_bottleneck = INF
        best_merged_cycles = None
        
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                c1 = cycles[i]
                c2 = cycles[j]
                # Try all edge swaps between c1 and c2
                for a1 in range(len(c1)):
                    u1 = c1[a1]
                    v1 = c1[(a1 + 1) % len(c1)]
                    for a2 in range(len(c2)):
                        u2 = c2[a2]
                        v2 = c2[(a2 + 1) % len(c2)]
                        # Swap: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
                        # or (u1,v2) and (v1,u2)
                        # Option 1: (u1,u2) and (v1,v2)
                        cost1 = max(arrival[u1][u2], arrival[v1][v2])
                        # Option 2: (u1,v2) and (v1,u2)
                        cost2 = max(arrival[u1][v2], arrival[v1][u2])
                        
                        # Pick the better option
                        if cost1 <= cost2:
                            new_edges = (cost1, 1)  # option 1
                        else:
                            new_edges = (cost2, 2)  # option 2
                        
                        new_bn = new_edges[0]
                        if new_bn < best_new_bottleneck:
                            best_new_bottleneck = new_bn
                            best_pair = (i, j, a1, a2, new_edges[1])
        
        if best_pair is None:
            break
            
        i, j, a1, a2, option = best_pair
        c1 = cycles[i]
        c2 = cycles[j]
        
        # Perform the merge
        # c1: ... u1 -> v1 ...
        # c2: ... u2 -> v2 ...
        # Option 1: connect u1->u2 and v1->v2
        # Option 2: connect u1->v2 and v1->u2
        
        # Build merged cycle
        # c1 without the edge (c1[a1], c1[(a1+1)%len(c1)])
        # c2 without the edge (c2[a2], c2[(a2+1)%len(c2)])
        
        # Let's rebuild:
        # c1_path = c1[a1+1], c1[a1+2], ..., c1[a1-1] (wrapping)
        # c2_path = c2[a2+1], c2[a2+2], ..., c2[a2-1] (wrapping)
        
        n1 = len(c1)
        n2 = len(c2)
        
        if option == 1:
            # u1=c1[a1], v1=c1[(a1+1)%n1]
            # u2=c2[a2], v2=c2[(a2+1)%n2]
            # New: u1 -> u2 (start c2 from u2) and v1 -> v2 (start c1 from v1)
            # Merged: u2, c2[a2+1], ..., c2[a2-1], v2, v1, c1[(a1+1)+1], ..., c1[a1-1], u1
            # Wait, let me think more carefully.
            # c1: v1 -> c1[(a1+1)%n1] -> ... -> c1[(a1-1)%n1] -> u1 -> v1
            # c2: v2 -> c2[(a2+1)%n2] -> ... -> c2[(a2-1)%n2] -> u2 -> v2
            # Remove edges (u1,v1) and (u2,v2)
            # Add edges (u1,u2) and (v1,v2)
            # So: ... -> u1 -> u2 -> c2[(a2+1)%n2] -> ... -> c2[(a2-1)%n2] -> v2 -> v1 -> c1[(a1+1)%n1] -> ... -> c1[(a1-1)%n1] -> u1
            merged = []
            # Start from u1
            merged.append(c1[a1])  # u1
            # Then u2 and rest of c2
            for k in range(n2):
                merged.append(c2[(a2 + k) % n2])
            # Then v1 and rest of c1 (from v1's successor)
            for k in range(1, n1):
                merged.append(c1[(a1 + k) % n1])
        else:
            # Option 2: (u1,v2) and (v1,u2)
            # ... -> u1 -> v2 -> c2[(a2+1)%n2] -> ... -> c2[(a2-1)%n2] -> u2 -> v1 -> c1[(a1+1)%n1] -> ... -> c1[(a1-1)%n1] -> u1
            merged = []
            merged.append(c1[a1])  # u1
            # Then v2 and rest of c2
            for k in range(n2):
                merged.append(c2[(a2 + k) % n2])
            # Then v1 and rest of c1
            for k in range(1, n1):
                merged.append(c1[(a1 + k) % n1])
        
        # Remove the two cycles and add the merged one
        new_cycles = []
        for idx in range(len(cycles)):
            if idx != i and idx != j:
                new_cycles.append(cycles[idx])
        new_cycles.append(merged)
        cycles = new_cycles
        merged = True
    
    if len(cycles) == 1:
        return cycles[0]
    else:
        # Fallback: just concatenate all cycles (not a valid Hamilton cycle, but we'll fix)
        # This shouldn't happen if merge_cycles works correctly
        result = []
        for c in cycles:
            result.extend(c)
        return result


def find_hamilton_cycle():
    """Find a Hamilton cycle with minimum bottleneck using Hungarian + cycle merging."""
    # Step 1: Solve the assignment problem
    assignment, total_cost = hungarian(W)
    
    # Step 2: Extract cycles
    cycles = extract_cycles(assignment)
    
    if len(cycles) == 1:
        return cycles[0]
    
    # Step 3: Merge cycles
    merged = merge_cycles(cycles, arrival)
    
    # Verify it's a valid Hamilton cycle (all 24 vertices exactly once)
    if len(merged) == N and len(set(merged)) == N:
        return merged
    
    # Fallback: greedy construction
    return greedy_hamilton_cycle()


def greedy_hamilton_cycle():
    """Greedy construction of a Hamilton cycle."""
    used = [False] * N
    cycle = [0]
    used[0] = True
    current = 0
    
    for _ in range(N - 1):
        # Find the neighbor with minimum arrival index
        best_next = -1
        best_idx = INF
        for v in range(N):
            if not used[v]:
                idx = arrival[current][v]
                if idx < best_idx:
                    best_idx = idx
                    best_next = v
        cycle.append(best_next)
        used[best_next] = True
        current = best_next
    
    return cycle


def local_search_improve(cycle, arrival, time_limit=5.0):
    """Improve the Hamilton cycle using 2-opt style moves."""
    start_time = time.time()
    n = len(cycle)
    best_cycle = cycle[:]
    best_bn = cycle_bottleneck(best_cycle, arrival)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse the segment from i to j
                new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                bn = cycle_bottleneck(new_cycle, arrival)
                if bn < best_bn:
                    best_bn = bn
                    best_cycle = new_cycle
                    improved = True
                    if best_bn <= 24:  # Can't do better than 24
                        return best_cycle
        # Also try swapping two non-adjacent vertices
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) == 1 or (i == 0 and j == n - 1):
                    continue
                new_cycle = cycle[:]
                new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]
                bn = cycle_bottleneck(new_cycle, arrival)
                if bn < best_bn:
                    best_bn = bn
                    best_cycle = new_cycle
                    improved = True
                    if best_bn <= 24:
                        return best_cycle
    
    return best_cycle


def construct():
    deadline = 80.0  # Safety margin before 90s limit
    start_time = time.time()
    
    # Get initial cycle from Hungarian + merge
    cycle = find_hamilton_cycle()
    best_cycle = cycle[:]
    best_bn = cycle_bottleneck(best_cycle, arrival)
    
    # Try local search if time permits
    remaining = deadline - (time.time() - start_time)
    if remaining > 3.0:
        improved = local_search_improve(best_cycle, arrival, time_limit=min(remaining - 2.0, 30.0))
        if cycle_bottleneck(improved, arrival) < best_bn:
            best_cycle = improved
            best_bn = cycle_bottleneck(best_cycle, arrival)
    
    # Try multiple random restarts if time permits
    remaining = deadline - (time.time() - start_time)
    rng = random.Random(614)
    while remaining > 5.0:
        # Random greedy with different starting points
        for start in range(N):
            if time.time() - start_time > deadline - 3.0:
                break
            used = [False] * N
            cyc = [start]
            used[start] = True
            cur = start
            for _ in range(N - 1):
                # Pick neighbor with some randomness
                candidates = [v for v in range(N) if not used[v]]
                # Sort by arrival index
                candidates.sort(key=lambda v: arrival[cur][v])
                # Pick from top 3 with some probability
                k = min(3, len(candidates))
                if rng.random() < 0.7:
                    nxt = candidates[0]
                elif k > 1:
                    nxt = candidates[rng.randint(0, k - 1)]
                else:
                    nxt = candidates[0]
                cyc.append(nxt)
                used[nxt] = True
                cur = nxt
            bn = cycle_bottleneck(cyc, arrival)
            if bn < best_bn:
                best_bn = bn
                best_cycle = cyc[:]
                # Try local search on this
                rem2 = deadline - (time.time() - start_time)
                if rem2 > 3.0:
                    imp = local_search_improve(best_cycle, arrival, time_limit=min(rem2 - 2.0, 10.0))
                    if cycle_bottleneck(imp, arrival) < best_bn:
                        best_bn = cycle_bottleneck(imp, arrival)
                        best_cycle = imp
    
    return best_cycle
# EVOLVE-BLOCK-END

