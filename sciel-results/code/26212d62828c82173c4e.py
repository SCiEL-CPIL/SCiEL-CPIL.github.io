"""Find Hamilton cycle with minimal max edge arrival index."""
import time
import sys
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

# Precompute adjacency with edge positions
adj_with_pos = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    adj_with_pos[u].append((v, i))
    adj_with_pos[v].append((u, i))
for v in range(N):
    adj_with_pos[v].sort(key=lambda x: x[1])

# Precompute all masks for binary search
def precompute_masks():
    """Precompute adjacency masks for each K."""
    masks_list = []
    for K in range(N, len(EDGE_ORDER) + 1):
        masks = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        masks_list.append(masks)
    return masks_list

MASKS = precompute_masks()

def check_connectivity(masks):
    """Check if graph with given masks is connected."""
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        neighbors = masks[u]
        while neighbors:
            lsb = neighbors & (-neighbors)
            nb = lsb.bit_length() - 1
            neighbors ^= lsb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    return count == N

def find_2_factor_milp(K, deadline):
    """Find a 2-factor in G_K using MILP."""
    # Variables x_e for each edge e
    # We only consider edges with index <= K
    active_edges = []
    for i in range(K):
        u, v = EDGE_ORDER[i]
        active_edges.append((u, v))
    
    M = len(active_edges)
    if M == 0:
        return None
    
    # c = 0 (feasibility problem)
    c = np.zeros(M)
    
    # A = degree constraints
    # For each vertex v, sum of x_e for e incident to v = 2
    A = np.zeros((N, M))
    for j, (u, v) in enumerate(active_edges):
        A[u, j] = 1
        A[v, j] = 1
    
    lb = np.ones(N) * 2
    ub = np.ones(N) * 2
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds for x: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))
    
    # Integrality: all integer
    integrality = np.ones(M)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={'time_limit': 5})
        if res.status != 0:
            return None
        x = res.x
        # Filter edges
        edges = []
        for j, (u, v) in enumerate(active_edges):
            if x[j] > 0.5:
                edges.append((u, v))
        return edges
    except Exception:
        return None

def get_cycles_from_edges(edges):
    """Extract cycles from a list of edges (2-factor)."""
    adj = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = set()
    cycles = []
    for i in range(N):
        if i not in visited:
            cycle = []
            curr = i
            prev = -1
            while True:
                visited.add(curr)
                cycle.append(curr)
                nexts = adj[curr]
                next_node = -1
                for nb in nexts:
                    if nb != prev:
                        next_node = nb
                        break
                if next_node == -1:
                    break
                if next_node == i:
                    break
                prev, curr = curr, next_node
            if len(cycle) > 1:
                cycles.append(cycle)
    return cycles

def merge_cycles(cycles, K, deadline):
    """Try to merge cycles into a single Hamilton cycle using edges in G_K."""
    # Create a set of edges in G_K for quick lookup
    # Note: We need to be careful with direction.
    # We have undirected edges.
    g_k_edges = set()
    for i in range(K):
        u, v = EDGE_ORDER[i]
        g_k_edges.add((min(u,v), max(u,v)))
    
    def edge_exists(u, v):
        return (min(u,v), max(u,v)) in g_k_edges
    
    # While we have more than 1 cycle
    changed = True
    while len(cycles) > 1 and time.time() < deadline - 2:
        changed = False
        # Try to merge any pair of cycles
        for i in range(len(cycles)):
            for j in range(i+1, len(cycles)):
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Try all pairs of edges in c1 and c2 to see if we can swap
                # Remove (u, v) from c1 and (x, y) from c2
                # Add (u, x) and (v, y) ? Or (u, y) and (v, x)?
                # We need to maintain degree 2 and form a single cycle.
                # Standard 2-opt:
                # c1: ... u -> v ...
                # c2: ... x -> y ...
                # Remove u->v, x->y. Add u->x, v->y.
                # This merges if the resulting graph is a single cycle.
                
                found_swap = False
                for k in range(len(c1)):
                    u = c1[k]
                    v = c1[(k+1)%len(c1)]
                    for l in range(len(c2)):
                        x = c2[l]
                        y = c2[(l+1)%len(c2)]
                        
                        if edge_exists(u, x) and edge_exists(v, y):
                            # Perform swap
                            # New c1 is c1 with u-v removed, u-x added, v-y added?
                            # Actually, it's easier to construct the new cycle.
                            # Path in c1 from v to u (excluding v, u at ends? No, u->v is edge)
                            # c1: u, ..., v
                            # c2: x, ..., y
                            # New: u, x, ..., y, v, ..., u
                            
                            # Extract path u->...->v in c1
                            # c1 is a list. u is at k, v is at k+1.
                            # Path u->...->v is c1[k], c1[k+1]... wait.
                            # c1: [u, v, ...]
                            # We remove edge (u,v).
                            # We have path from v to u in c1 (the rest of the cycle).
                            # Let's say c1 is [u, v, a, b, u].
                            # Remove (u,v). Path v->a->b->u.
                            # c2: [x, y, c, d, x].
                            # Remove (x,y). Path y->c->d->x.
                            # Add (u,x) and (v,y).
                            # New cycle: u, x, d, c, y, v, b, a, u.
                            
                            # Construct new cycle
                            # Path 1: v to u in c1 (reverse of u to v)
                            # c1[k] = u, c1[k+1] = v.
                            # The path from v to u is c1[k+1], c1[k+2], ..., c1[k-1] (mod len)
                            path1 = []
                            idx = (k+1) % len(c1)
                            while idx != k:
                                path1.append(c1[idx])
                                idx = (idx+1) % len(c1)
                            # path1 is [v, ..., neighbor_of_u]
                            
                            # Path 2: y to x in c2
                            # c2[l] = x, c2[l+1] = y.
                            # Path from y to x is c2[l+1], ..., c2[l-1]
                            path2 = []
                            idx = (l+1) % len(c2)
                            while idx != l:
                                path2.append(c2[idx])
                                idx = (idx+1) % len(c2)
                            # path2 is [y, ..., neighbor_of_x]
                            
                            # New cycle: u, x, path2 (excluding y? No, path2 starts with y), v, path1 (excluding v? No, path1 starts with v)
                            # Wait.
                            # u is connected to x.
                            # x is connected to path2[0] (which is y? No, path2[0] is y).
                            # So u-x-y...
                            # Let's trace:
                            # u -> x (new edge)
                            # x -> ... -> y (path from x to y in c2, which is reverse of path2? No.)
                            # In c2, x->y is removed. The rest is path y->...->x.
                            # So x is connected to the last node of path2?
                            # path2 is [y, ..., last]. Last is neighbor of x.
                            # So x is connected to last.
                            # So we have u-x-last...y.
                            # Then y is connected to v (new edge).
                            # So u-x...y-v.
                            # Then v is connected to path1[1]...
                            # path1 is [v, ..., first]. First is neighbor of u.
                            # So v is connected to first? No, v is connected to next in path1.
                            # path1[0] is v.
                            # So v -> path1[1] ... -> path1[-1] (neighbor of u).
                            # And path1[-1] is connected to u.
                            
                            new_cycle = [u]
                            # Traverse x to y in c2 (excluding x, y? No)
                            # We need the path from x to y in c2.
                            # c2: x, y, ...
                            # Path x->...->y is c2[l], c2[l+1]... NO.
                            # c2[l]=x, c2[l+1]=y.
                            # The path from x to y via the other way is c2[l], c2[l-1], ..., c2[l+1].
                            # Let's just list nodes in c2 in order starting from x, going backwards to y.
                            # c2 is a cycle.
                            # Nodes: x, prev(x), ..., y.
                            # Let's find indices.
                            # x is at l. y is at l+1.
                            # Reverse path from x to y: l, l-1, ..., l+1.
                            rev_path2 = []
                            idx = l
                            while True:
                                rev_path2.append(c2[idx])
                                if c2[idx] == y:
                                    break
                                idx = (idx - 1) % len(c2)
                            # rev_path2 is [x, ..., y]
                            
                            # Path from y to v is direct edge.
                            # Path from v to u in c1.
                            # v is at k+1. u is at k.
                            # Reverse path from v to u: k+1, k+2, ..., k.
                            rev_path1 = []
                            idx = (k+1) % len(c1)
                            while True:
                                rev_path1.append(c1[idx])
                                if c1[idx] == u:
                                    break
                                idx = (idx + 1) % len(c1)
                            # rev_path1 is [v, ..., u]
                            
                            # New cycle: rev_path2 (x..y) + rev_path1[1:] (v..u, excluding u which is start)
                            # Wait, u is start.
                            # u -> x ... y -> v ... u
                            # So: [u] + rev_path2[1:] + [y] + rev_path1[1:]
                            # rev_path2[0] is x. rev_path2[-1] is y.
                            # rev_path1[0] is v. rev_path1[-1] is u.
                            
                            new_cycle = [u] + rev_path2[1:] + rev_path1[1:-1] + [u] # Wait, u is already at start?
                            # Let's construct carefully.
                            # Start at u.
                            # Go to x (start of rev_path2).
                            # Traverse rev_path2 from x to y.
                            # Go to v (start of rev_path1).
                            # Traverse rev_path1 from v to u.
                            
                            new_cycle = [u]
                            new_cycle.extend(rev_path2[1:]) # x's neighbors to y
                            new_cycle.extend(rev_path1[1:]) # v's neighbors to u
                            # The last element is u.
                            
                            if len(new_cycle) == len(c1) + len(c2):
                                # Verify it's a valid cycle
                                # Check edges
                                valid = True
                                for idx in range(len(new_cycle)):
                                    a = new_cycle[idx]
                                    b = new_cycle[(idx+1)%len(new_cycle)]
                                    if not edge_exists(a, b):
                                        valid = False
                                        break
                                if valid:
                                    # Replace c1, c2 with new_cycle
                                    cycles[i] = new_cycle
                                    del cycles[j]
                                    changed = True
                                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
            
    if len(cycles) == 1:
        return cycles[0]
    return None

def dfs_find_cycle(masks, deadline, rng):
    """DFS with Warnsdorff heuristic to find Hamilton cycle."""
    sys.setrecursionlimit(10000)
    
    def dfs(v, visited_mask, path):
        if len(path) == N:
            if masks[v] & (1 << 0):
                return path[:]
            return None
        
        available = masks[v] & ~visited_mask
        if available == 0:
            return None
        
        # Get neighbors sorted by onward degree (Warnsdorff)
        neighbors = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            nb = lsb.bit_length() - 1
            temp ^= lsb
            if len(path) == N - 1:
                onward = 1 if (masks[nb] & (1 << 0)) else 0
            else:
                onward = bin(masks[nb] & ~visited_mask & ~(1 << v)).count('1')
            neighbors.append((onward, nb))
        
        # Sort by onward degree (ascending for Warnsdorff)
        neighbors.sort(key=lambda x: x[0])
        
        # Group by degree to randomize within groups
        from collections import defaultdict
        deg_map = defaultdict(list)
        for deg, nb in neighbors:
            deg_map[deg].append(nb)
        
        # Process degrees in increasing order
        for deg in sorted(deg_map.keys()):
            nbs = deg_map[deg]
            rng.shuffle(nbs)
            for nb in nbs:
                result = dfs(nb, visited_mask | (1 << nb), path + [nb])
                if result is not None:
                    return result
        return None
    
    # Try multiple restarts with different seeds
    for attempt in range(10):
        if time.time() > deadline - 2:
            break
        rng = random.Random(attempt * 7919 + 12345)
        cycle = dfs(0, 1 << 0, [0])
        if cycle is not None:
            return cycle
    
    return None

def two_opt_polish(cycle, edge_pos, deadline):
    """Apply 2-opt to minimize max edge position."""
    if cycle is None:
        return None
    
    def cycle_score(cyc):
        max_pos = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    # Try 2-opt moves
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        n = len(best_cycle)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse segment from i+1 to j
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best_cycle = None
    best_K = len(EDGE_ORDER)
    
    # Binary search for minimum K
    lo, hi = N, len(EDGE_ORDER)
    
    while lo <= hi and time.time() < deadline - 15:
        mid = (lo + hi) // 2
        masks = MASKS[mid - N]
        
        # Quick checks
        valid = True
        for v in range(N):
            if bin(masks[v]).count('1') < 2:
                valid = False
                break
        if not valid:
            lo = mid + 1
            continue
        
        if not check_connectivity(masks):
            lo = mid + 1
            continue
        
        # Try MILP 2-factor + Merge
        cycle = None
        if time.time() < deadline - 10:
            edges = find_2_factor_milp(mid, deadline)
            if edges:
                cycles = get_cycles_from_edges(edges)
                if len(cycles) == 1:
                    cycle = cycles[0]
                else:
                    cycle = merge_cycles(cycles, mid, deadline)
        
        # If MILP failed, try DFS
        if cycle is None and time.time() < deadline - 5:
            cycle = dfs_find_cycle(masks, deadline, random.Random(mid))
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to polish it with 2-opt
    if best_cycle is not None and time.time() < deadline - 5:
        polished = two_opt_polish(best_cycle, edge_pos, deadline)
        if polished is not None:
            best_cycle = polished
    
    # Fallback: try higher K values
    if best_cycle is None:
        for K in range(N, len(EDGE_ORDER) + 1, 5):
            if time.time() > deadline - 5:
                break
            masks = MASKS[K - N]
            
            valid = True
            for v in range(N):
                if bin(masks[v]).count('1') < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            if not check_connectivity(masks):
                continue
            
            cycle = dfs_find_cycle(masks, deadline, random.Random(K))
            if cycle is not None:
                best_cycle = cycle
                break
    
    if best_cycle is not None:
        return best_cycle
    
    return list(range(N))
