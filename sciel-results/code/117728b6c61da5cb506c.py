import time
import sys
from collections import deque

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position lookup
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v) if u < v else (v, u)] = i

# Bit masks
FULL_MASK = (1 << N) - 1
START_BIT = 1 << 0

def _has_cycle(adj, mask):
    """Does the subgraph induced by 'mask' contain a cycle? (bitwise, adj is list of int masks)"""
    # A graph has a cycle iff it has an edge whose removal... simpler: check if any vertex
    # has two neighbors in mask that are connected via another path. Use: if the number of
    # edges > number of vertices - number of components, there is a cycle.
    # Simpler DFS-based: if during DFS we find a back edge to an ancestor.
    # For small graphs, do a simple check: does there exist an edge (u,v) in mask such that
    # u and v are connected in mask \ {edge}? That's expensive.
    # Instead: a cycle exists iff the subgraph is not a forest, i.e., edges >= vertices in some component.
    # Count edges and vertices per component.
    visited = 0
    has_cycle = False
    for v in range(N):
        if not (mask & (1 << v)):
            continue
        if visited & (1 << v):
            continue
        # BFS from v
        comp_mask = 0
        edge_count = 0
        stack = [v]
        comp_mask |= (1 << v)
        while stack:
            x = stack.pop()
            # count edges from x to comp
            nb = adj[x] & comp_mask
            edge_count += nb.bit_count() // 1  # each edge counted once from each endpoint... 
            # Actually let's count differently: total degree in component
            nb_unvisited = adj[x] & mask & ~comp_mask
            while nb_unvisited:
                lsb = nb_unvisited & (-nb_unvisited)
                y = lsb.bit_length() - 1
                comp_mask |= lsb
                stack.append(y)
                nb_unvisited ^= lsb
        # edge_count: sum of degrees in component / 2
        deg_sum = 0
        m = comp_mask
        while m:
            lsb = m & (-m)
            x = lsb.bit_length() - 1
            deg_sum += (adj[x] & comp_mask).bit_count()
            m ^= lsb
        edges = deg_sum // 2
        verts = comp_mask.bit_count()
        if edges >= verts:
            has_cycle = True
            break
        visited |= comp_mask
    return has_cycle

def _is_connected(adj, mask):
    """Is the subgraph induced by 'mask' connected?"""
    if mask == 0:
        return True
    first_lsb = mask & (-mask)
    start = first_lsb.bit_length() - 1
    visited = 0
    stack = [start]
    visited |= (1 << start)
    while stack:
        x = stack.pop()
        nb = adj[x] & mask & ~visited
        while nb:
            lsb = nb & (-nb)
            y = lsb.bit_length() - 1
            visited |= lsb
            stack.append(y)
            nb ^= lsb
    return visited == mask

def _has_articulation(adj, mask):
    """Does the subgraph induced by 'mask' have an articulation point? (returns True if it does)"""
    if mask.bit_count() < 3:
        return False
    # Build adjacency restricted to mask
    # For each vertex v in mask, check if removing v disconnects mask
    # Use a quick method: for each v, check if mask without v is connected AND v is connected to all
    # Actually simpler: a vertex v is articulation if the subgraph (mask without v) has more than one
    # component that is adjacent to v.
    for v in range(N):
        if not (mask & (1 << v)):
            continue
        remaining = mask & ~(1 << v)
        if remaining == 0:
            continue
        # Check if 'remaining' is connected
        if _is_connected(adj, remaining):
            # If remaining is connected, v is articulation iff v is not adjacent to all of remaining
            # (i.e., removing v doesn't disconnect because remaining stays connected)
            # Actually if remaining is connected, v is NOT an articulation point (removing v leaves it connected)
            # unless v was the only connection... no, if remaining is connected without v, then v is not articulation
            continue
        else:
            # remaining is disconnected, so v IS an articulation point
            return True
    return False

def _count_unvisited_neighbors(adj, v, visited_mask):
    """Count unvisited neighbors of v."""
    return (adj[v] & ~visited_mask).bit_count()

def find_hamilton_cycle_fast(adj, deadline):
    """
    Fast backtracking with bitmasks to find a Hamilton cycle starting at vertex 0.
    adj: list of int bitmasks (adj[v] has bit u set if edge (u,v) exists)
    Returns list of vertices [0, v1, v2, ..., v23] or None.
    """
    sys.setrecursionlimit(10000)
    
    # Precompute: for each vertex, neighbors sorted by degree (for Warnsdorff)
    # adj is already a bitmask list
    
    path = [0]
    visited_mask = START_BIT
    current = 0
    
    # Check basic feasibility
    for v in range(N):
        if (adj[v]).bit_count() < 2:
            return None
    
    if not _is_connected(adj, FULL_MASK):
        return None
    
    # Precompute neighbor lists for ordering
    neighbor_lists = [sorted([u for u in range(N) if (adj[v] >> u) & 1]) for v in range(N)]
    
    def dfs(v, visited_mask, path, depth):
        if time.time() > deadline:
            return None
        
        if depth == N - 1:
            # All vertices visited. Check if v connects back to 0
            if (adj[v] >> 0) & 1:
                return path[:]
            return None
        
        # Get unvisited neighbors of v
        unvisited = adj[v] & ~visited_mask
        if unvisited == 0:
            return None
        
        # Build candidate list with Warnsdorff ordering
        candidates = []
        m = unvisited
        while m:
            lsb = m & (-m)
            nb = lsb.bit_length() - 1
            # Count unvisited neighbors of nb (excluding v which is already visited)
            new_visited = visited_mask | lsb
            unvisited_nb = adj[nb] & ~new_visited
            count = unvisited_nb.bit_count()
            candidates.append((count, nb))
            m ^= lsb
        
        candidates.sort(key=lambda x: x[0])
        
        for count, nb in candidates:
            new_visited = visited_mask | (1 << nb)
            remaining = N - depth - 1  # vertices left after adding nb
            
            if remaining > 0:
                # nb must have at least one unvisited neighbor to continue
                # OR if this is the last step (remaining==0), nb must connect to 0
                if count == 0:
                    # nb has no unvisited neighbors
                    if remaining == 0:
                        # Last vertex: must connect to 0
                        if not ((adj[nb] >> 0) & 1):
                            continue
                    else:
                        continue
                
                # Articulation point check on remaining subgraph
                # The remaining unvisited vertices must be connected (otherwise we can't complete)
                rem_mask = ~new_visited & FULL_MASK
                if rem_mask and not _is_connected(adj, rem_mask):
                    continue
                
                # Check that nb is not an articulation point of the remaining graph
                # that would isolate some vertices
                # (This is a stronger check but expensive; do it only for small remaining sets)
                if rem_mask.bit_count() <= 8:
                    # Check if removing nb from (new_visited's complement + nb) disconnects
                    # Actually, check if in the graph induced by (rem_mask | (1<<nb)),
                    # nb is an articulation point
                    sub_mask = rem_mask | (1 << nb)
                    if sub_mask.bit_count() >= 3:
                        if _has_articulation(adj, sub_mask):
                            # nb is articulation in the remaining subgraph - this might be ok
                            # if the path goes through it, but it means we can't visit all
                            # remaining vertices in one go. Skip if it would isolate vertices.
                            # Actually this is a valid concern: if nb is articulation, then
                            # after we leave nb, we can't come back, so one side would be
                            # isolated. But we're building a path, not a cycle yet.
                            # The real constraint: after visiting nb, the remaining vertices
                            # must be reachable from the next vertex we pick.
                            # Let's just do the connectivity check which we already did above.
                            pass
            
            result = dfs(nb, new_visited, path + [nb], depth + 1)
            if result is not None:
                return result
        
        return None
    
    return dfs(0, START_BIT, [0], 1)


def build_adjacency_bitmask(K):
    """Build adjacency as list of int bitmasks using edges with position <= K."""
    adj = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def cycle_score(cycle):
    """Compute the max edge position for a cycle."""
    max_pos = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        key = (u, v) if u < v else (v, u)
        pos = edge_pos[key]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def local_search_polish(cycle, deadline):
    """
    Try to improve a Hamilton cycle by swapping edges to reduce the max edge position.
    Uses 2-opt style: try reversing segments or swapping individual edges.
    """
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    # Build edge position array for quick lookup
    # Try swapping two non-adjacent edges in the cycle
    # A swap: if cycle is ...a-b...c-d..., we can try ...a-c...b-d... or ...a-d...c-b...
    # This is 2-opt for TSP.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best_cycle)
        for i in range(n):
            if time.time() > deadline:
                break
            a = best_cycle[i]
            b = best_cycle[(i + 1) % n]
            for j in range(i + 2, n):
                if time.time() > deadline:
                    break
                if j == n - 1 and i == 0:
                    continue  # adjacent edges
                c = best_cycle[j]
                d = best_cycle[(j + 1) % n]
                
                # Option 1: swap to a-c and b-d
                # Check if edges (a,c) and (b,d) exist
                key1 = (a, c) if a < c else (c, a)
                key2 = (b, d) if b < d else (d, b)
                if key1 in edge_pos and key2 in edge_pos:
                    pos1 = edge_pos[key1]
                    pos2 = edge_pos[key2]
                    # New max would be max of all other edges + pos1 + pos2
                    # Quick check: if both new edges are <= current max, we might improve
                    # or at least not worsen. Try the swap.
                    new_cycle = best_cycle[:i+1] + best_cycle[i+1:j][::-1] + best_cycle[j:]
                    # Wait, 2-opt reverses the segment between i+1 and j
                    # Original: a, (i+1), ..., (j-1), b, c, (j+1), ..., d, a
                    # After 2-opt: a, c, (j+1), ..., d, b, (j-1), ..., (i+1), a
                    # Hmm, let me think more carefully.
                    # Cycle: v0, v1, ..., v_{n-1}, v0
                    # Edges: (v0,v1), (v1,v2), ..., (v_{n-1},v0)
                    # 2-opt: pick two edges (vi, vi+1) and (vj, vj+1), remove them,
                    # add (vi, vj) and (vi+1, vj+1) or (vi, vj+1) and (vi+1, vj)
                    # Let's just try all pair swaps properly.
                    
                    # Let me redo: edges are (cycle[k], cycle[(k+1)%n]) for k=0..n-1
                    # Pick edge i: (cycle[i], cycle[i+1]) and edge j: (cycle[j], cycle[j+1])
                    # Option A: new edges (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                    # Option B: new edges (cycle[i], cycle[j+1]) and (cycle[i+1], cycle[j])
                    
                    # For option A:
                    ka = (a, c) if a < c else (c, a)
                    kb = (b, d) if b < d else (d, b)
                    if ka in edge_pos and kb in edge_pos:
                        # Build new cycle
                        # Remove edges i and j, reconnect
                        # New cycle: 0..i, j, j+1..i+1 (reversed), i+2..j-1, j+1..n-1, 0
                        # This is getting complex. Let me just try a simpler approach:
                        # Try swapping the positions of two vertices in the cycle.
                        pass
                
                # Option 2: swap to a-d and c-b
                key3 = (a, d) if a < d else (d, a)
                key4 = (c, b) if c < b else (b, c)
                if key3 in edge_pos and key4 in edge_pos:
                    pass
        
        # Simpler local search: try moving one vertex to a different position
        for pos in range(1, n - 1):
            if time.time() > deadline:
                break
            v = best_cycle[pos]
            for new_pos in range(1, n - 1):
                if new_pos == pos or new_pos == pos + 1 or new_pos == pos - 1:
                    continue
                # Remove v from pos, insert at new_pos
                new_cycle = best_cycle[:pos] + best_cycle[pos+1:]
                insert_at = new_pos if new_pos < pos else new_pos - 1
                new_cycle = new_cycle[:insert_at] + [v] + new_cycle[insert_at:]
                # Check validity: edges (new_cycle[pos-1], v) and (v, new_cycle[pos]) must exist
                # Also the edge that was at pos is now split
                prev_v = new_cycle[(pos - 1) % n]
                next_v = new_cycle[pos % n]
                prev_ins = new_cycle[(insert_at - 1) % n]
                next_ins = new_cycle[insert_at % n]
                
                key1 = (prev_v, v) if prev_v < v else (v, prev_v)
                key2 = (v, next_v) if v < next_v else (next_v, v)
                key3 = (prev_ins, next_ins) if prev_ins < next_ins else (next_ins, prev_ins)
                
                if key1 in edge_pos and key2 in edge_pos and key3 in edge_pos:
                    sc = cycle_score(new_cycle)
                    if sc < best_score:
                        best_cycle = new_cycle
                        best_score = sc
                        improved = True
                        break
            if improved:
                break
    
    return best_cycle, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s kill)
    
    best_cycle = None
    best_K = None
    
    # Strategy: binary search for minimum K where a Hamilton cycle exists
    # Use the fast bitset solver
    lo, hi = 24, len(EDGE_ORDER)
    
    while lo <= hi and time.time() < deadline - 15:
        mid = (lo + hi) // 2
        
        adj = build_adjacency_bitmask(mid)
        
        # Quick checks
        for v in range(N):
            if adj[v].bit_count() < 2:
                lo = mid + 1
                break
        else:
            if not _is_connected(adj, FULL_MASK):
                lo = mid + 1
                continue
            
            # Try to find Hamilton cycle
            found = find_hamilton_cycle_fast(adj, time.time() + 5.0)
            if found is not None:
                best_cycle = found
                best_K = mid
                hi = mid - 1
            else:
                lo = mid + 1
    
    if best_cycle is None:
        # Fallback: try larger K values
        for K in range(100, len(EDGE_ORDER) + 1, 5):
            if time.time() > deadline - 10:
                break
            adj = build_adjacency_bitmask(K)
            ok = True
            for v in range(N):
                if adj[v].bit_count() < 2:
                    ok = False
                    break
            if not ok:
                continue
            if not _is_connected(adj, FULL_MASK):
                continue
            found = find_hamilton_cycle_fast(adj, time.time() + 5.0)
            if found is not None:
                best_cycle = found
                best_K = K
                break
    
    if best_cycle is None:
        # Last resort: identity permutation (will have a high score but is valid)
        best_cycle = list(range(N))
    
    # Local search polish
    if time.time() < deadline - 5:
        best_cycle, best_score = local_search_polish(best_cycle, deadline - 2)
    
    return best_cycle


# Test
if __name__ == "__main__":
    t0 = time.time()
    cycle = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Cycle: {cycle}")
    if len(cycle) == N and len(set(cycle)) == N:
        max_pos = 0
        for i in range(N):
            u = cycle[i]
            v = cycle[(i+1) % N]
            key = (u, v) if u < v else (v, u)
            pos = edge_pos[key]
            max_pos = max(max_pos, pos)
            print(f"  {u}-{v}: pos {pos}")
        print(f"Score (max edge pos): {max_pos}")
    else:
        print("INVALID CYCLE")
