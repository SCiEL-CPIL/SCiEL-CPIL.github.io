"""Packing edge-disjoint Hamiltonian cycles on the 21-vertex 5-regular host."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
EDGES = P['edges']

# Build adjacency and edge-key helpers
_adj = [[] for _ in range(N)]
for a, b in EDGES:
    _adj[a].append(b)
    _adj[b].append(a)

# Sort adjacency for deterministic-ish but we'll shuffle per restart
for v in range(N):
    _adj[v].sort()

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def find_hamiltonian_cycle(used_edges, start=0, time_limit=None, start_time=None):
    """Find a Hamiltonian cycle using only edges not in used_edges.
    Returns list of vertices (cycle) or None."""
    if time_limit is None:
        start_time = time.time()
    remaining = [set() for _ in range(N)]
    for v in range(N):
        for w in _adj[v]:
            if _edge_key(v, w) not in used_edges:
                remaining[v].add(w)
    # Check start has at least 2 remaining edges (needs in and out)
    if len(remaining[start]) < 2:
        return None

    # Degree-based ordering for branching: prefer vertices with fewer options
    # We'll do DFS with a heuristic: at each step, pick next vertex.
    # To make it faster, use a "forced move" detection: if a vertex in the
    # remaining path has only 1 available neighbor not yet used in path, it's forced.

    path = [start]
    path_set = {start}
    # For the cycle, we need to return to start at the end.
    # During construction, start has one edge used (to first neighbor) at the end.
    # Let's track: the cycle is start -> v1 -> v2 -> ... -> v_{n-1} -> start
    # So start needs 2 edges: to v1 and from v_{n-1}.

    # We'll use recursion with backtracking.
    # Optimization: if any unvisited vertex has degree 0 in remaining, fail.
    # If any unvisited vertex has degree 1 in remaining, it's forced (must use that edge).

    def backtrack():
        if time.time() - start_time > time_limit:
            return False
        if len(path) == N:
            # Check if last vertex connects to start
            last = path[-1]
            if start in remaining[last]:
                return True
            return False

        current = path[-1]
        # Get candidates: neighbors of current not in path, with available edge
        candidates = []
        for w in remaining[current]:
            if w not in path_set:
                candidates.append(w)

        # Forced move check: for each unvisited vertex, if it has exactly 1
        # available neighbor not in path (or in path but not current, meaning
        # it must connect to that specific vertex), it constrains the choice.
        # Simplified: if an unvisited vertex has only 1 available neighbor total,
        # and that neighbor is in path (and not current), then that vertex must
        # connect to that neighbor, but since it's unvisited, the edge from that
        # neighbor to this vertex must be used when we reach the neighbor.
        # More directly: if an unvisited vertex u has exactly 1 available neighbor v,
        # then if v is in path and v != current, u must be the NEXT vertex after v,
        # but v is already in path, so u must connect to v via an edge that will be
        # used. This means when we extend from v, we must go to u. But v is already
        # placed. So if v is in path and u is unvisited, and u has only neighbor v
        # available, then u must be placed adjacent to v. Since v is in path, u must
        # be either the next after v (if v is current) or it's a constraint that
        # the edge (v,u) must be used, but v's other connection is already fixed.
        # This is complex. Let's use a simpler heuristic: sort candidates by the
        # number of remaining edges of the candidate (fewer first - most constrained).
        candidates.sort(key=lambda w: len(remaining[w] - path_set))

        # Additional pruning: if any unvisited vertex has 0 available neighbors
        # (excluding path vertices it's not adjacent to... actually just check
        # if it has any available neighbor at all)
        for u in range(N):
            if u not in path_set:
                avail = [w for w in remaining[u] if w not in path_set or w == start]
                # u can connect to unvisited vertices or to start (for the last edge)
                # Actually u needs to connect to path vertices or unvisited.
                # For now, just check if u has any available neighbor
                if len(remaining[u]) == 0:
                    return False

        for w in candidates:
            # Try adding w
            path.append(w)
            path_set.add(w)
            # Mark edge (current, w) as used in remaining
            edge = _edge_key(current, w)
            remaining[current].discard(w)
            remaining[w].discard(current)

            if backtrack():
                return True

            # Undo
            remaining[current].add(w)
            remaining[w].add(current)
            path.pop()
            path_set.discard(w)

        return False

    if backtrack():
        return list(path)
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Try to find 2 edge-disjoint Hamiltonian cycles
    best_result = [list(range(N))]  # Fallback: the seed's single cycle

    # First, find one Hamiltonian cycle
    cycle1 = find_hamiltonian_cycle(set(), time_limit=deadline - time.time())
    if cycle1 is None:
        # Try with different start
        for s in range(N):
            if time.time() > deadline:
                break
            cycle1 = find_hamiltonian_cycle(set(), start=s, time_limit=deadline - time.time())
            if cycle1 is not None:
                break
    if cycle1 is None:
        return best_result

    used_edges = set()
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        used_edges.add(_edge_key(u, v))

    # Try to find a second edge-disjoint Hamiltonian cycle
    cycle2 = find_hamiltonian_cycle(used_edges, time_limit=deadline - time.time())
    if cycle2 is not None:
        # Verify cycle2 is valid and edge-disjoint
        valid = True
        for i in range(len(cycle2)):
            u = cycle2[i]
            v = cycle2[(i + 1) % len(cycle2)]
            if _edge_key(u, v) in used_edges:
                valid = False
                break
        if valid:
            # Convert to proper int lists
            c1 = [int(v) for v in cycle1]
            c2 = [int(v) for v in cycle2]
            return [c1, c2]

    # If we can't find 2, return just 1
    c1 = [int(v) for v in cycle1]
    return [c1]

# EVOLVE-BLOCK-END
