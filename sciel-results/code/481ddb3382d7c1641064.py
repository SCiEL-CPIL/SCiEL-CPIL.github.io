"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE_MARGIN = 15  # seconds to spare before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + 90 - DEADLINE_MARGIN

    n = P['n']
    edges_list = P['edges']
    num_edges = len(edges_list)

    # Build edge index mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        key = (min(u, v), max(u, v))
        edge_index[key] = i

    # Adjacency with edge indices
    adj = [[] for _ in range(n)]  # list of (neighbor, edge_idx)
    adj_set = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(edges_list):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        adj_set[u].add(v)
        adj_set[v].add(u)

    # Sort adjacency for deterministic ordering
    for i in range(n):
        adj[i].sort(key=lambda x: x[0])

    # Theoretical max: min(degree//2, edges//n)
    min_degree = min(len(adj[v]) for v in range(n))
    max_possible = min(10, min_degree // 2, num_edges // n)

    def time_left():
        return deadline - time.time()

    def find_hamiltonian_cycle(used_edges, start_vertex):
        """
        Find a Hamiltonian cycle starting at start_vertex, avoiding used edges.
        Uses backtracking with connectivity pruning.
        Returns list of vertices or None.
        """
        if time_left() < 2:
            return None

        # Build available edges (not used)
        # For each vertex, track which neighbors are reachable via unused edges
        # State: path (list of vertices), used_edge_set (local tracking)
        # We fix start_vertex as first vertex

        path = [start_vertex]
        visited = [False] * n
        visited[start_vertex] = True
        local_used = set()  # edge indices used in this cycle

        # Pre-compute: for each vertex, which edges connect to it
        # Use a recursive DFS with pruning

        def dfs(current, depth):
            if time_left() < 1:
                return None

            if depth == n:
                # Check if we can close the cycle: current -> start_vertex
                key = (min(current, start_vertex), max(current, start_vertex))
                if key in edge_index:
                    eidx = edge_index[key]
                    if eidx not in used_edges and eidx not in local_used:
                        return list(path)
                return None

            last = path[-1]

            # Get available neighbors (not visited, edge not used)
            candidates = []
            for neighbor, eidx in adj[last]:
                if visited[neighbor]:
                    continue
                if eidx in used_edges or eidx in local_used:
                    continue
                candidates.append((neighbor, eidx))

            # Pruning: check connectivity of unvisited vertices
            # The unvisited vertices (plus current as a "port") must be connected
            # via unused edges to each other and to the start (for closure)
            # Simple check: all unvisited vertices must be reachable from 'current'
            # through unused, unvisited-vertex edges
            if candidates:
                unvisited = [v for v in range(n) if not visited[v]]
                # Check that all unvisited vertices are in the same connected component
                # when considering only unused edges between unvisited vertices
                # (and edges from current to unvisited)
                if len(unvisited) > 1:
                    # BFS from current through unused edges to unvisited vertices
                    reachable = set()
                    stack = [current]
                    while stack:
                        node = stack.pop()
                        for nb, eidx in adj[node]:
                            if eidx in used_edges or eidx in local_used:
                                continue
                            if not visited[nb] and nb not in reachable:
                                reachable.add(nb)
                                stack.append(nb)
                    if len(reachable) != len(unvisited):
                        return None

            # Try candidates in order
            for neighbor, eidx in candidates:
                path.append(neighbor)
                visited[neighbor] = True
                local_used.add(eidx)
                result = dfs(neighbor, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited[neighbor] = False
                local_used.discard(eidx)

            return None

        return dfs(start_vertex, 1)

    def find_all_cycles(k_target):
        """Find up to k_target edge-disjoint Hamiltonian cycles."""
        used_edges = set()
        cycles = []

        for c in range(k_target):
            if time_left() < 5:
                break

            found = None
            # Try different start vertices to increase chances
            for sv in range(n):
                if time_left() < 3:
                    break
                found = find_hamiltonian_cycle(used_edges, sv)
                if found is not None:
                    break

            if found is None:
                break

            # Mark edges of this cycle as used
            for i in range(len(found)):
                u = found[i]
                v = found[(i + 1) % len(found)]
                key = (min(u, v), max(u, v))
                eidx = edge_index[key]
                used_edges.add(eidx)

            cycles.append(found)

        return cycles

    # Try to find the maximum number of edge-disjoint Hamiltonian cycles
    best_cycles = None
    best_count = 0

    for k in range(max_possible, 0, -1):
        if time_left() < 10:
            break

        print(f"Trying to find {k} edge-disjoint Hamiltonian cycles...", file=sys.stderr)
        cycles = find_all_cycles(k)

        if cycles is not None and len(cycles) > best_count:
            best_cycles = cycles
            best_count = len(cycles)
            print(f"Found {best_count} cycles!", file=sys.stderr)

            if best_count >= max_possible:
                break

    # Fallback: if no Hamiltonian cycles found, try to find any valid cycle set
    if best_cycles is None or len(best_cycles) == 0:
        # Try to find a single triangle
        for u in range(n):
            for v in adj_set[u]:
                if v > u:
                    common = adj_set[u] & adj_set[v]
                    for w in common:
                        if w != u and w != v:
                            best_cycles = [[u, v, w]]
                            best_count = 1
                            break
                if best_cycles:
                    break
            if best_cycles:
                break

    if best_cycles is None or len(best_cycles) == 0:
        # Absolute fallback
        return [[0, 1, 2]]

    # Validate and convert to proper types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])

    return result

# EVOLVE-BLOCK-END
