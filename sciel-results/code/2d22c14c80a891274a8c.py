"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin

    N = 256

    # Perfect squares up to 2*N
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Forbidden: numbers where 2*a is a perfect square (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]

    # Partition by parity
    odds = [x for x in valid_numbers if x % 2 == 1]
    evens = [x for x in valid_numbers if x % 2 == 0]

    # Build bipartite conflict graph: edge between odd o and even e iff o+e is a perfect square
    odd_set = set(odds)
    even_set = set(evens)

    # Map indices
    odd_idx = {x: i for i, x in enumerate(odds)}
    even_idx = {x: i for i, x in enumerate(evens)}

    n_odds = len(odds)
    n_evens = len(evens)

    # Build adjacency: for each odd, which evens conflict
    odd_to_even = [set() for _ in range(n_odds)]
    for i, o in enumerate(odds):
        for e in evens:
            if o + e in squares:
                odd_to_even[i].add(even_idx[e])

    # Build flow network
    # Nodes:
    #   0 = source
    #   1 .. n_odds = odd nodes
    #   n_odds+1 .. n_odds+n_evens = even nodes
    #   n_odds+n_evens+1 = sink
    total_nodes = 2 + n_odds + n_evens
    source = 0
    sink = total_nodes - 1

    # Adjacency list for flow network (directed)
    # Each edge: (to, capacity)
    # We'll use a dict of dicts for residual capacities
    # residual[u][v] = capacity of edge u->v
    residual = {}
    reverse = {}  # reverse[v][u] = u for quick reverse lookup

    def add_edge(u, v, cap):
        if v not in residual[u]:
            residual[u][v] = 0
            reverse[v][u] = u
        residual[u][v] += cap
        if u not in residual[v]:
            residual[v][u] = 0
            reverse[u][v] = v
        # residual[v][u] stays as is (will be updated during flow)

    # Source -> each odd node (capacity 1)
    for i in range(n_odds):
        add_edge(source, 1 + i, 1)

    # Each even node -> sink (capacity 1)
    for j in range(n_evens):
        add_edge(1 + n_odds + j, sink, 1)

    # Odd -> Even edges (capacity 1) for conflicts
    for i in range(n_odds):
        for j in odd_to_even[i]:
            add_edge(1 + i, 1 + n_odds + j, 1)

    # Ensure reverse edges exist in residual
    for u in residual:
        for v in residual[u]:
            if v not in residual:
                residual[v] = {}
            if u not in residual[v]:
                residual[v][u] = 0

    # Also ensure all nodes have entries
    for node in range(total_nodes):
        if node not in residual:
            residual[node] = {}

    # Edmonds-Karp max flow
    from collections import deque

    def bfs_max_flow():
        """Edmonds-Karp: find augmenting paths via BFS."""
        max_flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * total_nodes
            parent[source] = source
            queue = deque([source])
            found = False

            while queue and not found:
                u = queue.popleft()
                for v in residual[u]:
                    if residual[u][v] > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == sink:
                            found = True
                            break
                        queue.append(v)

            if not found:
                break

            # Find bottleneck
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            # Update residual
            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = u

            max_flow += path_flow

        return max_flow

    max_flow_value = bfs_max_flow()

    # Find min vertex cover via max-flow residual
    # In the residual graph after max flow:
    #   S = set of nodes reachable from source
    #   Min vertex cover = (odds not in S) union (evens in S)
    #   Max independent set = complement = (odds in S) union (evens not in S)

    # BFS from source in residual graph
    reachable = [False] * total_nodes
    reachable[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in residual[u]:
            if residual[u][v] > 0 and not reachable[v]:
                reachable[v] = True
                queue.append(v)

    # Max independent set:
    # Odd nodes in S (reachable from source)
    # Even nodes NOT in S
    result = []
    for i in range(n_odds):
        if reachable[1 + i]:
            result.append(odds[i])
    for j in range(n_evens):
        if not reachable[1 + n_odds + j]:
            result.append(evens[j])

    # Verify the result is valid
    result_set = set(result)
    valid = True
    for a in result:
        for b in result:
            if a <= b and (a + b) in squares:
                valid = False
                break
        if not valid:
            break

    if not valid or not result:
        # Fallback: greedy on valid numbers
        # Use min-degree heuristic
        conflict = [set() for _ in range(N + 1)]
        for a in range(1, N + 1):
            for sq in squares:
                b = sq - a
                if 1 <= b <= N:
                    conflict[a].add(b)

        # Greedy: sort by degree ascending, add if no conflict with already selected
        selected = set()
        candidates = sorted(valid_numbers, key=lambda x: len(conflict[x] - forbidden))
        for x in candidates:
            if x in forbidden:
                continue
            if not (conflict[x] & selected):
                selected.add(x)
        result = sorted(selected)

    return sorted(result)
# EVOLVE-BLOCK-END
