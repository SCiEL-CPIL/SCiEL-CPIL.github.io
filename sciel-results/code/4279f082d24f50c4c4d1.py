import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    N = 128

    # 1. Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) and lab == lcm(b, c):
                    bad_triples.append((a, b, c))

    # 2. Build conflict graph (2-shadow of bad triples)
    # Two nodes are in conflict if they appear together in at least one bad triple
    conflict = {i: set() for i in range(1, N + 1)}
    for a, b, c in bad_triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)

    # 3. Greedy construction of initial solution
    # Start with all nodes, remove those in bad triples
    S = set(range(1, N + 1))

    # Pre-compute for each node, the list of triples it belongs to
    node_triples = {i: [] for i in range(1, N + 1)}
    for t in bad_triples:
        for x in t:
            node_triples[x].append(t)

    def is_active(triple, S):
        return all(x in S for x in triple)

    # Greedy removal phase
    while True:
        if time.time() > deadline - 30:
            break
        # Find an active bad triple
        active_triple = None
        for t in bad_triples:
            if is_active(t, S):
                active_triple = t
                break

        if active_triple is None:
            break

        a, b, c = active_triple

        # Calculate active degrees
        def get_active_degree(x, S):
            count = 0
            for t in node_triples[x]:
                if is_active(t, S):
                    count += 1
            return count

        da = get_active_degree(a, S)
        db = get_active_degree(b, S)
        dc = get_active_degree(c, S)

        # Remove the one with highest active degree
        candidates = [(da, a), (db, b), (dc, c)]
        candidates.sort(key=lambda x: (-x[0], -x[1]))

        victim = candidates[0][1]
        S.remove(victim)

    # 4. Local search improvement: try to add back removed nodes
    removed = set(range(1, N + 1)) - S

    def can_add(node, S):
        # Check if adding node creates any bad triple
        for t in node_triples[node]:
            other1, other2 = [x for x in t if x != node]
            if other1 in S and other2 in S:
                return False
        return True

    # Try to add back nodes in order of how "safe" they are
    for node in sorted(removed, key=lambda x: -len([t for t in node_triples[x] if all(y in S for y in t if y != node)])):
        if time.time() > deadline - 20:
            break
        if can_add(node, S):
            S.add(node)

    # 5. Branch and bound on conflict graph
    # Upper bound: n - size of maximum matching in induced subgraph
    def max_matching_size(nodes):
        """Find maximum matching in the induced subgraph on nodes."""
        # Use a simple greedy matching for speed
        nodes_list = list(nodes)
        matched = set()
        matching_size = 0
        for u in nodes_list:
            if u in matched:
                continue
            # Find an unmatched neighbor
            for v in conflict[u]:
                if v in nodes and v not in matched and v > u:
                    matched.add(u)
                    matched.add(v)
                    matching_size += 1
                    break
        return matching_size

    def upper_bound(nodes):
        n = len(nodes)
        if n == 0:
            return 0
        mm = max_matching_size(nodes)
        return n - mm

    def branch_and_bound(nodes, current_set, best_size_ref, time_limit):
        """Branch and bound for maximum independent set."""
        if time.time() > time_limit:
            return

        if not nodes:
            if len(current_set) > best_size_ref[0]:
                best_size_ref[0] = len(current_set)
            return

        ub = upper_bound(nodes)
        if len(current_set) + ub <= best_size_ref[0]:
            return

        # Pick a node to branch on: highest degree in induced subgraph
        node_degrees = {}
        for u in nodes:
            d = 0
            for v in conflict[u]:
                if v in nodes:
                    d += 1
            node_degrees[u] = d

        pivot = max(nodes, key=lambda x: node_degrees[x])

        # Branch 1: include pivot
        new_current = set(current_set)
        new_current.add(pivot)
        new_nodes = set(nodes) - {pivot} - (conflict[pivot] & nodes)
        branch_and_bound(new_nodes, new_current, best_size_ref, time_limit)

        # Branch 2: exclude pivot
        new_nodes2 = set(nodes) - {pivot}
        branch_and_bound(new_nodes2, current_set, best_size_ref, time_limit)

    best_size_ref = [len(S)]
    all_nodes = set(range(1, N + 1))

    # Run B&B with time limit
    if time.time() < deadline - 5:
        branch_and_bound(all_nodes, set(), best_size_ref, deadline - 2)

    # Reconstruct best solution if B&B found something better
    # For simplicity, return current S (which is already good from greedy + local search)
    return list(S)
