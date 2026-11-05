"""Combined program: Todd-Coxeter subgroup enumeration for cage problem."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_DEADLINE = 120.0
_MARGIN = 6.0


def _validate_graph(n, edges, k, g, max_vertices=None):
    """Validate simple k-regular graph with girth >= g."""
    if max_vertices is not None and n > max_vertices:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Girth check via non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (g - 5) % 2 == 0:
        d = g - 4
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _todd_coxeter_enumerate(max_index, rng, max_steps=200000):
    """
    Enumerate subgroups of C5 * C5 up to max_index using Todd-Coxeter.
    Returns list of (index, table) where table is a list of lists.
    """
    # Generators: a (order 5), b (order 5)
    # a^5 = 1, b^5 = 1
    # We'll use random relators to guide enumeration
    
    results = []
    
    for target_idx in range(2, max_index + 1):
        # Try multiple random seeds for this index
        for attempt in range(3):
            table, success = _tc_single(target_idx, rng, max_steps)
            if success and table is not None:
                results.append((target_idx, table))
                break  # Found one for this index, move to next
    
    return results


def _tc_single(target_index, rng, max_steps):
    """
    Attempt Todd-Coxeter enumeration to find a subgroup of the given index.
    Returns (table, success).
    """
    # table[i][j] = result of applying generator j to coset i
    # -1 means undefined
    # Generators: 0=a, 1=a^-1, 2=b, 3=b^-1
    # Relations: a^5=1, b^5=1
    
    n = target_index
    table = [[-1, -1, -1, -1] for _ in range(n)]
    
    # Initialize: coset 0 is the identity coset
    # a^5 = 1 means applying a five times returns to same coset
    # We'll set up the relation a^5 = 1 for coset 0
    # table[0][0] should eventually form a 5-cycle
    
    # Start with coset 0
    # Try to build the a-cycle for coset 0
    if rng.random() < 0.5:
        # a^5 = 1: 0 -> x1 -> x2 -> x3 -> x4 -> 0
        if n >= 5:
            cycle = rng.sample(range(1, n), 4)
            table[0][0] = cycle[0]
            table[cycle[0]][0] = cycle[1]
            table[cycle[1]][0] = cycle[2]
            table[cycle[2]][0] = cycle[3]
            table[cycle[3]][0] = 0
            # Inverses
            table[0][1] = cycle[3]
            table[cycle[0]][1] = 0
            table[cycle[1]][1] = cycle[0]
            table[cycle[2]][1] = cycle[1]
            table[cycle[3]][1] = cycle[2]
        else:
            # Index < 5, a must act trivially or with shorter period
            # But a has order 5, so if index < 5, a must map 0 to 0
            table[0][0] = 0
            table[0][1] = 0
    else:
        # b^5 = 1: 0 -> x1 -> x2 -> x3 -> x4 -> 0
        if n >= 5:
            cycle = rng.sample(range(1, n), 4)
            table[0][2] = cycle[0]
            table[cycle[0]][2] = cycle[1]
            table[cycle[1]][2] = cycle[2]
            table[cycle[2]][2] = cycle[3]
            table[cycle[3]][2] = 0
            # Inverses
            table[0][3] = cycle[3]
            table[cycle[0]][3] = 0
            table[cycle[1]][3] = cycle[0]
            table[cycle[2]][3] = cycle[1]
            table[cycle[3]][3] = cycle[2]
        else:
            table[0][2] = 0
            table[0][3] = 0
    
    # Now use Todd-Coxeter procedure to complete the table
    # Define relators: a^5, b^5
    # For each coset and each generator, if undefined, define it and propagate
    
    # We'll use a simpler approach: randomly fill in and check consistency
    # This is a heuristic, not full Todd-Coxeter
    
    # Actually, let's do a proper simplified Todd-Coxeter
    # The key relations are a^5=1 and b^5=1
    
    # For each coset i, the action of a must have order dividing 5
    # The action of b must have order dividing 5
    
    # Let's try to complete the table by defining undefined entries
    # and checking for contradictions
    
    for step in range(max_steps):
        # Find an undefined entry
        found = False
        for i in range(n):
            for j in range(4):
                if table[i][j] == -1:
                    # Define table[i][j]
                    # Choose a random coset, preferring unvisited ones
                    # Count how many cosets are reachable from i via generator j's orbit
                    # Simple: pick random
                    target = rng.randint(0, n - 1)
                    table[i][j] = target
                    
                    # Now propagate: if table[target][j'] is defined, check consistency
                    # For a^5 = 1: table[table[table[table[table[i][0]][0]][0]][0]][0] should be i
                    # This is complex, so we'll check at the end
                    
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Table is complete, check for contradictions
            valid = True
            for i in range(n):
                # Check a^5 = 1
                cur = i
                for _ in range(5):
                    if table[cur][0] == -1:
                        valid = False
                        break
                    cur = table[cur][0]
                if cur != i:
                    valid = False
                    break
                # Check b^5 = 1
                cur = i
                for _ in range(5):
                    if table[cur][2] == -1:
                        valid = False
                        break
                    cur = table[cur][2]
                if cur != i:
                    valid = False
                    break
                # Check inverses
                if table[i][1] != -1 and table[i][0] != -1:
                    if table[table[i][0]][1] != i:
                        valid = False
                        break
                if table[i][3] != -1 and table[i][2] != -1:
                    if table[table[i][2]][3] != i:
                        valid = False
                        break
            if valid:
                return table, True
            else:
                # Contradiction, this attempt failed
                return None, False
        
        # Also check for contradictions periodically
        if step % 100 == 0:
            for i in range(n):
                cur = i
                for _ in range(5):
                    if table[cur][0] == -1:
                        break
                    cur = table[cur][0]
                else:
                    if cur != i:
                        return None, False
                cur = i
                for _ in range(5):
                    if table[cur][2] == -1:
                        break
                    cur = table[cur][2]
                else:
                    if cur != i:
                        return None, False
    
    return None, False


def _check_girth_from_table(table, g):
    """
    Check if the Schreier graph defined by the table has girth >= g.
    The graph has vertices 0..n-1, and edges from i to table[i][j] for j in {0,1,2,3}.
    But we need to deduplicate: edge (i, table[i][0]) is same as (table[i][0], i) via inverse.
    """
    n = len(table)
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(4):
            j2 = table[i][j]
            if j2 != -1 and j2 != i:
                adj[i].add(j2)
    
    # Check degrees - should be 4 for a proper Schreier graph
    # But some edges may be duplicated if the action is not free
    # For now, just check girth
    
    # Girth check via non-backtracking BFS
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            if depth + 1 >= g:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _table_to_edges(table):
    """Convert a permutation table to an edge list."""
    n = len(table)
    edges = set()
    for i in range(n):
        for j in range(4):
            j2 = table[i][j]
            if j2 != -1 and j2 != i:
                e = (min(i, j2), max(i, j2))
                edges.add(e)
    return list(edges)


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # LUW baseline
    best_n, best_edges = _luw_graph(k, g)
    
    if k != 4 or g != 10:
        return best_n, best_edges
    
    # Try Todd-Coxeter enumeration for (4,10)
    rng = random.Random(12345)
    
    # The LUW graph has 320 vertices. Try to find smaller ones.
    # Start from smaller indices and work up.
    # But we need to be efficient - check girth quickly.
    
    # For C5 * C5, the Schreier graph of a subgroup H has degree 4
    # (each coset connects to 4 neighbors via a, a^-1, b, b^-1, but
    #  a and a^-1 give the same undirected edge, so degree is 4 if
    #  the action is free, or less if not)
    
    # Actually, for a free action, each vertex has exactly 4 neighbors.
    # The girth is the length of the shortest non-trivial word in a,b
    # that fixes a coset.
    
    # Try indices from 243 (lower bound) up to 319
    # But this is expensive, so let's be smart about it.
    
    # First, try a few random small indices to see if any work
    # Then if time permits, do a more systematic search
    
    # For efficiency, let's try indices that are multiples of 5
    # (since the group has order 5 generators, subgroups often have index divisible by 5)
    # Actually, not necessarily. Let's just try a range.
    
    # Given time constraints, let's try a limited number of indices
    # with multiple random attempts each
    
    indices_to_try = []
    # Try indices near the LUW size first, then smaller
    for idx in range(243, 320):
        indices_to_try.append(idx)
    
    # Shuffle to add randomness
    rng.shuffle(indices_to_try)
    
    # But prioritize smaller indices
    indices_to_try.sort()
    
    found_smaller = False
    for idx in indices_to_try:
        if time.time() > deadline:
            break
        
        if idx >= best_n:
            break  # No point trying indices >= best_n
        
        # Try to find a valid subgroup of this index
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            table, success = _tc_single(idx, rng, max_steps=50000)
            if not success or table is None:
                continue
            
            # Check girth
            if _check_girth_from_table(table, g):
                edges = _table_to_edges(table)
                # Verify it's a valid graph
                if _validate_graph(idx, edges, k, g):
                    best_n = idx
                    best_edges = edges
                    found_smaller = True
                    break
        
        if found_smaller:
            # Continue looking for even smaller
            pass
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
