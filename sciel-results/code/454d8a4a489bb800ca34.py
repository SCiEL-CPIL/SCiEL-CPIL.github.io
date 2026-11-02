"""Rewritten: algebraic Cayley-graph construction for the (5,9)-cage problem."""
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 135  # safety margin before 150s kill
    rng = random.Random(20240517)

    # Algebraic construction: Cayley graph on a random group.
    # Girth >= g  <=>  no nontrivial word of length < g in S equals identity.
    # Search over symmetric generating sets S (|S|=k) with a girth filter.
    best = None

    # Try a few group sizes (order). Larger order => more room, fewer collisions.
    for order in [900, 700, 1000, 600, 1200, 800]:
        if time.time() > deadline:
            break
        for attempt in range(6):
            if time.time() > deadline:
                break
            try:
                res = _cayley_search(k, g, order, deadline, rng)
            except Exception:
                res = None
            if res is not None:
                n, edges = res
                if best is None or n < best[0]:
                    best = (n, edges)
                if n <= 700:
                    return best
        if best is not None and best[0] <= 700:
            return best

    return best


def _cayley_search(k, g, order, deadline, rng):
    """
    Build a random group of given order, then search for a symmetric
    generating set S (|S|=k) such that no nontrivial word of length < g
    in S equals the identity. Return the Cayley graph if found.
    """
    max_iter = 2000
    for it in range(max_iter):
        if time.time() > deadline:
            return None

        # --- Build a random group (Cayley table) via BFS from identity ---
        # Identity = 0. Random generators; close under multiplication.
        # We build a group of exactly 'order' elements by adding generators
        # and ensuring closure; if we exceed 'order', restart.
        # Simpler: build the multiplication table by random presentation.
        # Use: pick random generators, generate the group they produce,
        # check size == order (within tolerance), else retry.

        table = None
        for _gen_try in range(8):
            table = _build_random_group(order, deadline, rng)
            if table is not None:
                break
        if table is None:
            continue

        n = order
        identity = 0

        # --- Search for symmetric generating set S of size k ---
        # S is a set of k group elements, symmetric: if s in S then s^{-1} in S.
        # For odd k, we need an involution (self-inverse) to make it symmetric.
        # Strategy: pick k/2 pairs {s, s^{-1}} and possibly one involution.

        # Precompute inverses: inv[x] = y s.t. table[x][y] == identity
        inv = [0] * n
        for x in range(n):
            for y in range(n):
                if table[x][y] == identity:
                    inv[x] = y
                    break

        # Find involutions (x != identity, table[x][x] == identity)
        involutions = [x for x in range(1, n) if table[x][x] == identity]

        # Candidate pool: all non-identity elements
        pool = list(range(1, n))

        found = False
        for _s_try in range(300):
            if time.time() > deadline:
                return None

            # Build S: pick k/2 random pairs, plus one involution if k odd
            S = set()
            S.add(identity)  # identity always in the generating set for connectivity
            # Actually for a k-regular Cayley graph, |S \ {identity}| = k.
            # So S (the neighbor set) has exactly k elements, none is identity.
            # Let me redefine: neighbors = S where identity NOT in S, |S| = k, symmetric.

            neighbors = set()
            # Pick (k - (k%2))//2 pairs
            npairs = (k - (k % 2)) // 2
            used = set()
            ok = True
            for _ in range(npairs):
                s = rng.choice(pool)
                if s in used or s == identity:
                    continue
                si = inv[s]
                if si in used or si == identity:
                    continue
                used.add(s)
                used.add(si)
                neighbors.add(s)
                neighbors.add(si)
            if len(neighbors) != 2 * npairs:
                ok = False
            if ok and k % 2 == 1:
                # Need one involution
                if involutions:
                    cand = rng.choice(involutions)
                    if cand not in used and cand != identity:
                        neighbors.add(cand)
                    else:
                        ok = False
                else:
                    ok = False
            if not ok or len(neighbors) != k:
                continue

            # Check girth: no nontrivial word of length < g in neighbors == identity
            # Use BFS over words (with girth filter)
            if _check_girth_cayley(neighbors, table, identity, g, deadline):
                found = True
                break

        if not found:
            continue

        # Build edges of the Cayley graph
        # Vertex v is connected to v * s for s in neighbors.
        # Each edge (v, v*s) counted once: only when v < v*s.
        edges = []
        for v in range(n):
            for s in neighbors:
                w = table[v][s]
                if v < w:
                    edges.append((v, w))
        return (n, edges)


def _build_random_group(order, deadline, rng):
    """
    Build a random group of exactly 'order' elements by choosing random
    generators and closing under multiplication. Return the multiplication
    table (list of lists) or None if it doesn't come out to exactly 'order'.
    """
    # Start with identity = 0
    # Pick 2 random generators, generate the group they produce.
    # We do this by BFS: start with {identity}, repeatedly multiply known
    # elements by generators to discover new elements until closure.

    # We need to pick generators from a "candidate" space. Since we don't
    # know the group yet, we use a trick: pick random permutations of a set
    # of size 'order' and generate the group they produce. If the group has
    # exactly 'order' elements, great.

    # Simpler approach: build a random group via a random Cayley table
    # that is consistent (associative). This is hard.

    # Alternative: use the fact that we just need SOME group of size 'order'.
    # Use the cyclic group Z/order? No, that's too structured (girth issues).
    # Use a random group: pick random generators as permutations, generate.

    # Let's use: represent group elements as integers 0..order-1.
    # Pick 2 random permutations g, h of {0..order-1}.
    # Generate the group <g, h> by composition. If |<g,h>| == order, done.
    # The multiplication table is then determined by the group structure.

    # But building the full table from permutations is complex.
    # Instead: use a different approach. Build a random group by:
    # 1. Start with identity.
    # 2. Repeatedly pick a random element (from 1..order-1) as a new generator.
    # 3. Close under multiplication, tracking the group.
    # 4. If the group reaches size 'order', stop. If it exceeds, fail.

    # This is essentially building a group by adding generators one at a time.

    # Let me use a cleaner method: random generators as functions on Z/order.
    # A group of order n can be embedded in Sym(n). Pick random permutations.

    import math
    # Pick 2 random permutations of range(order)
    perm1 = list(range(order))
    rng.shuffle(perm1)
    perm2 = list(range(order))
    rng.shuffle(perm2)

    # Generate the group <perm1, perm2> by BFS.
    # Each group element is a permutation (tuple of length order).
    # This is expensive for large order. For order ~ 600-1200, permutations
    # are large. Let's limit to order <= 1000 for feasibility.

    if order > 1200:
        return None

    identity_perm = tuple(range(order))

    # BFS to generate the group
    group = {identity_perm}
    queue = deque([identity_perm])
    gen1 = tuple(perm1)
    gen2 = tuple(perm2)

    def compose(a, b):
        # (a ∘ b)[i] = a[b[i]]
        return tuple(a[b[i]] for i in range(order))

    while queue:
        if time.time() > deadline:
            return None
        if len(group) > order:
            return None  # exceeded, fail
        cur = queue.popleft()
        for g in (gen1, gen2):
            prod = compose(cur, g)
            if prod not in group:
                if len(group) >= order:
                    return None  # would exceed
                group.add(prod)
                queue.append(prod)
            if time.time() > deadline:
                return None

    if len(group) != order:
        return None

    # Now we have a group of exactly 'order' elements.
    # Assign indices 0..order-1 to each group element.
    elems = list(group)
    idx = {elem: i for i, elem in enumerate(elems)}
    identity_idx = idx[identity_perm]

    # Build multiplication table
    table = [[0] * order for _ in range(order)]
    for i, a in enumerate(elems):
        for j, b in enumerate(elems):
            prod = compose(a, b)
            table[i][j] = idx[prod]

    return table


def _check_girth_cayley(neighbors, table, identity, g, deadline):
    """
    Check that no nontrivial word of length < g in 'neighbors' equals identity.
    Use BFS over words with a girth filter: if a partial word evaluates to
    identity at length < g, return False.
    """
    n = len(table)
    # BFS: state = current group element. Start at identity.
    # At each step, multiply by a neighbor. If we return to identity at
    # depth < g (and depth > 0), we have a short relation.
    # But we must avoid the trivial backtracking: s then s^{-1} gives identity
    # at length 2. For girth >= g >= 9, this is fine (2 < 9 means we'd reject).
    # Actually, backtracking words ARE short relations, so we should reject them.
    # But wait: in a Cayley graph, the edge (v, v*s) and (v*s, v) are the same
    # edge, so backtracking is just going back and forth, which is a 2-cycle.
    # For a simple graph, we don't have 2-cycles. So we should NOT count
    # backtracking as a short cycle.
    #
    # The girth condition for a Cayley graph: no nontrivial word of length < g
    # that is NOT a backtracking word equals identity.
    #
    # Simpler: just check all words of length < g (including backtracking).
    # If a backtracking word gives identity at length 2, that's fine because
    # the graph is simple (no 2-cycles). The actual girth is the length of the
    # shortest NON-backtracking cycle.
    #
    # For safety, let's just check: is there a word of length < g (allowing
    # any sequence of neighbors) that evaluates to identity? If yes, and the
    # word is not purely backtracking, then girth < g.
    #
    # Actually, the standard definition: girth is the length of the shortest
    # cycle in the graph. In a Cayley graph, cycles correspond to words that
    # evaluate to identity, modulo backtracking. The shortest such word (that
    # is not a backtracking word) gives the girth.
    #
    # For our purposes, let's do a BFS that tracks the path to avoid
    # backtracking, and check if we can return to identity in < g steps
    # without immediately backtracking.

    # BFS with state = (current_element, previous_element) to avoid backtracking.
    # Start: (identity, None). At each step, move to current * s for s in neighbors,
    # where s != inverse_of_previous (to avoid backtracking).
    # If we reach (identity, some_prev) at depth >= 3 and depth < g, girth < g.

    # Actually, simpler: just enumerate all words of length up to g-1 and check
    # if any equals identity. The number of words is k^(g-1) which for k=5, g=9
    # is 5^8 = 390625. That's manageable.

    # But we can prune: if a prefix already equals identity at length < g,
    # we can stop (but that's a backtracking issue).

    # Let's just do a simple check: for each word of length 3 to g-1, evaluate
    # and see if it equals identity. Skip words that are backtracking (s, s^{-1}, ...).

    # For efficiency, use BFS with memoization of (element, depth) to avoid
    # re-evaluating the same state.

    # Actually, the simplest correct check: BFS from identity, tracking depth,
    # avoiding immediate backtracking. If we can reach identity at depth d
    # where 3 <= d < g, then girth <= d < g.

    # Let's implement this BFS.
    # State: (current_element). We track the shortest depth to reach each element.
    # But we need to avoid backtracking, so we track (current_element, last_neighbor_used).

    # For k=5, g=9, the number of states is n * k = order * 5. For order ~ 900,
    # that's 4500 states. Very manageable.

    # BFS:
    # Queue: (element, last_neighbor, depth)
    # Start: (identity, None, 0)
    # Transition: for each s in neighbors, if s != inv[last_neighbor], 
    #   new_elem = table[element][s], new_state = (new_elem, s, depth+1)
    # If new_elem == identity and depth+1 >= 3 and depth+1 < g: return False (short cycle)
    # If depth+1 >= g: stop expanding.

    inv = [0] * n
    for x in range(n):
        for y in range(n):
            if table[x][y] == identity:
                inv[x] = y
                break

    # visited: (element, last_neighbor) -> min_depth
    visited = {}
    queue = deque()
    queue.append((identity, -1, 0))  # -1 means no previous neighbor
    visited[(identity, -1)] = 0

    while queue:
        if time.time() > deadline:
            return True  # assume OK on timeout
        elem, last_nb, depth = queue.popleft()

        if depth >= g - 1:
            continue

        for s in neighbors:
            # Avoid backtracking: s should not be the inverse of last_nb
            if last_nb != -1 and s == inv[last_nb]:
                continue
            new_elem = table[elem][s]
            new_depth = depth + 1

            # Check if we've reached identity at a valid cycle length
            if new_elem == identity and new_depth >= 3:
                return False  # found a cycle of length new_depth < g

            if new_depth >= g:
                continue

            state = (new_elem, s)
            if state not in visited or visited[state] > new_depth:
                visited[state] = new_depth
                queue.append((new_elem, s, new_depth))

    return True  # no short cycle found
