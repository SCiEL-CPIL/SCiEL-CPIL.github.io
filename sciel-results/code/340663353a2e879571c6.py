"""
Difference triangle set (7,5) solver.

Combines:
- Small-difference-awareness and scope-bounded candidate enumeration (inspiration 1)
- Compatibility graph + maximum clique + local search + simulated annealing + polish (current)

The small-difference heuristic biases construction toward rows that consume fewer
small differences, since small differences are the scarce resource that forces
larger gaps. Scope-bounded enumeration ensures we never build a row exceeding
our best known scope.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


def _small_diff_count(diffs, threshold=15):
    """Count how many differences are below the threshold."""
    return sum(1 for d in diffs if d < threshold)


def _generate_row_greedy_small_aware(used, k, rng, best_scope=None, pool_size=5):
    """
    Generate a row using small-difference awareness.
    Prefers candidates that introduce fewer small differences.
    Bounded by best_scope if provided.
    Returns (row, diffs) or (None, None).
    """
    row = [0]
    new_diffs = set()
    x = 0
    threshold = 15

    while len(row) < k + 1:
        candidates = []
        x += 1
        upper = best_scope - 1 if best_scope and best_scope < float('inf') else 200

        while x <= upper and len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                small_count = _small_diff_count(new, threshold)
                candidates.append((small_count, x, new))
            x += 1

        if not candidates:
            return None, None

        # Sort by small_count then x
        candidates.sort(key=lambda c: (c[0], c[1]))

        # Pick best with 70% probability, random from top 5 otherwise
        if rng.random() < 0.7:
            chosen = candidates[0]
        else:
            top_k = min(5, len(candidates))
            chosen = rng.choice(candidates[:top_k])

        x_val = chosen[1]
        new_set = chosen[2]
        row.append(x_val)
        new_diffs |= new_set
        x = x_val

    return row, frozenset(new_diffs)


def _generate_row_greedy_pure(used, k, best_scope=None):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    upper = best_scope - 1 if best_scope and best_scope < float('inf') else 200

    while len(row) < k + 1:
        x += 1
        if x > upper:
            return None, None
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new

    return row, frozenset(new_diffs)


def _generate_row_random_walk(k, rng, max_scope=150, used=None):
    """
    Generate a random row of k+1 integers starting at 0 with max entry <= max_scope.
    Returns (row, diffs) or None.
    """
    if used is None:
        used = set()
    row = [0]
    new_diffs = set()
    x = 0
    threshold = 15

    while len(row) < k + 1:
        min_val = x + 1
        if min_val > max_scope:
            return None, None
        x = rng.randint(min_val, min(max_scope, x + rng.randint(1, 10)))
        new = {x - a for a in row}
        if new & new_diffs or new & used:
            continue
        # Small-difference awareness: reject if too many small diffs
        if _small_diff_count(new, threshold) > 3:
            continue
        row.append(x)
        new_diffs |= new

    return row, frozenset(new_diffs)


def _generate_candidate_pool(n_rows, k, rng, pool_size=200, best_scope=None):
    """
    Generate a pool of candidate rows using multiple strategies.
    Returns list of (row, diffs) tuples.
    """
    candidates = []

    # Start with small-aware greedy for a base
    used = set()
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_small_aware(used, k, rng, best_scope, pool_size=3)
        if row is None:
            row, diffs = _generate_row_greedy_pure(used, k, best_scope)
        if row is not None:
            candidates.append((row, diffs))
            used |= diffs

    # Generate more candidates with randomization
    for _ in range(pool_size):
        used_subset = set()
        for c in candidates:
            if rng.random() < 0.3:
                used_subset |= c[1]

        ps = rng.choice([2, 3, 5, 8])
        row, diffs = _generate_row_greedy_small_aware(used_subset, k, rng, best_scope, pool_size=ps)
        if row is None:
            row, diffs = _generate_row_greedy_pure(used_subset, k, best_scope)
        if row is not None:
            candidates.append((row, diffs))

    # Also try some random walk rows
    for _ in range(pool_size // 2):
        row, diffs = _generate_row_random_walk(k, rng, max_scope=120, used=set())
        if row is not None:
            candidates.append((row, diffs))

    return candidates


def _build_compatibility_graph(candidates):
    """
    Build a compatibility graph where edge (i, j) means candidates i and j
    have disjoint difference sets.
    Returns adjacency list as list of sets.
    """
    n = len(candidates)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (candidates[i][1] & candidates[j][1]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _max_clique_branch_and_bound(adj, deadline):
    """
    Branch-and-bound for maximum clique using coloring upper bound.
    """
    n = len(adj)
    if n == 0:
        return []

    best_clique = []
    best_size = 0
    node_count = [0]

    def color_upper_bound(vertices):
        if not vertices:
            return [], []

        vertex_colors = {}
        verts_sorted = sorted(vertices, key=lambda v: -len(adj[v]))

        for v in verts_sorted:
            used = set()
            for u in adj[v]:
                if u in vertex_colors:
                    used.add(vertex_colors[u])
            c = 0
            while c in used:
                c += 1
            vertex_colors[v] = c

        def sort_key(v):
            return (vertex_colors[v], -len(adj[v]))

        ordered = sorted(vertices, key=sort_key)

        upper_bounds = [0] * len(ordered)
        max_color = 0
        for i in range(len(ordered) - 1, -1, -1):
            max_color = max(max_color, vertex_colors[ordered[i]] + 1)
            upper_bounds[i] = max_color

        return ordered, upper_bounds

    def expand(clique, candidates):
        nonlocal best_clique, best_size

        if time.time() > deadline:
            return

        node_count[0] += 1

        if not candidates:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return

        ordered, upper_bounds = color_upper_bound(candidates)

        for i, v in enumerate(ordered):
            if time.time() > deadline:
                return

            if len(clique) + upper_bounds[i] <= best_size:
                break

            compatible = True
            for u in clique:
                if u not in adj[v]:
                    compatible = False
                    break
            if not compatible:
                continue

            clique.append(v)
            new_candidates = [u for u in ordered[i + 1:] if u in adj[v]]
            expand(clique, new_candidates)
            clique.pop()

    all_vertices = list(range(n))
    expand([], all_vertices)

    return best_clique


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows.
    Uses shuffled row orders to break symmetry.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    scope = _total_scope(rows)
                    improved = True
                    break

            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        scope = _total_scope(rows)
                        improved = True
                        break

    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    best_scope = _total_scope(rows)

    for _ in range(5):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break

    return rows


def _simulated_annealing_scope(rows, k, rng, deadline):
    """
    Simulated annealing to escape local optima in scope minimization.
    Occasionally accepts worse solutions to explore the search space.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    current_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope

    temperature = 10.0
    cooling_rate = 0.995

    while time.time() < deadline and temperature > 0.1:
        # Pick a random row and position
        idx = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, len(rows[idx]) - 1)
        row = rows[idx]
        old_val = row[pos]

        # Try a random new value
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        if new_val <= row[pos - 1] or (pos + 1 < len(row) and new_val >= row[pos + 1]):
            continue

        # Compute diff changes
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)

        new_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            if i < pos:
                new_diffs.add(new_val - row[i])
            else:
                new_diffs.add(row[i] - new_val)

        remaining = used - old_diffs
        if new_diffs & remaining:
            continue

        # Compute new scope
        new_scope = max(current_scope, new_val)
        if new_val == current_scope:
            other_max = max(max(r) for i, r in enumerate(rows) if i != idx)
            new_scope = max(other_max, new_val)

        # Accept if better or with probability exp(-delta/temperature)
        delta_scope = new_scope - current_scope
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temperature):
            # Accept
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            current_scope = new_scope

            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]

        temperature *= cooling_rate

    return best_rows


def _multi_start_local_search(n_rows, k, rng, deadline, best_scope=None):
    """
    Multiple restarts of local search with different initial solutions.
    Keeps the best result found.
    """
    best_rows = None
    best_local_scope = best_scope if best_scope else float('inf')

    start_time = time.time()
    num_starts = 0

    while time.time() < deadline and num_starts < 10:
        num_starts += 1
        # Generate a fresh solution
        used = set()
        rows = []
        for _ in range(n_rows):
            # Mix of small-aware greedy and pure greedy
            if rng.random() < 0.6:
                row, diffs = _generate_row_greedy_small_aware(used, k, rng, best_local_scope, pool_size=rng.choice([2, 3, 5]))
            else:
                row, diffs = _generate_row_greedy_pure(used, k, best_local_scope)
            if row is None:
                row, diffs = _generate_row_greedy_pure(used, k, None)
            if row is not None:
                rows.append(row)
                used |= diffs

        if len(rows) < n_rows:
            continue

        # Local search
        rows = _local_search_scope(rows, k, rng, deadline - 5)
        scope = _total_scope(rows)

        if scope < best_local_scope:
            best_local_scope = scope
            best_rows = [r[:] for r in rows]

    return best_rows


def _greedy_construct(n_rows, k, best_scope=None):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k, best_scope)
        if row is None:
            # Retry without scope bound
            row, diffs = _generate_row_greedy_pure(used, k, None)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


def _validate(rows, n_rows, k):
    """Validate that rows are correct."""
    if len(rows) != n_rows:
        return False
    all_d = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = set(_diffs_of_row(row))
        if rd & all_d:
            return False
        all_d |= rd
    return True


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin

    # Step 1: Initial greedy construction with small-difference awareness
    best_scope = float('inf')
    best_rows = None

    # Try small-aware greedy first
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_small_aware(used, k, rng, None, pool_size=3)
        if row is None:
            row, diffs = _generate_row_greedy_pure(used, k, None)
        if row is not None:
            rows.append(row)
            used |= diffs

    if len(rows) == n_rows:
        best_rows = [r[:] for r in rows]
        best_scope = _total_scope(rows)
    else:
        # Fallback to pure greedy
        rows = _greedy_construct(n_rows, k)
        if rows is not None:
            best_rows = [r[:] for r in rows]
            best_scope = _total_scope(rows)

    # Step 2: Generate candidate pool and try maximum clique
    if best_rows is not None and time.time() < deadline - 60:
        candidates = _generate_candidate_pool(n_rows, k, rng, pool_size=300, best_scope=best_scope)

        # Deduplicate
        seen_diffs = set()
        unique_candidates = []
        for row, diffs in candidates:
            if diffs not in seen_diffs:
                seen_diffs.add(diffs)
                unique_candidates.append((row, diffs))

        candidates = unique_candidates

        if len(candidates) >= n_rows and time.time() < deadline - 50:
            adj = _build_compatibility_graph(candidates)
            clique = _max_clique_branch_and_bound(adj, deadline - 30)

            if len(clique) >= n_rows:
                selected = clique[:n_rows]
                clique_rows = [candidates[i][0] for i in selected]
                clique_scope = _total_scope(clique_rows)
                if clique_scope < best_scope:
                    best_scope = clique_scope
                    best_rows = [r[:] for r in clique_rows]

    # Step 3: Local search to reduce scope
    if best_rows is not None and time.time() < deadline - 25:
        rows = _local_search_scope(best_rows, k, rng, deadline - 25)
        scope = _total_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # Step 4: Simulated annealing to escape local optima
    if best_rows is not None and time.time() < deadline - 20:
        sa_rows = _simulated_annealing_scope(best_rows, k, rng, deadline - 20)
        sa_scope = _total_scope(sa_rows)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]

    # Step 5: Multi-start local search
    if best_rows is not None and time.time() < deadline - 15:
        ms_rows = _multi_start_local_search(n_rows, k, rng, deadline - 15, best_scope)
        if ms_rows is not None:
            ms_scope = _total_scope(ms_rows)
            if ms_scope < best_scope:
                best_scope = ms_scope
                best_rows = [r[:] for r in ms_rows]

    # Step 6: Final polish
    if best_rows is not None and time.time() < deadline - 5:
        rows = _polish_pass(best_rows, k, deadline - 5)
        scope = _total_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # Final validation
    if best_rows is not None and _validate(best_rows, n_rows, k):
        return best_rows

    # Fallback: pure greedy without scope bound
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)

    return rows

# EVOLVE-BLOCK-END
