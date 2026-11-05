import time
import random
import math
import sys
from collections import defaultdict


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    all_diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.append(row[j] - row[i])
    return len(all_diffs) == len(set(all_diffs))


def _row_diffs(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i + 1, n)]


def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _solve_row_from_diffs(diff_set, max_val, rng, time_limit=None, start_time=None):
    """
    Given a set of 15 positive integers (the differences of a 6-element row),
    find a 6-element increasing sequence starting at 0 whose pairwise differences
    are exactly diff_set, with all elements <= max_val.
    Uses backtracking with forward checking.
    """
    if start_time is None:
        start_time = time.time()

    diffs = sorted(diff_set)
    n_elem = 6

    # Precompute: for each diff, which pairs (i,j) could produce it
    # For a 6-element row, pair (i,j) with i<j gives diff d = a[j]-a[i]
    # We need to assign each diff to exactly one pair

    # Backtracking: build row element by element
    # State: row so far, remaining diffs, used positions
    # Key insight: a[j] - a[i] for all i < j must be in diff_set

    def _backtrack(row, remaining_diffs, used_mask):
        if time.time() - start_time > time_limit:
            return None

        if len(row) == n_elem:
            if not remaining_diffs:
                return row[:]
            return None

        last = row[-1]
        # Try candidate values for next element
        # The next element must be > last and <= max_val
        # Also, the differences from all previous elements must be in remaining_diffs

        # Determine valid range for next element
        # For each previous element row[i], (val - row[i]) must be in remaining_diffs
        # So val = row[i] + d for some d in remaining_diffs
        # Also val > last and val <= max_val

        candidates = set()
        for i in range(len(row)):
            for d in remaining_diffs:
                val = row[i] + d
                if val > last and val <= max_val:
                    # Check all differences from previous elements
                    valid = True
                    for j in range(len(row)):
                        diff_j = val - row[j]
                        if diff_j not in remaining_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.add(val)

        # Sort candidates: prefer smaller values (tighter scope)
        candidates = sorted(candidates)

        # Randomize order slightly for diversity
        if len(candidates) > 4:
            # Keep first few in order, shuffle the rest
            keep = min(3, len(candidates))
            rest = candidates[keep:]
            rng.shuffle(rest)
            candidates = candidates[:keep] + rest

        for val in candidates:
            # Compute which diffs are used
            used_diffs = set()
            for i in range(len(row)):
                used_diffs.add(val - row[i])

            new_remaining = remaining_diffs - used_diffs

            # Forward checking: for the new element, future elements must be able
            # to produce diffs from new_remaining
            # Simple check: the smallest diff in new_remaining must be achievable
            # (i.e., there exists some future pair that could produce it)
            # This is complex, so we skip heavy forward checking

            result = _backtrack(row + [val], new_remaining, used_mask)
            if result is notNone:
                return result

        return None

    # Start with [0]
    result = _backtrack([0], set(diffs), 0)
    return result


def _flow_assignment(n_rows, n_diffs_per_row, M, cost_weights=None, rng=None):
    """
    Assign n_rows * n_diffs_per_row distinct differences (from 1..M) to n_rows rows.
    Each row gets exactly n_diffs_per_row differences.
    Uses min-cost max-flow.
    Returns: list of sets, one per row, or None if infeasible.
    """
    # Build flow network
    # Source -> diff nodes (capacity 1, cost 0)
    # Diff nodes -> row nodes (capacity 1, cost = cost_weight)
    # Row nodes -> sink (capacity n_diffs_per_row, cost 0)

    # For min-cost max-flow, we'll use a simple successive shortest path with potentials
    # Since the graph is bipartite and small enough, we can use a simpler approach

    # Actually, for this bipartite matching with costs, we can use the Hungarian-like
    # approach or just greedy with backtracking. Given the size (105 diffs, 7 rows),
    # let's use a direct approach.

    # Simpler: use min-cost max-flow with SPFA (Bellman-Ford based)
    # Nodes: source(0), diffs(1..M), rows(M+1..M+n_rows), sink(M+n_rows+1)
    # Total nodes: M + n_rows + 2

    total_nodes = M + n_rows + 2
    source = 0
    sink = total_nodes - 1
    diff_start = 1
    row_start = 1 + M

    # Adjacency list for residual graph
    # Each edge: (to, capacity, cost, reverse_index)
    adj = [[] for _ in range(total_nodes)]

    def add_edge(u, v, cap, cost):
        adj[u].append([v, cap, cost, len(adj[v])])
        adj[v].append([u, 0, -cost, len(adj[u]) - 1])

    # Source -> diff nodes
    for d in range(1, M + 1):
        add_edge(source, diff_start + d - 1, 1, 0)

    # Diff nodes -> row nodes
    for d in range(1, M + 1):
        d_node = diff_start + d - 1
        for r in range(n_rows):
            r_node = row_start + r
            # Cost: penalize large differences, and add row-specific weight
            if cost_weights is not None:
                cost = cost_weights[r] * d
            else:
                cost = d  # Prefer smaller differences globally
            add_edge(d_node, r_node, 1, cost)

    # Row nodes -> sink
    for r in range(n_rows):
        r_node = row_start + r
        add_edge(r_node, sink, n_diffs_per_row, 0)

    # Min-cost max-flow using successive shortest augmenting path with Bellman-Ford
    total_flow = 0
    total_cost = 0
    required_flow = n_rows * n_diffs_per_row

    # We'll use SPFA for shortest path (handles negative edges in residual)
    INF = float('inf')

    while total_flow < required_flow:
        # Find shortest path from source to sink in residual graph
        dist = [INF] * total_nodes
        in_queue = [False] * total_nodes
        prev_node = [-1] * total_nodes
        prev_edge = [-1] * total_nodes

        dist[source] = 0
        queue = [source]
        in_queue[source] = True

        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for ei, (v, cap, cost, _) in enumerate(adj[u]):
                if cap > 0 and dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev_node[v] = u
                    prev_edge[v] = ei
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if dist[sink] == INF:
            break  # No more augmenting paths

        # Augment
        path_flow = 1  # All capacities are 1 or n_diffs_per_row, but we augment 1 at a time
        v = sink
        while v != source:
            u = prev_node[v]
            ei = prev_edge[v]
            adj[u][ei][1] -= 1  # Decrease forward capacity
            rev = adj[u][ei][3]
            adj[v][rev][1] += 1  # Increase reverse capacity
            v = u

        total_flow += 1
        total_cost += dist[sink]

    if total_flow < required_flow:
        return None

    # Extract assignment
    assignment = [set() for _ in range(n_rows)]
    for d in range(1, M + 1):
        d_node = diff_start + d - 1
        for r in range(n_rows):
            r_node = row_start + r
            # Check if there's flow from d_node to r_node
            for edge in adj[d_node]:
                if edge[0] == r_node and edge[1] == 0:  # Forward edge used up
                    assignment[r].add(d)
                    break

    # Verify each row has exactly n_diffs_per_row diffs
    for r in range(n_rows):
        if len(assignment[r]) != n_diffs_per_row:
            return None

    return assignment


def _flow_assignment_greedy(n_rows, n_diffs_per_row, M, rng, time_limit=5.0):
    """
    Greedy assignment with randomization: assign differences to rows one by one.
    Each diff goes to the row that "needs" it most (fewest diffs assigned).
    Uses cost = diff_value * row_weight.
    """
    start = time.time()
    total_needed = n_rows * n_diffs_per_row

    # Start with all differences 1..M available
    available = set(range(1, M + 1))
    assignment = [set() for _ in range(n_rows)]

    # Randomize the order of rows
    row_order = list(range(n_rows))
    rng.shuffle(row_order)

    for r in row_order:
        # Assign n_diffs_per_row differences to row r
        # Prefer smaller differences
        # But also consider what's left for other rows
        assigned_count = 0
        # Try to pick the smallest available differences
        sorted_avail = sorted(available)

        # Use a weighted selection: prefer small diffs but with some randomness
        for _ in range(n_diffs_per_row):
            if time.time() - start > time_limit:
                return None
            if not sorted_avail:
                return None

            # Weight: 1/d (prefer small) with random perturbation
            candidates = sorted_avail[:min(30, len(sorted_avail))]
            weights = [1.0 / d * (1.0 + rng.random() * 0.5) for d in candidates]
            total_w = sum(weights)
            r_val = rng.random() * total_w
            cum = 0
            chosen = candidates[0]
            for i, (d, w) in enumerate(zip(candidates, weights)):
                cum += w
                if r_val <= cum:
                    chosen = d
                    break

            assignment[r].add(chosen)
            available.remove(chosen)
            sorted_avail.remove(chosen)
            assigned_count += 1

    return assignment


def _flow_assignment_optimized(n_rows, n_diffs_per_row, M, rng, time_limit=10.0):
    """
    Optimized assignment: use min-cost max-flow but with a smaller effective M
    and better cost structure.
    """
    start = time.time()

    # Use the flow-based approach but with a tighter M
    # The counting bound is ceil(sqrt(2 * total_diffs))
    total_diffs = n_rows * n_diffs_per_row
    counting_bound = math.ceil(math.sqrt(2 * total_diffs))

    # Try M from counting_bound upward
    for M_try in range(counting_bound, min(M, counting_bound + 30)):
        if time.time() - start > time_limit:
            break

        # Cost: for each row, prefer to use smaller differences
        # But we want diversity across rows
        # Use cost = d * (1 + 0.1 * r) to slightly differentiate rows
        cost_weights = [1.0 + 0.1 * r for r in range(n_rows)]

        result = _flow_assignment(n_rows, n_diffs_per_row, M_try, cost_weights, rng)
        if result is not None:
            # Verify: all diffs are distinct
            all_diffs = []
            for r in range(n_rows):
                all_diffs.extend(result[r])
            if len(all_diffs) == len(set(all_diffs)):
                return result

    return None


def _repair_assignment(assignment, n_rows, n_diffs_per_row, M, rng, time_limit=20.0):
    """
    Repair an infeasible assignment by re-solving the flow with additional
    difference slots. If a row's diff set is not additive, we try to swap
    differences with other rows to make it additive.
    """
    start = time.time()
    assignment = [set(s) for s in assignment]

    # Check which rows are feasible
    feasible = []
    for r in range(n_rows):
        row = _solve_row_from_diffs(assignment[r], M, rng, time_limit=2.0, start_time=start)
        feasible.append(row is not None)

    # If all feasible, return
    if all(feasible):
        return assignment

    # For infeasible rows, try to swap differences with feasible rows
    # to make them additive
    for attempt in range(50):
        if time.time() - start > time_limit:
            break

        # Find an infeasible row
        infeasible_rows = [r for r in range(n_rows) if not feasible[r]]
        if not infeasible_rows:
            break

        r_bad = rng.choice(infeasible_rows)
        r_good = rng.choice([r for r in range(n_rows) if feasible[r]]) if any(feasible) else None

        if r_good is None:
            # All infeasible: try random perturbations
            # Remove a random diff from r_bad and add a new one
            if assignment[r_bad]:
                d_remove = rng.choice(list(assignment[r_bad]))
                assignment[r_bad].discard(d_remove)
                # Add a random new diff
                new_d = rng.randint(1, M)
                if new_d not in assignment[r_bad]:
                    assignment[r_bad].add(new_d)
                    row = _solve_row_from_diffs(assignment[r_bad], M, rng, time_limit=1.0, start_time=start)
                    feasible[r_bad] = row is not None
            continue

        # Try to swap a diff between r_bad and r_good
        if assignment[r_bad] and assignment[r_good]:
            d1 = rng.choice(list(assignment[r_bad]))
            d2 = rng.choice(list(assignment[r_good]))
            if d1 != d2:
                assignment[r_bad].discard(d1)
                assignment[r_good].discard(d2)
                assignment[r_bad].add(d2)
                assignment[r_good].add(d1)

                # Re-check feasibility
                row_bad = _solve_row_from_diffs(assignment[r_bad], M, rng, time_limit=1.0, start_time=start)
                row_good = _solve_row_from_diffs(assignment[r_good], M, rng, time_limit=1.0, start_time=start)
                feasible[r_bad] = row_bad is not None
                feasible[r_good] = row_good is not None

    return assignment


def _local_search_reduce(rows, num_elements, rng, time_limit):
    """
    Local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    # Compute global mask
    global_mask = 0
    row_masks = []
    for row in current_rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m

    n_rows = len(current_rows)
    n_elem = num_elements

    max_iter = 200000
    stagnation = 0
    max_stagnation = 10000

    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break

        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]

        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope

        if low >= old_val:
            continue

        new_val = rng.randint(low, old_val - 1)

        # Check if moving is valid
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
        if not valid:
            row[pos] = old_val
            continue

        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)

        other_mask = global_mask & ~row_masks[row_idx]
        if new_row_mask & other_mask:
            row[pos] = old_val
            continue

        # Accept the move
        row[pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask

        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
            stagnation = 0
        else:
            stagnation += 1

    return best_rows, best_scope


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=30.0):
    """
    Simulated annealing to minimize scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    global_mask = 0
    row_masks = []
    for row in current_rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m

    n_rows = len(current_rows)
    n_elem = num_elements

    temp = initial_temp
    cooling = 0.9997
    min_temp = 1.0

    iterations = 0
    max_iter = 300000

    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1

        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]

        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 15

        if low >= high:
            continue

        if old_val > low + 2 and rng.random() < 0.6:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)

        if new_val == old_val:
            continue

        # Check validity
        row[pos] = new_val
        diffs_in_row = set()
        valid = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs_in_row:
                    valid = False
                    break
                diffs_in_row.add(d)
        if not valid:
            row[pos] = old_val
            continue

        new_row_mask = 0
        for d in diffs_in_row:
            new_row_mask |= (1 << d)

        other_mask = global_mask & ~row_masks[row_idx]
        if new_row_mask & other_mask:
            row[pos] = old_val
            continue

        # Accept or reject based on SA
        row[pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = (global_mask & ~old_row_mask) | new_row_mask

        new_scope = _scope(current_rows)
        delta = new_scope - current_scope

        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            row[pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask

        temp *= cooling

    return best_rows, best_scope


def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin from 150s

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Flow-based assignment + structure recovery
    # Try multiple M values and cost structures
    for seed in range(20):
        if time.time() > DEADLINE - 60:
            break
        rng_local = random.Random(seed * 17 + 3)

        # Try different M values
        for M_offset in range(0, 40, 3):
            if time.time() > DEADLINE - 60:
                break
            M = counting_bound + M_offset

            # Get assignment via flow
            assignment = _flow_assignment_optimized(
                n_rows, num_pairs, M, rng_local, time_limit=5.0
            )
            if assignment is None:
                # Try greedy assignment
                assignment = _flow_assignment_greedy(
                    n_rows, num_pairs, M, rng_local, time_limit=3.0
                )
            if assignment is None:
                continue

            # Repair infeasible assignments
            assignment = _repair_assignment(
                assignment, n_rows, num_pairs, M, rng_local, time_limit=10.0
            )

            # Try to recover rows from diff sets
            rows = []
            valid = True
            for r in range(n_rows):
                row = _solve_row_from_diffs(
                    assignment[r], M, rng_local, time_limit=3.0,
                    start_time=start_time
                )
                if row is None:
                    valid = False
                    break
                rows.append(row)

            if valid and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows
                if s <= counting_bound + 10:
                    break

        if best_scope <= counting_bound + 10:
            break

    # Phase 2: If flow-based approach failed, use constructive backtracking
    if best_rows is None or best_scope > counting_bound + 30:
        # Simple backtracking construction
        for seed in range(10):
            if time.time() > DEADLINE - 50:
                break
            rng_local = random.Random(seed * 23 + 7)
            M = counting_bound + 20

            rows = []
            used_mask = 0
            valid = True
            for r in range(n_rows):
                row = [0]
                for pos in range(1, num_elements):
                    found = False
                    for val in range(row[-1] + 1, M + 1):
                        new_diffs = set()
                        valid_inner = True
                        for prev in row:
                            d = val - prev
                            if d in new_diffs:
                                valid_inner = False
                                break
                            new_diffs.add(d)
                        if not valid_inner:
                            continue
                        new_mask = 0
                        for d in new_diffs:
                            new_mask |= (1 << d)
                        if new_mask & used_mask:
                            continue
                        row.append(val)
                        used_mask |= new_mask
                        found = True
                        break
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
                rows.append(row)

            if valid and _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows

    # Fallback if nothing found
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)

    # Phase 3: Local search to reduce scope
    remaining = DEADLINE - time.time()
    if remaining > 40:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 40
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 4: Simulated annealing for fine-tuning
    remaining = DEADLINE - time.time()
    if remaining > 30:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20,
            initial_temp=max(5.0, best_scope * 0.1)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Phase 5: Final local search pass
    remaining = DEADLINE - time.time()
    if remaining > 20:
        improved_rows, improved_scope = _local_search_reduce(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 10
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
