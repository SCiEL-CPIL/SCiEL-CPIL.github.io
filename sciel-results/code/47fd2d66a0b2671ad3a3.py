"""Rewritten program: structural change to a product-biased value set + collision-aware edge selection + simulated-annealing local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

N = 14
M = 35
MAXV = 10000
DEADLINE = 85  # seconds; safety margin under 90s


def _distinct_count(vals):
    return len(set(vals))


def _objective(vals, edges):
    sums = [vals[i] + vals[j] for i, j in edges]
    prods = [vals[i] * vals[j] for i, j in edges]
    return max(len(set(sums)), len(set(prods)))


def _all_edges():
    return list(combinations(range(N), 2))


def _gen_product_biased_values(rng):
    """Generate 14 distinct positive ints <=10000 biased toward small factors to encourage product collisions."""
    # Start with a small base set of numbers with many small factors (highly composite-ish)
    base = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    vals = set()
    # Ensure distinct and within range
    for b in base:
        if 1 <= b <= MAXV:
            vals.add(b)
    while len(vals) < N:
        # Pick random small numbers
        v = rng.randint(1, 30)
        vals.add(v)
    vals = list(vals)
    rng.shuffle(vals)
    return vals


def _gen_random_values(rng):
    vals = set()
    while len(vals) < N:
        v = rng.randint(1, MAXV)
        vals.add(v)
    return list(vals)


def _select_edges_by_collision(vals, rng, all_edges):
    """Select M edges to maximize collision density in sums and products."""
    # Precompute sum and product for each edge
    edge_data = []
    for (i, j) in all_edges:
        s = vals[i] + vals[j]
        p = vals[i] * vals[j]
        edge_data.append((i, j, s, p))

    # Count frequencies
    sum_counts = {}
    prod_counts = {}
    for (i, j, s, p) in edge_data:
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1

    # Score each edge by how much it contributes to collisions
    def edge_score(i, j, s, p):
        # Higher score if sum or product is already frequent
        return sum_counts.get(s, 0) + prod_counts.get(p, 0)

    # Greedy selection: pick edges with highest collision potential
    selected = []
    selected_sums = set()
    selected_prods = set()
    remaining = list(edge_data)
    rng.shuffle(remaining)

    for _ in range(M):
        best_idx = 0
        best_score = -1
        for idx, (i, j, s, p) in enumerate(remaining):
            # Score: how many existing selected edges share this sum or product
            score = 0
            if s in selected_sums:
                score += 2
            if p in selected_prods:
                score += 2
            # Also consider global frequency
            score += sum_counts.get(s, 0) * 0.1 + prod_counts.get(p, 0) * 0.1
            if score > best_score:
                best_score = score
                best_idx = idx
        i, j, s, p = remaining.pop(best_idx)
        selected.append((i, j))
        selected_sums.add(s)
        selected_prods.add(p)

    return selected


def _local_search(vals, edges, rng, time_limit):
    """Simulated annealing local search on values and edges."""
    edges = list(edges)
    current_obj = _objective(vals, edges)
    best_vals = list(vals)
    best_edges = list(edges)
    best_obj = current_obj

    start_time = time.time()
    temp = 10.0
    cooling = 0.999
    all_edges_set = set(_all_edges())

    while time.time() - start_time < time_limit:
        # Choose move type
        move_type = rng.random()
        new_vals = list(vals)
        new_edges = list(edges)

        if move_type < 0.4:
            # Perturb values
            idx = rng.randint(0, N - 1)
            old_v = new_vals[idx]
            # Try to move toward values that create collisions
            new_v = rng.randint(1, MAXV)
            if new_v != old_v:
                new_vals[idx] = new_v
        elif move_type < 0.7:
            # Swap two values
            i, j = rng.sample(range(N), 2)
            new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
        else:
            # Swap an edge
            if len(edges) > 0:
                e_idx = rng.randint(0, len(edges) - 1)
                old_edge = tuple(edges[e_idx])
                # Find a random alternative edge
                candidates = [e for e in all_edges_set if e not in set(tuple(ed) for ed in edges)]
                if candidates:
                    new_edge = rng.choice(candidates)
                    new_edges[e_idx] = list(new_edge)

        new_obj = _objective(new_vals, new_edges)
        delta = new_obj - current_obj

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 0.01)):
            vals = new_vals
            edges = new_edges
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_vals = list(vals)
                best_edges = list(edges)

        temp *= cooling

    return best_vals, best_edges, best_obj


def construct():
    rng = random.Random(42)
    start_time = time.time()

    # Initial solution
    vals = _gen_product_biased_values(rng)
    all_edges = _all_edges()
    edges = _select_edges_by_collision(vals, rng, all_edges)
    current_obj = _objective(vals, edges)

    best_vals = list(vals)
    best_edges = list(edges)
    best_obj = current_obj

    # Multiple restarts with different strategies
    time_budget = 70  # seconds for search
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return {'values': best_vals, 'edges': [list(e) for e in best_edges]}

    # Run local search
    vals2, edges2, obj2 = _local_search(vals, edges, rng, time_budget - (time.time() - start_time))
    if obj2 < best_obj:
        best_obj = obj2
        best_vals = vals2
        best_edges = edges2

    # Try a few more random restarts if time permits
    for restart in range(5):
        if time.time() - start_time > time_budget - 5:
            break
        vals_r = _gen_random_values(rng)
        edges_r = _select_edges_by_collision(vals_r, rng, all_edges)
        obj_r = _objective(vals_r, edges_r)
        if obj_r < best_obj:
            best_obj = obj_r
            best_vals = vals_r
            best_edges = edges_r

        # Quick local search
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 3:
            vals_l, edges_l, obj_l = _local_search(vals_r, edges_r, rng, remaining_time - 1)
            if obj_l < best_obj:
                best_obj = obj_l
                best_vals = vals_l
                best_edges = edges_l

    # Ensure types are correct
    best_vals = [int(v) for v in best_vals]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    # Final validation
    assert len(best_vals) == N
    assert len(set(best_vals)) == N
    assert all(1 <= v <= MAXV for v in best_vals)
    assert len(best_edges) == M
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        assert e[0] != e[1]
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
        assert 0 <= e[0] < N and 0 <= e[1] < N

    return {'values': best_vals, 'edges': best_edges}
# EVOLVE-BLOCK-END
