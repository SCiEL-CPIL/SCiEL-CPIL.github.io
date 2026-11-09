"""Capacitated VRP heuristic: savings-based marginal-cost lookahead."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 115.0  # safety margin under the 120 s evaluator limit
_t0 = time.time()


def _savings_completion(dist, depot, customers, demands, cap):
    """Estimate total route length for a set of customers using Clarke-Wright
    savings: base = 2*sum(depot legs); subtract savings for links we can form
    while respecting capacity (greedy, degree <= 2)."""
    n = len(customers)
    if n == 0:
        return 0.0
    if n == 1:
        c = customers[0]
        return float(dist[depot, c] + dist[c, depot])

    # base cost if each customer were its own route
    base = 0.0
    for c in customers:
        base += dist[depot, c] + dist[c, depot]

    # candidate savings links (i, j) with i, j in customers
    links = []
    for a in range(n):
        ca = customers[a]
        for b in range(a + 1, n):
            cb = customers[b]
            sav = dist[depot, ca] + dist[depot, cb] - dist[ca, cb]
            if sav > 1e-9:
                links.append((sav, a, b))
    if not links:
        return base
    links.sort(reverse=True)

    deg = np.zeros(n, dtype=int)
    # route loads: each component is a chain; track load per chain via union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # load per root
    load = np.zeros(n, dtype=float)
    for i in range(n):
        load[i] = demands[customers[i]]

    total_sav = 0.0
    for sav, a, b in links:
        if deg[a] >= 2 or deg[b] >= 2:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue  # would create a cycle
        # capacity check: merging two chains must keep load <= cap
        if load[ra] + load[rb] > cap + 1e-9:
            continue
        # union
        if ra != rb:
            parent[ra] = rb
            load[rb] += load[ra]
        deg[a] += 1
        deg[b] += 1
        total_sav += sav
        if total_sav >= base:
            break
    return max(0.0, base - total_sav)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return int(depot)

    dist = distance_matrix
    unvisited = list(unvisited_nodes)
    cap = None  # not needed directly; rest_capacity governs feasibility

    # deadline guard: if little time left, use nearest feasible
    if time.time() - _t0 > _DEADLINE:
        feasible = [c for c in unvisited if demands[c] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(feiblest[np.argmin(dist[current_node, feasible])] if False else
                  feasible[np.argmin(dist[current_node, feasible])])

    # marginal cost of going current->c and then completing the rest from c
    best_score = float('inf')
    best_node = None
    best_is_depot = True

    # option: close route (go to depot)
    cost_depot = dist[current_node, depot]
    # after closing, remaining customers start fresh from depot; estimate their completion
    remaining_after_depot = unvisited  # all still unvisited
    # estimate total future cost if we close now:
    future_depot = _savings_completion(dist, depot, remaining_after_depot, demands, rest_capacity)
    # note: rest_capacity resets after closing; we approximate capacity as large for the estimate
    # (true capacity unknown here; use a generous cap so savings links are not over-restricted)
    score_depot = cost_depot + future_depot

    for c in unvisited:
        if demands[c] > rest_capacity:
            continue
        # marginal: current -> c, then complete rest from c (with reduced capacity)
        marginal = dist[current_node, c]
        rest_after = [x for x in unvisited if x != c]
        # estimate completion of rest starting from c with capacity rest_capacity - demands[c]
        # approximate: use savings completion but from c as the "start depot" is not standard;
        # we instead estimate: route continues c -> ... -> depot. Approximate by savings on rest
        # plus the tail c->depot is captured if rest empty.
        if not rest_after:
            future = dist[c, depot]
        else:
            # use savings completion for rest, plus we must get back to depot from c eventually;
            # approximate the whole continuation as savings completion of rest with c as an anchor:
            # simplest robust surrogate: savings_completion(rest) + dist[c, depot] (close from last)
            # but better: treat c as already-in-route; estimate rest route length via savings.
            future = _savings_completion(dist, depot, rest_after, demands, rest_capacity - demands[c]) + dist[c, depot]
        score = marginal + future
        if score < best_score:
            best_score = score
            best_node = c
            best_is_depot = False

    if best_node is None:
        return int(depot)
    if best_is_depot and score_depot <= best_score:
        return int(depot)
    return int(best_node)
# EVOLVE-BLOCK-END
