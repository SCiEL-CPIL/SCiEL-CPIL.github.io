import time
import random

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190
EDGE_LIST = [(u, v) for u in range(N) for v in range(u + 1, N)]
EDGE_INDEX = {(u, v): i for i, (u, v) in enumerate(EDGE_LIST)}
EDGE_INDEX.update({(v, u): i for i, (u, v) in enumerate(EDGE_LIST)})

DEADLINE = 140.0  # safety margin before 150s kill
POP_SIZE = 24


def _popcount(x):
    return bin(x).count('1')


def _count_edges(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2


def _is_valid(adj):
    """K2,3-free iff no two vertices share 3+ common neighbours."""
    for u in range(N):
        for v in range(u + 1, N):
            if _popcount(adj[u] & adj[v]) >= 3:
                return False
    return True


def _can_add(adj, u, v):
    """Check if edge (u,v) can be added without creating a K2,3."""
    if adj[u] & (1 << v):
        return True
    # Pair (u,v): common neighbours must stay <= 2
    if _popcount(adj[u] & adj[v]) >= 3:
        return False
    # For each w in N(v): v becomes a new common neighbour of (u,w)
    # so |N(u) & N(w)| must be <= 1
    for w in range(N):
        if w != u and w != v and (adj[v] >> w) & 1:
            if _popcount(adj[u] & adj[w]) >= 2:
                return False
    # For each w in N(u): u becomes a new common neighbour of (v,w)
    for w in range(N):
        if w != u and w != v and (adj[u] >> w) & 1:
            if _popcount(adj[v] & adj[w]) >= 2:
                return False
    return True


def _greedy_build(rng):
    """Random-order greedy construction of a valid K2,3-free graph."""
    adj = [0] * N
    order = list(range(TOTAL_EDGES))
    rng.shuffle(order)
    for idx in order:
        u, v = EDGE_LIST[idx]
        if _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj


def _greedy_add_all(adj):
    """Add every valid missing edge; return (adj, number_added)."""
    added = 0
    for idx in range(TOTAL_EDGES):
        u, v = EDGE_LIST[idx]
        if not (adj[u] >> v) & 1 and _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            added += 1
    return adj, added


def _local_polish(adj, rng, time_limit):
    """Greedy add + a few 1-remove/2-add swaps to escape local optima."""
    adj = [b for b in adj]
    adj, _ = _greedy_add_all(adj)
    current = _count_edges(adj)
    start = time.time()
    for _ in range(40):
        if time.time() - start > time_limit:
            break
        # pick a random present edge to remove
        present = [i for i in range(TOTAL_EDGES)
                   if (adj[EDGE_LIST[i][0]] >> EDGE_LIST[i][1]) & 1]
        if not present:
            break
        idx = rng.choice(present)
        u, v = EDGE_LIST[idx]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj, added = _greedy_add_all(adj)
        new_count = _count_edges(adj)
        if new_count <= current:
            # revert
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj, _ = _greedy_add_all(adj)
        else:
            current = new_count
    return adj, current


def _crossover(p1, p2, rng):
    """Union of parent edges, then repair by re-adding dropped edges.
    Returns None if the repaired child is not valid or has fewer edges
    than the better parent (preserves validity, no degradation)."""
    best_parent_edges = max(_count_edges(p1), _count_edges(p2))
    child = [0] * N
    for u in range(N):
        child[u] = p1[u] | p2[u]
    # repair: re-add any edge that was dropped but is still addable
    child, _ = _greedy_add_all(child)
    if not _is_valid(child):
        return None
    if _count_edges(child) < best_parent_edges:
        return None
    return child


def _mutate(adj, rng):
    """Random edge toggle: add a valid missing edge, or remove a present one."""
    adj = [b for b in adj]
    if rng.random() < 0.7:
        # try to add a random missing edge
        for _ in range(20):
            idx = rng.randrange(TOTAL_EDGES)
            u, v = EDGE_LIST[idx]
            if not (adj[u] >> v) & 1 and _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                break
    else:
        # remove a random present edge (escape move)
        present = [i for i in range(TOTAL_EDGES)
                   if (adj[EDGE_LIST[i][0]] >> EDGE_LIST[i][1]) & 1]
        if present:
            idx = rng.choice(present)
            u, v = EDGE_LIST[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj


def construct():
    start = time.time()
    rng = random.Random(20240517)

    # ---- Initialize population with greedy builds ----
    population = []
    for _ in range(POP_SIZE):
        population.append(_greedy_build(rng))

    # ---- Genetic algorithm ----
    best_adj = max(population, key=_count_edges)
    best_edges = _count_edges(best_adj)

    gen = 0
    while time.time() - start < DEADLINE:
        new_pop = [best_adj]  # elitism: always keep the best

        while len(new_pop) < POP_SIZE and time.time() - start < DEADLINE:
            # tournament selection
            a = rng.choice(population)
            b = rng.choice(population)
            if _count_edges(a) < _count_edges(b):
                a, b = b, a
            child = _crossover(a, b, rng)
            if child is None:
                child = [x for x in a]
            child = _mutate(child, rng)
            # light polish to recover validity/edges
            child, _ = _greedy_add_all(child)
            if not _is_valid(child):
                child = [x for x in a]
            new_pop.append(child)

        population = new_pop

        # ---- Local polish on the current best each generation ----
        cur_best = max(population, key=_count_edges)
        polished, pc = _local_polish(cur_best, rng, 0.6)
        if pc > best_edges:
            best_edges = pc
            best_adj = [b for b in polished]
            population[0] = best_adj
            # also polish a couple of other strong members occasionally
            if gen % 3 == 0:
                for cand in population[1:4]:
                    if time.time() - start >= DEADLINE:
                        break
                    pol, c = _local_polish(cand, rng, 0.4)
                    if c > best_edges:
                        best_edges = c
                        best_adj = [b for b in pol]
                        population[0] = best_adj
        gen += 1

    # ---- Final: ensure best is valid and fully polished ----
    best_adj, _ = _greedy_add_all(best_adj)
    if not _is_valid(best_adj):
        # fall back to a fresh greedy build (should not happen)
        best_adj = _greedy_build(rng)
        best_adj, _ = _greedy_add_all(best_adj)

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print("edges:", len(res), "time:", round(time.time() - t0, 2))
