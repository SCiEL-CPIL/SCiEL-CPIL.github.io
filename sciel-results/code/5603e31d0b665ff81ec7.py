import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)


def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _canonical(cycle):
    if len(cycle) < 3:
        return None
    m = len(cycle)
    best = None
    for k in range(m):
        rot = cycle[k:] + cycle[:k]
        rev = list(reversed(rot))
        cand = rot if rot < rev else rev
        if best is None or cand < best:
            best = cand
    return tuple(best)


def _generate_cycle_pool(adj, n, edge_set, rng, time_limit):
    """Generate a pool of simple cycles (canonical tuples of vertices)."""
    start_time = time.time()
    pool = []
    seen = set()

    def _add(cycle):
        if len(cycle) < 3:
            return
        c = _canonical(cycle)
        if c is None:
            return
        # verify edges exist
        m = len(c)
        for i in range(m):
            a = c[i]
            b = c[(i + 1) % m]
            if tuple(sorted((a, b))) not in edge_set:
                return
        if c not in seen:
            seen.add(c)
            pool.append(list(c))

    # Strategy 1: many randomized DFS walks, harvest any back-edge cycle
    def _random_walk_cycle(start):
        # iterative DFS with random neighbor order; record parent to extract cycle
        visited = {start}
        parent = {start: None}
        stack = [start]
        while stack:
            node = stack[-1]
            nbrs = [v for v in adj[node] if v in visited or True]
            # try to find a visited neighbor (back edge) first
            back = None
            for v in adj[node]:
                if v in visited and v != parent.get(node):
                    back = v
                    break
            if back is not None:
                # extract cycle from back .. node
                cyc = [back]
                cur = node
                while cur != back:
                    cyc.append(cur)
                    cur = parent[cur]
                cyc.append(back)
                return cyc
            # else pick an unvisited neighbor
            unvisited = [v for v in adj[node] if v not in visited]
            if not unvisited:
                stack.pop()
                continue
            rng.shuffle(unvisited)
            nxt = unvisited[0]
            visited.add(nxt)
            parent[nxt] = node
            stack.append(nxt)
        return None

    # Strategy 2: DFS from a start, force return to start to get a start-based cycle
    def _start_cycle(start):
        visited = {start}
        parent = {start: None}
        path = [start]
        stack = [(start, iter(list(adj[start])))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt == start and len(path) >= 3:
                    return path[:]
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = node
                    path.append(nxt)
                    stack.append((nxt, iter(list(adj[nxt]))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if path:
                    path.pop()
                # mark back? keep visited to allow other branches (simple path constraint)
        return None

    # Generate a batch
    for _ in range(20000):
        if time.time() - start_time > time_limit:
            break
        # pick a start weighted by degree
        degs = [len(adj[v]) for v in range(n)]
        total = sum(degs)
        if total == 0:
            break
        r = rng.random() * total
        acc = 0
        start = n - 1
        for v in range(n):
            acc += degs[v]
            if r <= acc:
                start = v
                break

        if rng.random() < 0.5:
            cyc = _random_walk_cycle(start)
        else:
            cyc = _start_cycle(start)
        if cyc:
            _add(cyc)
        if len(pool) >= 3000:
            break

    # Strategy 3: enumerate all short cycles (length 3 and 4) deterministically
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # triangles
            for w in adj[v]:
                if w > v and w in adj[u]:
                    _add([u, v, w])
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w <= v:
                    continue
                if w in adj[u]:
                    continue
                for z in adj[w]:
                    if z <= w:
                        continue
                    if z in adj[u]:
                        _add([u, v, w, z])

    return pool


def _solve_sa(pool, edge_set, n, rng, time_limit):
    """
    Simulated Annealing to find a set of edge-disjoint cycles maximizing sum(|c|-1).
    """
    start_time = time.time()
    m = len(pool)
    if m == 0:
        return []

    # Precompute cycle edge lists and lengths
    cyc_edges = []
    cyc_lens = []
    for cyc in pool:
        L = len(cyc)
        es = []
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            es.append(tuple(sorted((a, b))))
        cyc_edges.append(es)
        cyc_lens.append(L)

    # Objective: maximize sum(|c|-1) for selected cycles
    # Equivalent to minimizing (num_edges - sum(|c|)) = minimizing num_pieces
    
    best_solution = []
    best_score = -1

    # Initialize with a greedy solution
    def _greedy_init():
        used = set()
        sel = []
        # Sort by length descending, with random tie-break
        order = list(range(m))
        keys = [(-cyc_lens[i], rng.random()) for i in range(m)]
        order.sort(key=lambda i: keys[i])
        for i in order:
            es = cyc_edges[i]
            if any(e in used for e in es):
                continue
            for e in es:
                used.add(e)
            sel.append(i)
        return sel

    # Evaluate score of a solution
    def _score(sel):
        s = 0
        for i in sel:
            s += cyc_lens[i] - 1
        return s

    # Check validity (edge-disjoint)
    def _is_valid(sel):
        used = set()
        for i in sel:
            for e in cyc_edges[i]:
                if e in used:
                    return False
                used.add(e)
        return True

    # Simulated Annealing
    current = _greedy_init()
    current_score = _score(current)
    best_solution = current[:]
    best_score = current_score

    T = 10.0
    cooling = 0.999
    min_T = 0.01

    while T > min_T and time.time() - start_time < time_limit:
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random unselected cycle
            unselected = [i for i in range(m) if i not in set(current)]
            if not unselected:
                continue
            cand = rng.choice(unselected)
            # Check if it can be added (no conflict)
            es = cyc_edges[cand]
            used = set()
            for i in current:
                for e in cyc_edges[i]:
                    used.add(e)
            if any(e in used for e in es):
                continue
            new_sel = current + [cand]
            new_score = current_score + (cyc_lens[cand] - 1)
            
        elif move_type < 0.7:
            # Remove a random selected cycle
            if not current:
                continue
            cand = rng.choice(current)
            new_sel = [i for i in current if i != cand]
            new_score = current_score - (cyc_lens[cand] - 1)
            
        else:
            # Swap: remove one, add one
            if not current:
                continue
            remove_cand = rng.choice(current)
            unselected = [i for i in range(m) if i not in set(current)]
            if not unselected:
                continue
            add_cand = rng.choice(unselected)
            
            # Check if add_cand conflicts with remaining selected cycles
            remaining = [i for i in current if i != remove_cand]
            used = set()
            for i in remaining:
                for e in cyc_edges[i]:
                    used.add(e)
            es = cyc_edges[add_cand]
            if any(e in used for e in es):
                continue
            
            new_sel = remaining + [add_cand]
            new_score = current_score - (cyc_lens[remove_cand] - 1) + (cyc_lens[add_cand] - 1)
        
        # Accept if better, or with probability exp((new_score - current_score) / T)
        if new_score >= current_score:
            current = new_sel
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_solution = current[:]
        else:
            delta = new_score - current_score
            prob = pow(2.718281828, delta / T)
            if rng.random() < prob:
                current = new_sel
                current_score = new_score
        
        T *= cooling

    return [pool[i] for i in best_solution]


def _solve_one(edges, n, deadline):
    if not edges:
        return []

    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)

    # time budget split
    t0 = time.time()
    pool_time = min(8.0, max(1.0, (deadline - t0) * 0.5))
    sa_time = min(10.0, max(1.0, (deadline - t0) * 0.5))

    rng = random.Random(777)
    pool = _generate_cycle_pool(adj, n, edge_set, rng, pool_time)

    selected = _solve_sa(pool, edge_set, n, rng, sa_time)

    # Build pieces: selected cycles + leftover edges as singletons
    used = set()
    for cyc in selected:
        L = len(cyc)
        for k in range(L):
            a = cyc[k]
            b = cyc[(k + 1) % L]
            used.add(tuple(sorted((a, b))))

    pieces = [list(c) for c in selected]
    for e in edge_set:
        if e not in used:
            pieces.append(list(e))

    return pieces


def construct():
    start_time = time.time()
    total_budget = 145.0
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # give each graph a slice; ensure last finishes before budget
        remaining = total_budget - (time.time() - start_time)
        per = remaining / (len(specs) - i)
        graph_deadline = time.time() + per
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0

        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)

        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break

    return results
