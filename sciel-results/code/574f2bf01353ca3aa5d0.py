"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        
        if not P:
            return
        
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        pivot = max(P | X, key=lambda u: len(P & adj[u]))
        candidates = [v for v in P if v not in adj[pivot]]
        
        for v in candidates:
            bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _greedy_set_cover_bitmask(cliques, n, deadline):
    """Bitmask-accelerated greedy set cover with lookahead."""
    if not cliques:
        return []
    
    m = len(cliques)
    # Build bitmask for each vertex: which cliques it covers
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    full_mask = (1 << m) - 1
    remaining = full_mask
    solution = []
    
    while remaining and time.time() < deadline:
        best_v = -1
        best_score = -1
        for v in range(n):
            if vertex_mask[v] == 0:
                continue
            count = bin(remaining & vertex_mask[v]).count('1')
            if count == 0:
                continue
            # Lookahead: prefer vertices that cover cliques with few other candidates
            bonus = 0
            for i in range(m):
                if (remaining >> i) & 1:
                    # Count how many vertices in this clique are still available
                    clique = cliques[i]
                    avail = sum(1 for u in clique if u not in solution)
                    if avail <= 2:
                        bonus += 10
            score = count * 100 + bonus
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v < 0:
            break
        
        solution.append(best_v)
        remaining &= ~vertex_mask[best_v]
    
    return sorted(solution)

def _local_search_bitmask(solution, cliques, n, deadline):
    """Local search with bitmask: removals, 1-1 swaps, and 2-1 swaps."""
    if not solution:
        return solution
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    # Precompute vertex masks
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    def covered_mask(sol):
        c = 0
        for v in sol:
            c |= vertex_mask[v]
        return c
    
    best = list(solution)
    best_cm = covered_mask(best)
    best_set = set(best)
    
    for iteration in range(5000):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Try removing each vertex
        for idx, v in enumerate(best):
            new_cm = 0
            for j, u in enumerate(best):
                if j != idx:
                    new_cm |= vertex_mask[u]
            if new_cm == full_mask:
                best.pop(idx)
                best_cm = new_cm
                best_set.discard(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-1 swap: remove v, add w
        found_swap = False
        for idx, v in enumerate(best):
            rest_cm = 0
            for j, u in enumerate(best):
                if j != idx:
                    rest_cm |= vertex_mask[u]
            missing = full_mask & ~rest_cm
            if missing == 0:
                continue
            for w in range(n):
                if w in best_set:
                    continue
                if (rest_cm | vertex_mask[w]) == full_mask:
                    best[idx] = w
                    best_cm = rest_cm | vertex_mask[w]
                    best_set.discard(v)
                    best_set.add(w)
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if improved:
            continue
        
        # Try 2-1 swap: remove v1, v2, add w
        found_21 = False
        for i1 in range(len(best)):
            for i2 in range(i1 + 1, len(best)):
                rest_cm = 0
                for j, u in enumerate(best):
                    if j != i1 and j != i2:
                        rest_cm |= vertex_mask[u]
                missing = full_mask & ~rest_cm
                if missing == 0:
                    # Both redundant
                    new_best = [u for j, u in enumerate(best) if j != i1 and j != i2]
                    best = new_best
                    best_cm = rest_cm
                    best_set = set(best)
                    improved = True
                    found_21 = True
                    break
                for w in range(n):
                    if w in best_set:
                        continue
                    if (rest_cm | vertex_mask[w]) == full_mask:
                        new_best = [u for j, u in enumerate(best) if j != i1 and j != i2]
                        new_best.append(w)
                        best = new_best
                        best_cm = rest_cm | vertex_mask[w]
                        best_set = set(best)
                        improved = True
                        found_21 = True
                        break
                if found_21:
                    break
            if found_21:
                break
        
        if not improved:
            break
    
    return sorted(best)

def _shrink_polish(solution, cliques, n, deadline, rng):
    """Try multiple random removal orders to escape local minima."""
    if not solution:
        return solution
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    def covered(sol):
        c = 0
        for v in sol:
            c |= vertex_mask[v]
        return c
    
    best = list(solution)
    best_len = len(best)
    
    # Multiple random removal passes
    for trial in range(50):
        if time.time() >= deadline:
            break
        
        cur = list(solution)
        # Shuffle order for removal attempts
        order = list(range(len(cur)))
        rng.shuffle(order)
        
        for idx in order:
            if idx >= len(cur):
                continue
            v = cur[idx]
            new_sol = [x for i, x in enumerate(cur) if i != idx]
            if covered(new_sol) == full_mask:
                cur = new_sol
        
        if len(cur) < best_len:
            best = cur
            best_len = len(cur)
    
    return sorted(best)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time proportionally to graph size
    total_n = sum(n for n, p, s in instances)
    time_alloc = {}
    remaining_time = deadline - start_time - 5  # Reserve 5s buffer
    
    for n, p, seed in instances:
        time_alloc[(n, p, seed)] = remaining_time * (n / total_n)
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
            # Not enough time left; return empty for remaining
            results.append([])
            continue
        
        inst_deadline = start_time + time_alloc[(n, p, seed)]
        # Ensure we don't exceed global deadline
        inst_deadline = min(inst_deadline, deadline - 3)
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy set cover with bitmask
        sol = _greedy_set_cover_bitmask(cliques, n, inst_deadline - 2)
        
        # Local search with bitmask (includes 2-1 swaps)
        sol = _local_search_bitmask(sol, cliques, n, inst_deadline - 1)
        
        # Shrink polish: multiple random removal orders + SA
        rng = random.Random(seed + 999)
        sol = _shrink_polish(sol, cliques, n, inst_deadline, rng)
        
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
