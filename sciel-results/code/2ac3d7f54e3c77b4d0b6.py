"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')


def _color_bound(available, adj, n):
    """
    Compute an upper bound on the independence number using greedy coloring
    of the subgraph G[available]. In a triangle-free graph, an independent set
    corresponds to a clique in the complement. We color the available vertices
    such that each color class is an independent set in G[available], meaning
    vertices in the same color class are NOT adjacent in G. The number of colors
    used is an upper bound on the maximum independent set size.
    
    Actually, for independence number: we need to cover vertices with cliques in G.
    Each clique in G contributes at most 1 vertex to an independent set.
    So we greedily build cliques: pick a vertex, add all its neighbors in the remaining set.
    """
    unassigned = available
    k = 0
    while unassigned:
        k += 1
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        unassigned ^= lsb
        # Greedily add neighbors of v that are still unassigned
        candidates = unassigned & adj[v]
        while candidates:
            lsb_u = candidates & -candidates
            u = lsb_u.bit_length() - 1
            candidates ^= lsb_u
            unassigned ^= lsb_u
            candidates &= adj[u]
    return k


def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                if adj[i] & adj[j]:
                    return True
    return False


def _max_independent_set_size_bitmask(adj, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]
    full_mask = (1 << n) - 1
    # Precompute for faster access
    adj_local = adj

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        bound = _color_bound(available, adj_local, n)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj_local[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v
        new_avail_include = available & ~(1 << v) & ~adj_local[v]
        _bnb(new_avail_include, current_size + 1)

        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        _bnb(new_avail_exclude, current_size)

    _bnb(full_mask, 0)
    return best[0]


def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True


def _greedy_triangle_free_bitmask(n, rng):
    """
    Build a triangle-free graph greedily using bitmasks.
    """
    adj = [0] * n
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)

    for i, j in edges:
        if _remaining_time() < 2:
            break
        if adj[i] & adj[j]:
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)

    return adj


def _local_search_bitmask(adj, n, rng, max_iter=2000):
    """
    Improve graph by adding edges to reduce independence number.
    """
    adj = adj[:]
    alpha = _max_independent_set_size_bitmask(adj, n)

    if alpha < 10:
        return adj, alpha

    def _find_one_mis(adj, n):
        """Find one maximum independent set using greedy."""
        remaining = (1 << n) - 1
        selected = []
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    deg = _bit_count(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return selected

    no_improve_count = 0
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break

        target_is = _find_one_mis(adj, n)
        if len(target_is) < 10:
            alpha = _max_independent_set_size_bitmask(adj, n)
            if alpha < 10:
                break

        addable_in_is = []
        for idx_i in range(len(target_is)):
            for idx_j in range(idx_i + 1, len(target_is)):
                i, j = target_is[idx_i], target_is[idx_j]
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        addable_in_is.append((i, j))

        if not addable_in_is:
            no_improve_count += 1
            if no_improve_count > 20:
                break
            continue

        improved = False
        for _ in range(15):
            if _remaining_time() < 2:
                break
            i, j = rng.choice(addable_in_is)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

            new_alpha = _max_independent_set_size_bitmask(adj, n)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                no_improve_count = 0
                if alpha < 10:
                    break
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)

        if alpha < 10:
            break
        if not improved:
            no_improve_count += 1
            if no_improve_count > 30:
                break

    return adj, alpha


def _try_construction_bitmask(n, seed, max_local_iters=500):
    """
    Try to build a valid graph on n vertices.
    """
    rng = random.Random(seed)

    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_bitmask(adj, n)

    if alpha < 10:
        return adj, alpha

    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_bitmask(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls

    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_bitmask(adj2, n)

        if alpha2 < 10:
            return adj2, alpha2

        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_bitmask(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2

    return None


def _paley_graph(n):
    """
    Construct Paley graph of order n where n is a prime congruent to 3 mod 4.
    Vertices are 0..n-1. Edge (i,j) iff (j-i) is a quadratic residue mod n.
    Paley graphs are self-complementary and triangle-free when n ≡ 3 mod 4.
    """
    if n <= 1:
        return [0] * n

    # Compute quadratic residues mod n
    qr = set()
    for x in range(1, n):
        qr.add((x * x) % n)
    qr.discard(0)

    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in qr:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    return adj


def _cyclic_graph(n, shifts):
    """
    Construct a circulant graph with given shifts.
    Edge (i,j) iff (j-i) mod n is in shifts.
    """
    adj = [0] * n
    shift_set = set(shifts)
    for i in range(n):
        for s in shift_set:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            elif j < i:
                # Already handled when j was the smaller index
                pass
    # Ensure symmetry
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in shift_set or (n - d) % n in shift_set:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _try_circulant_construction(n):
    """
    Try to find a circulant graph on n vertices that is triangle-free with alpha < 10.
    Use systematic search over shift sets with greedy filtering.
    """
    if _remaining_time() < 5:
        return None

    rng = random.Random(12345)
    half = n // 2

    # First try: systematic greedy construction of shift set
    # Start with no shifts, add shifts that don't create triangles
    # A triangle in circulant graph with shifts S exists if there exist
    # a, b, c in S such that a+b ≡ c mod n, or a ≡ b+c, etc.
    # More precisely: vertices 0, a, a+b form a triangle if a, b, a+b are all in S ∪ {n-S}
    
    def _shifts_create_triangle(shifts_set):
        """Check if the shift set creates a triangle in the circulant graph."""
        # In a circulant graph, vertices i, i+a, i+a+b form a triangle if
        # a, b, and (a+b) mod n are all in the "distance set"
        # The distance set includes s and n-s for each s in shifts
        dist_set = set()
        for s in shifts_set:
            dist_set.add(s)
            dist_set.add(n - s)
        
        for a in shifts_set:
            for b in shifts_set:
                c = (a + b) % n
                if c != 0 and c in dist_set:
                    # Check: 0, a, a+b - edges are (0,a) if a in dist, (a, a+b) if b in dist, (0, a+b) if (a+b) in dist
                    if a in dist_set and b in dist_set and c in dist_set:
                        return True
                c2 = (a - b) % n
                if c2 != 0 and c2 in dist_set:
                    if a in dist_set and b in dist_set and c2 in dist_set:
                        return True
        return False

    # Greedy: try to build a large shift set without triangles
    best_adj = None
    best_alpha = float('inf')
    
    # Try multiple greedy strategies
    for strategy_seed in range(30):
        if _remaining_time() < 3:
            break
        
        rng_s = random.Random(12345 + strategy_seed)
        available_shifts = list(range(1, half + 1))
        rng_s.shuffle(available_shifts)
        
        current_shifts = set()
        for s in available_shifts:
            trial = current_shifts | {s}
            if not _shifts_create_triangle(trial):
                current_shifts.add(s)
        
        if not current_shifts:
            continue
        
        adj = _cyclic_graph(n, list(current_shifts))
        if _has_triangle_bitmask(adj, n):
            continue
        
        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj
    
    # Randomized search for more diverse shift sets
    for attempt in range(100):
        if _remaining_time() < 3:
            break

        num_shifts = rng.randint(n // 5, n // 3)
        num_shifts = min(num_shifts, half)
        shifts = rng.sample(range(1, half + 1), num_shifts)

        adj = _cyclic_graph(n, shifts)

        if _has_triangle_bitmask(adj, n):
            continue

        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < 10:
            return adj
        if alpha < best_alpha and best_adj is not None:
            best_alpha = alpha
            best_adj = adj

    return best_adj


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')

    # Try circulant constructions first (most promising per cell findings)
    for n_target in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if _remaining_time() < 8:
            break
        if best_n >= n_target:
            continue
        adj = _try_circulant_construction(n_target)
        if adj is not None:
            alpha = _max_independent_set_size_bitmask(adj, n_target)
            if alpha < 10 and n_target > best_n:
                best_adj = adj
                best_n = n_target
                best_alpha = alpha
                break  # Found valid graph at this n, no need to try smaller

    # Try Paley graphs for primes congruent to 3 mod 4
    if best_n < 37:
        paley_primes = [37, 29, 23, 19, 11, 7]
        for p in paley_primes:
            if _remaining_time() < 5:
                break
            if best_n >= p:
                continue
            adj = _paley_graph(p)
            if _has_triangle_bitmask(adj, p):
                continue
            alpha = _max_independent_set_size_bitmask(adj, p)
            if alpha < 10:
                best_adj = adj
                best_n = p
                best_alpha = alpha
                break

    # If we haven't found a good graph, use randomized search
    if best_n < 30:
        for n_target in range(40, 20, -1):
            if _remaining_time() < 8:
                break
            if best_n >= n_target:
                continue

            found = False
            for seed in range(50):
                if _remaining_time() < 3:
                    break

                result = _try_construction_bitmask(n_target, seed)
                if result is not None:
                    adj_res, alpha_res = result
                    if alpha_res < best_alpha:
                        best_adj = adj_res
                        best_n = n_target
                        best_alpha = alpha_res
                    if alpha_res < 10:
                        found = True
                        break

            if found:
                break

    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
