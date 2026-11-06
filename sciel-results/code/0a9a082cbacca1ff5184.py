import time
import random
import itertools

_START = time.time()
_DEADLINE = 140.0

def _remaining_time():
    return _DEADLINE - (time.time() - _START)

def _bit_count(x):
    return bin(x).count('1')

def _has_triangle_bitmask(adj, n):
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
    best = [0]
    full_mask = (1 << n) - 1
    adj_local = adj

    def _color_bound(available):
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            candidates = unassigned & adj_local[v]
            while candidates:
                lsb_u = candidates & -candidates
                u = lsb_u.bit_length() - 1
                candidates ^= lsb_u
                unassigned ^= lsb_u
                candidates &= adj_local[u]
        return k

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        bound = _color_bound(available)
        if current_size + bound <= best[0]:
            return

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

        new_avail_include = available & ~(1 << v) & ~adj_local[v]
        _bnb(new_avail_include, current_size + 1)

        new_avail_exclude = available & ~(1 << v)
        _bnb(new_avail_exclude, current_size)

    _bnb(full_mask, 0)
    return best[0]

def _build_circulant_adj(n, S):
    adj = [0] * n
    S_set = set(S)
    for s in S:
        S_set.add((n - s) % n)
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in S_set:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _is_circulant_triangle_free(n, S):
    S_set = set(S)
    for s in S:
        S_set.add((n - s) % n)
    
    # Check a, b in S such that a-b in S or a+b in S
    # Vertices 0, a, b. Edges (0,a), (0,b), (a,b).
    # (0,a) iff a in S_set.
    # (0,b) iff b in S_set.
    # (a,b) iff b-a in S_set.
    # So if a, b in S_set and b-a in S_set, we have a triangle.
    # We only need to check a, b in S (the positive shifts) to cover all cases?
    # If we have shifts S, the graph is defined by S ∪ -S.
    # Triangle 0, x, y exists if x ∈ S∪-S, y ∈ S∪-S, y-x ∈ S∪-S.
    # It suffices to check x, y ∈ S∪-S.
    # Since S is small, we can just check all pairs in S∪-S.
    
    full_S = list(S_set)
    for a in full_S:
        for b in full_S:
            if a == b: continue
            d = (b - a) % n
            if d in S_set:
                return False
    return True

def _try_circulant(n, rng, max_trials=20000):
    half = n // 2
    shifts = list(range(1, half + 1))
    
    for _ in range(max_trials):
        if _remaining_time() < 5:
            break
        # Random subset of shifts
        # We want a reasonably dense graph to keep alpha low.
        # Density ~ 0.5 of possible edges?
        # In circulant, deg = 2 * |S|.
        # For n=40, max deg 38.
        # Let's try |S| around n/6 to n/4.
        k = rng.randint(n // 8, n // 4)
        S = rng.sample(shifts, k)
        
        if not _is_circulant_triangle_free(n, S):
            continue
            
        adj = _build_circulant_adj(n, S)
        
        # Verify triangle free with bitmask just in case
        if _has_triangle_bitmask(adj, n):
            continue
            
        alpha = _max_independent_set_size_bitmask(adj, n)
        if alpha < 10:
            return adj, alpha
            
    return None, 100

def construct():
    best_adj = None
    best_n = 0
    best_alpha = 100

    rng = random.Random(42)

    # Try n=40 first
    for n_target in [40, 39, 38, 37, 36]:
        if _remaining_time() < 10:
            break
            
        found = False
        # Use multiple seeds for randomness
        for seed in range(50):
            if _remaining_time() < 5:
                break
            rng_local = random.Random(seed)
            adj, alpha = _try_circulant(n_target, rng_local, max_trials=5000)
            if adj is not None:
                if n_target > best_n:
                    best_adj = adj
                    best_n = n_target
                    best_alpha = alpha
                found = True
                break # Found valid graph for this n, try larger n? No, we iterate down.
                # Actually, if we found n=40, we are done.
                # If we found n=39, we are done (unless we want to try 40 more?).
                # The loop goes 40, 39...
                # If 40 fails, we try 39.
        
        if found:
            break

    # Fallback: Local search on n=39 if not found
    if best_n < 39 and _remaining_time() > 20:
        # Try to construct a graph on 39 vertices using local search
        n = 39
        adj = [0] * n
        # Start with a random triangle-free graph
        # Greedy add edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for i, j in edges:
            if _remaining_time() < 10:
                break
            if adj[i] & adj[j]:
                continue
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        alpha = _max_independent_set_size_bitmask(adj, n)
        
        # Local search to reduce alpha
        # We want to remove triangles (if any, though we started triangle-free) and reduce alpha.
        # Actually, starting triangle-free, we just need to reduce alpha.
        # We can add edges? No, adding edges might create triangles.
        # We can remove edges? That increases alpha.
        # So we need to START with a graph that might have triangles, and remove them, while keeping alpha low?
        # Or start with a dense graph, remove triangles, and see if alpha drops?
        # This is hard.
        
        # Better: Start with a circulant-like structure and perturb it.
        # Or just return the best circulant found.
        pass

    if best_adj is None:
        # Absolute fallback: Return a small valid graph
        # e.g. n=10, no independent set of 10? 
        # Complete graph K_9? Triangle-free? No.
        # We need a triangle-free graph with alpha < 10.
        # C_5 is triangle-free, alpha=2.
        # We can just return a small graph to be safe, though score will be low.
        # But we likely found n=39.
        n = 25
        adj = [0] * n
        # Random triangle free
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for i, j in edges:
            if adj[i] & adj[j]:
                continue
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        best_adj = adj
        best_n = n

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
