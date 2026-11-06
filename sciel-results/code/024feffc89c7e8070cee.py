import time
import random
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free (no triple has >= 3 common neighbors).
    Returns True if valid.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            common_ij = ai & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Only triples containing u or v can become invalid, since only
    N(u) and N(v) change (each gains the other).
    """
    n = len(adj_bitmasks)
    new_u = adj_bitmasks[u] | (1 << v)
    new_v = adj_bitmasks[v] | (1 << u)

    # Triples {u, i, j} for i<j, i!=u, j!=u
    for i in range(n):
        if i == u:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u:
                continue
            common = new_u & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    # Triples {v, i, j} for i<j, i!=v, j!=v
    for i in range(n):
        if i == v:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == v:
                continue
            common = new_v & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s

    best_edges = 0
    best_adj_bitmasks = None

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            mi = bitmasks[i]
            for j in range(i + 1, n):
                if not (mi & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks):
        """Add any valid non-edge repeatedly until none remain (exhaustive)."""
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks):
        """Local search: greedy add + swap moves to escape local optima."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # First, greedily add valid edges
        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline:
            moved = False

            # Try to add edges: exhaustive pass
            non_edges = get_non_edges(bitmasks)
            if non_edges:
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]

            if not moved:
                # Try swap: remove (a,b), add (c,d)
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(40):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        # Remove (a,b)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try add (c,d)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if has_k33_fast(bm2):
                                bitmasks = bm2
                                current_edges = count_edges(bitmasks)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_bitmasks = bitmasks[:]
                                break

            if not moved:
                # Try 2-edge swap: remove (a,b), add (c,d) and (e,f)
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if len(all_edges) >= 1 and len(non_edges) >= 2:
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        e, f = random.choice(non_edges)
                        # Ensure (c,d) != (e,f)
                        if (c, d) == (e, f) or (c, d) == (f, e):
                            continue
                        # Remove (a,b)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try add (c,d)
                        if not is_add_valid(bm2, c, d):
                            continue
                        bm2[c] |= (1 << d)
                        bm2[d] |= (1 << c)
                        # Try add (e,f)
                        if not is_add_valid(bm2, e, f):
                            continue
                        bm2[e] |= (1 << f)
                        bm2[f] |= (1 << e)
                        if has_k33_fast(bm2):
                            bitmasks = bm2
                            current_edges = count_edges(bitmasks)
                            moved = True
                            if current_edges > best_local_edges:
                                best_local_edges = current_edges
                                best_local_bitmasks = bitmasks[:]
                            break

            if not moved:
                no_improve += 1
                if no_improve > 100:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r == 0:
            # Start empty, greedy add
            init_bitmasks = [0] * n
        elif r == 1:
            # Start with a random moderate graph, repair, then greedy
            init_bitmasks = [0] * n
            p = 0.3
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1
        else:
            # Random sparse start
            init_bitmasks = [0] * n
            p = 0.1 + 0.01 * (r % 10)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1

        if not has_k33_fast(init_bitmasks):
            continue

        local_best_bitmasks, local_best_edges = local_search(init_bitmasks)

        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks[:]

    # Final: one more greedy pass on the best found
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:])
        improved_edges = count_edges(improved)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj_bitmasks = improved

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    # Verify K3,3-free
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid K3,3-free: {valid}")
