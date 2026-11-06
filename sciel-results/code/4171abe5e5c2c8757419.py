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

def find_k33(adj_bitmasks):
    """Return (i, j, k, mask) where mask has >=3 bits (a K3,3 witness), or None."""
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            cij = ai & adj_bitmasks[j]
            if cij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                c = cij & adj_bitmasks[k]
                if c.bit_count() >= 3:
                    return (i, j, k, c)
    return None

def is_add_valid_fast(adj_bitmasks, u, v):
    """
    Fast check: adding edge (u,v) keeps graph K3,3-free.
    Only triples containing u or v can become invalid.
    """
    n = len(adj_bitmasks)
    new_u = adj_bitmasks[u] | (1 << v)
    new_v = adj_bitmasks[v] | (1 << u)

    # Triples {u, i, j}: need i,j both neighbors of u (in new_u), and |N(i)∩N(j)∩N(u)|>=3
    # N(u) gains v, so the common set with new_u = (old common) ∪ possibly v if v in N(i)∩N(j)
    # We need: for each pair (i,j) with i,j in new_u, i!=u, j!=u, i<j:
    #   (new_u & N(i) & N(j)).bit_count() < 3
    # Since new_u = old_u | {v}, this equals (old_u & N(i) & N(j)) plus possibly v.
    # v is in N(i)∩N(j) iff both i~v and j~v.
    # So count = (old_u & N(i) & N(j)).bit_count() + (1 if (v in N(i) and v in N(j)) else 0)
    # But we can just compute directly with bitmasks.
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

    def greedy_add_exhaustive(bitmasks, time_limit=None):
        """Add any valid non-edge repeatedly until none remain (exhaustive)."""
        if time_limit is None:
            time_limit = deadline
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks):
        """Local search: greedy add + targeted swap moves to escape local optima."""
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
                    if time.time() > deadline:
                        break
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]

            if not moved:
                # Targeted 1-for-1 swap: find a K3,3 witness, remove an edge from it,
                # then try to add a non-edge that was blocked.
                witness = find_k33(bitmasks)
                if witness is None:
                    # Already K3,3-free (shouldn't happen if we maintain validity)
                    break

                i, j, k, mask = witness
                # The K3,3 has parts {i,j,k} and 3 neighbors in mask.
                # Remove one of the 9 edges incident to this K3,3.
                # Candidate edges to remove: (i,x), (j,x), (k,x) for x in mask.
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb

                remove_candidates = []
                for part in (i, j, k):
                    for x in mask_bits:
                        if bitmasks[part] & (1 << x):
                            remove_candidates.append((part, x))

                if remove_candidates:
                    random.shuffle(remove_candidates)
                    non_edges = get_non_edges(bitmasks)
                    random.shuffle(non_edges)

                    for a, b in remove_candidates:
                        if time.time() > deadline:
                            break
                        # Try removing (a,b) and adding various non-edges
                        for c, d in non_edges[:20]:
                            if time.time() > deadline:
                                break
                            bm2 = bitmasks[:]
                            bm2[a] &= ~(1 << b)
                            bm2[b] &= ~(1 << a)
                            if is_add_valid(bm2, c, d):
                                bm2[c] |= (1 << d)
                                bm2[d] |= (1 << c)
                                # Verify full validity
                                if has_k33_fast(bm2):
                                    bitmasks = bm2
                                    current_edges = count_edges(bitmasks)
                                    moved = True
                                    if current_edges > best_local_edges:
                                        best_local_edges = current_edges
                                        best_local_bitmasks = bitmasks[:]
                                    break
                        if moved:
                            break

            if not moved:
                # Random 1-for-1 swap as fallback
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
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
                no_improve += 1
                if no_improve > 80:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 40
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break

        if r == 0:
            # Start empty, greedy add
            init_bitmasks = [0] * n
        elif r == 1:
            # Start with a random moderate graph, repair, then greedy
            init_bitmasks = [0] * n
            p = 0.35
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid using targeted removal
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 5000:
                witness = find_k33(init_bitmasks)
                if witness is None:
                    break
                i, j, k, mask = witness
                # Remove a random edge from this K3,3
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb
                part = random.choice((i, j, k))
                x = random.choice(mask_bits)
                init_bitmasks[part] &= ~(1 << x)
                init_bitmasks[x] &= ~(1 << part)
                attempts += 1
        else:
            # Random sparse start
            init_bitmasks = [0] * n
            p = 0.15 + 0.015 * (r % 10)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            # Repair until valid using targeted removal
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 5000:
                witness = find_k33(init_bitmasks)
                if witness is None:
                    break
                i, j, k, mask = witness
                mask_bits = []
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    mask_bits.append(x)
                    m -= lsb
                part = random.choice((i, j, k))
                x = random.choice(mask_bits)
                init_bitmasks[part] &= ~(1 << x)
                init_bitmasks[x] &= ~(1 << part)
                attempts += 1

        if not has_k33_fast(init_bitmasks):
            continue

        local_best_bitmasks, local_best_edges = local_search(init_bitmasks)

        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks[:]

    # Final: one more greedy pass on the best found
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:], time_limit=deadline)
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
