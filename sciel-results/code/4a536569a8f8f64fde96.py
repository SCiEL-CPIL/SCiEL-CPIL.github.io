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
    Key insight: only triples where u or v is the "new" common neighbor can become invalid.
    For triple {u, i, j}: the new common neighbor is v (if v is neighbor of both i and j).
    So we only need to check if v becomes a 3rd common neighbor of some pair.
    """
    n = len(adj_bitmasks)
    old_u = adj_bitmasks[u]
    old_v = adj_bitmasks[v]
    
    # Adding (u,v): N(u) gains v, N(v) gains u.
    # For triple {u, i, j}: common = N(u)∩N(i)∩N(j). Old common was valid (<3).
    # New common = old_common ∪ {v if v∈N(i)∩N(j)}. So invalid iff v∈N(i)∩N(j) AND old_common had exactly 2.
    # Actually simpler: just check if v is in N(i)∩N(j) for any i,j where |N(u)∩N(i)∩N(j)| == 2.
    # Even simpler: check all pairs (i,j) where v is in N(i)∩N(j), and see if |N(u)∩N(i)∩N(j)∪{v}| >= 3.
    # That equals (N(u)∩N(i)∩N(j)).bit_count() + 1 if v not in N(u) (which it isn't since u-v not edge yet).
    # Wait, v is not in N(u) before adding. So new common = (N(u)∩N(i)∩N(j)) ∪ {v} if v∈N(i)∩N(j).
    # Count = (N(u)∩N(i)∩N(j)).bit_count() + 1 (since v∉N(u)).
    # Invalid iff (N(u)∩N(i)∩N(j)).bit_count() >= 2 AND v∈N(i)∩N(j).
    
    # Similarly for triple {v, i, j}: invalid iff (N(v)∩N(i)∩N(j)).bit_count() >= 2 AND u∈N(i)∩N(j).
    
    # For triple {i, j, k} where none is u or v: unchanged, still valid.
    # For triple {u, v, i}: common = N(u)∩N(v)∩N(i). Old: (N(u)\{v})∩(N(v)\{u})∩N(i) = N(u)∩N(v)∩N(i) (since u∉N(v), v∉N(u) before).
    # New: (N(u)∪{v})∩(N(v)∪{u})∩N(i) = N(u)∩N(v)∩N(i) ∪ {v if v∈N(i)} ∪ {u if u∈N(i)}.
    # This could gain up to 2 elements. Need to check.
    
    # Let's just do the direct but optimized check:
    new_u = old_u | (1 << v)
    new_v = old_v | (1 << u)
    
    # Triples {u, i, j}: only need to check if the count increases to >=3.
    # Since old was <3, new can be >=3 only if it was 2 and v is added, or was 3 (impossible).
    # So: check pairs (i,j) with v in N(i)∩N(j), and (N(u)∩N(i)∩N(j)).bit_count() >= 2.
    # But also need i,j not equal to u, and i≠j.
    
    # Optimized: iterate over common neighbors of v (in old_v), check pairs among them.
    # v's neighbors in old_v are candidates for i,j where v∈N(i)∩N(j) means i,j are neighbors of v.
    # Wait no: v∈N(i) means i is neighbor of v, i.e., i∈N(v)=old_v.
    
    # So for triple {u,i,j}: need i,j ∈ old_v (neighbors of v), and (N(u)∩N(i)∩N(j)).bit_count() >= 2.
    # Also i≠u, j≠u, i≠j.
    
    # Similarly for {v,i,j}: need i,j ∈ old_u, and (N(v)∩N(i)∩N(j)).bit_count() >= 2.
    
    # And for {u,v,i}: need (N(u)∩N(v)∩N(i)).bit_count() + (1 if i∈old_v else 0) + (1 if i∈old_u else 0) >= 3.
    # Wait: new_u = old_u|{v}, new_v = old_v|{u}.
    # common = new_u ∩ new_v ∩ N(i) = (old_u∪{v}) ∩ (old_v∪{u}) ∩ N(i)
    # = (old_u∩old_v∩N(i)) ∪ ({v}∩old_v∩N(i)) ∪ (old_u∩{u}∩N(i)) ∪ ({v}∩{u}∩N(i))
    # v∉old_v? No, v is not its own neighbor. u∉old_u. So:
    # = (old_u∩old_v∩N(i)) ∪ ({v}∩N(i)) ∪ ({u}∩N(i))
    # v∈N(i) iff i∈old_v. u∈N(i) iff i∈old_u.
    # So count = (old_u∩old_v∩N(i)).bit_count() + (1 if i∈old_v) + (1 if i∈old_u)
    # But v∉old_u and u∉old_v, so {v}∩old_u=∅, {u}∩old_v=∅. The sets are disjoint from old_u∩old_v.
    # Actually v∉old_u (u-v not edge), u∉old_v. So {v} and {u} are not in old_u∩old_v.
    # So count = (old_u∩old_v∩N(i)).bit_count() + (1 if i∈old_v) + (1 if i∈old_u)
    # Invalid if this >= 3.
    
    # This is getting complex. Let me just use the bitmask approach but with early termination.
    
    for i in range(n):
        if i == u or i == v:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u or j == v:
                continue
            common = new_u & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False
    
    for i in range(n):
        if i == u or i == v:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u or j == v:
                continue
            common = new_v & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False
    
    # Also check triples {u, v, i}
    for i in range(n):
        if i == u or i == v:
            continue
        common = new_u & new_v & adj_bitmasks[i]
        if common.bit_count() >= 3:
            return False
    
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Only triples containing u or v can become invalid.
    Uses the optimized check from is_add_valid_fast.
    """
    return is_add_valid_fast(adj_bitmasks, u, v)

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
                if no_improve > 40:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 80
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

        # Quick greedy pass first to get a baseline
        quick_bm = init_bitmasks[:]
        quick_bm = greedy_add_exhaustive(quick_bm, time_limit=min(deadline - 5, time.time() + 2))
        quick_edges = count_edges(quick_bm)
        
        if quick_edges >= best_edges:
            # Worth doing full local search
            local_best_bitmasks, local_best_edges = local_search(init_bitmasks)
        else:
            # Skip full local search if quick greedy doesn't beat best
            local_best_bitmasks, local_best_edges = quick_bm, quick_edges

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
