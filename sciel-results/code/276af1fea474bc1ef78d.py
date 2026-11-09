"""Hexagon-free graph construction on 40 vertices - SA + bitmask C6 check."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 40
EDGE_BIT = 1 << 63  # sentinel to mark edge in masks (bit 63 is free since vertices 0..39 use bits 0..39)


def _masks_from_edges(edges):
    """Build adjacency bitmask array from edge list."""
    masks = [0] * N
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks


def _edges_from_masks(masks):
    """Extract edge list from masks (only u < v to avoid duplicates, exclude sentinel)."""
    edges = []
    for u in range(N):
        m = masks[u] & ((1 << N) - 1)  # mask off sentinel bit
        for v in range(u + 1, N):
            if m & (1 << v):
                edges.append([u, v])
    return edges


def _count_c6_through_edge(masks, u, v):
    """
    Count the number of C6s that pass through edge (u,v).
    A C6 through (u,v) has form: u - a - b - c - d - v - u (or reverse).
    So we need paths of length 4 from u to v (u-a-b-c-d-v) where a,b,c,d are distinct and not u or v.
    Equivalently: a in N(u), d in N(v), and there's a path a-b-c-d of length 3 with all distinct and not u,v.
    
    More efficient: count 4-edge paths from u to v avoiding u and v internally.
    Path: u - a - b - c - d - v
    a in N(u)\\{v}, d in N(v)\\{u}, b in N(a)\\{u,v,a}, c in N(b)\\{u,v,a,b}, and c adjacent to d, d adjacent to v.
    
    Actually simpler enumeration:
    For each a in N(u) (a != v):
      For each d in N(v) (d != u):
        Count paths a - b - c - d where b,c distinct from u,v,a,d and b!=c, and b in N(a), c in N(b), d in N(c).
    
    Let me use: for a in N(u)\{v}, for d in N(v)\{u}, count common neighbors paths of length 3 from a to d avoiding u,v.
    
    Even simpler: iterate all 4-tuples (a,b,c,d) with a in N(u), d in N(v), b in N(a), c in N(b), c in N(d), all distinct from u,v and each other.
    """
    count = 0
    mask_u = masks[u] & ((1 << N) - 1)
    mask_v = masks[v] & ((1 << N) - 1)
    # a must be in N(u) but not v
    a_mask = mask_u & ~(1 << v)
    # d must be in N(v) but not u
    d_mask = mask_v & ~(1 << u)
    
    # For each a
    a = 0
    while a_mask:
        if a_mask & 1:
            a = 0
            # find position of lowest set bit
            aa = a_mask
            pos = 0
            while not (aa & 1):
                aa >>= 1
                pos += 1
            a = pos
            # Now for each d in d_mask
            dm = d_mask
            while dm:
                if dm & 1:
                    d = 0
                    dd = dm
                    pos2 = 0
                    while not (dd & 1):
                        dd >>= 1
                        pos2 += 1
                    d = pos2
                    if d != a:
                        # Count paths a-b-c-d of length 3, with b,c not in {u,v,a,d}
                        # b in N(a), c in N(b) intersect N(d), c not in {u,v,a,d}, b not in {u,v,a,d}
                        b_mask = masks[a] & ((1 << N) - 1)
                        b_mask &= ~(1 << u)
                        b_mask &= ~(1 << v)
                        b_mask &= ~(1 << a)
                        b_mask &= ~(1 << d)
                        bm = b_mask
                        while bm:
                            if bm & 1:
                                b = 0
                                bb = bm
                                pos3 = 0
                                while not (bb & 1):
                                    bb >>= 1
                                    pos3 += 1
                                b = pos3
                                # c in N(b) intersect N(d), c not in {u,v,a,d,b}
                                c_mask = masks[b] & masks[d] & ((1 << N) - 1)
                                c_mask &= ~(1 << u)
                                c_mask &= ~(1 << v)
                                c_mask &= ~(1 << a)
                                c_mask &= ~(1 << d)
                                c_mask &= ~(1 << b)
                                if c_mask:
                                    # popcount
                                    c_mask &= ((1 << N) - 1)
                                    count += bin(c_mask).count('1')
                dm &= (dm - 1)
        a_mask &= (a_mask - 1)
    return count


def _has_c6_full(masks):
    """Check if graph has any C6. Returns True if C6 exists."""
    for u in range(N):
        for v in range(u + 1, N):
            if masks[u] & (1 << v):
                if _count_c6_through_edge(masks, u, v) > 0:
                    return True
    return False


def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges


def _is_valid_c6_free(masks):
    """Check no C6 exists."""
    for u in range(N):
        for v in range(u + 1, N):
            if masks[u] & (1 << v):
                if _count_c6_through_edge(masks, u, v) > 0:
                    return False
    return True


def _edge_count(masks):
    return sum(bin(m & ((1 << N) - 1)).count('1') for m in masks) // 2


def _simulated_annealing(masks, time_limit, seed):
    """
    SA that works on the space of graphs, with energy = (#C6s * 1000) - (#edges).
    We want to minimize energy. Moves: add edge, remove edge, swap edge.
    """
    random.seed(seed)
    start_time = time.time()
    best_count = _edge_count(masks)
    best_masks = masks[:]
    
    # Start temperature
    T = 50.0
    cooling = 0.98
    
    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    current_count = best_count
    
    # Compute initial C6 count for energy
    # For SA we need to track C6 count incrementally
    # c6_count = total number of C6s in the graph
    # When we add edge (u,v): delta_c6 = _count_c6_through_edge
    # When we remove edge (u,v): delta_c6 = -_count_c6_through_edge (since those C6s were only through this edge? No, a C6 might pass through multiple edges)
    
    # Actually a C6 passes through exactly 6 edges. So counting C6s through each edge and summing gives 6 * (total C6s).
    # For incremental tracking, we need total C6 count.
    # Let's track total_c6.
    
    total_c6 = 0
    for u in range(N):
        for v in range(u + 1, N):
            if masks[u] & (1 << v):
                total_c6 += _count_c6_through_edge(masks, u, v)
    total_c6 //= 6  # each C6 counted 6 times (once per edge)
    
    # Actually _count_c6_through_edge counts C6s that use that specific edge, so sum over all edges = 6 * total_C6
    # But wait - a C6 uses exactly 6 edges, and each edge in the C6 would count it. So yes, sum = 6 * total.
    
    while time.time() - start_time < time_limit:
        # Choose a random move
        move_type = random.random()
        if move_type < 0.4:
            # Add a random non-edge
            u, v = random.choice(all_pairs)
            if masks[u] & (1 << v):
                continue
            delta_c6 = _count_c6_through_edge(masks, u, v)
            delta_e = delta_c6 * 1000 - 1  # energy increase
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                current_count += 1
                total_c6 += delta_c6
                if current_count > best_count and total_c6 == 0:
                    best_count = current_count
                    best_masks = masks[:]
        elif move_type < 0.7:
            # Remove a random edge
            # Pick random edge
            for _ in range(10):
                u, v = random.choice(all_pairs)
                if masks[u] & (1 << v):
                    break
            else:
                continue
            delta_c6 = -_count_c6_through_edge(masks, u, v)
            delta_e = delta_c6 * 1000 + 1  # removing edge: c6 decreases by delta_c6 (negative), edges decrease by 1
            # Actually: removing edge (u,v) removes all C6s through it. delta_c6 = -count_through
            # energy change = (new_c6 - old_c6)*1000 + (new_edges - old_edges) = (-count_through)*1000 + (-1)
            # Wait: new_c6 = total_c6 - count_through, new_edges = current_count - 1
            # delta_energy = (new_c6 - total_c6)*1000 + (new_edges - current_count)
            #             = -count_through*1000 - 1
            delta_e = -_count_c6_through_edge(masks, u, v) * 1000 - 1
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                current_count -= 1
                total_c6 += delta_c6  # delta_c6 is negative
                if current_count > best_count and total_c6 == 0:
                    best_count = current_count
                    best_masks = masks[:]
        else:
            # Swap: remove one edge, add another
            for _ in range(10):
                u1, v1 = random.choice(all_pairs)
                if masks[u1] & (1 << v1):
                    break
            else:
                continue
            u2, v2 = random.choice(all_pairs)
            if masks[u2] & (1 << v2) or (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                continue
            
            # Remove (u1,v1), add (u2,v2)
            # Net edge change: 0
            c6_removed = _count_c6_through_edge(masks, u1, v1)
            # After removing, count C6s through (u2,v2) in the modified graph
            # This is complex; approximate:
            masks[u1] &= ~(1 << v1)
            masks[v1] &= ~(1 << u1)
            c6_added = _count_c6_through_edge(masks, u2, v2)
            delta_e = (c6_added - c6_removed) * 1000
            # Apply or revert
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                total_c6 += (c6_added - c6_removed)
                if current_count > best_count and total_c6 == 0:
                    best_count = current_count
                    best_masks = masks[:]
            else:
                # Revert removal of (u1,v1)
                masks[u1] |= (1 << v1)
                masks[v1] |= (1 << u1)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_masks, best_count


def _greedy_add(masks, time_limit, seed):
    """Greedy add edges in random order, stopping when no more can be added."""
    random.seed(seed)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    start_time = time.time()
    count = _edge_count(masks)
    
    for u, v in all_pairs:
        if time.time() - start_time > time_limit:
            break
        if masks[u] & (1 << v):
            continue
        if _count_c6_through_edge(masks, u, v) == 0:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            count += 1
    
    return masks, count


def _local_search_greedy(masks, time_limit, seed):
    """Repeatedly try to add edges; when stuck, do a swap to unstick."""
    random.seed(seed)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    start_time = time.time()
    count = _edge_count(masks)
    best_count = count
    best_masks = masks[:]
    
    # First, greedily add all possible edges
    progress = True
    while progress and time.time() - start_time < time_limit:
        progress = False
        for u, v in all_pairs:
            if time.time() - start_time > time_limit:
                break
            if masks[u] & (1 << v):
                continue
            if _count_c6_through_edge(masks, u, v) == 0:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                count += 1
                progress = True
        if count > best_count:
            best_count = count
            best_masks = masks[:]
    
    # Now do swaps
    while time.time() - start_time < time_limit * 0.8:
        # Pick a random edge to remove
        for _ in range(20):
            u1, v1 = random.choice(all_pairs)
            if masks[u1] & (1 << v1):
                break
        else:
            break
        
        # Remove it
        masks[u1] &= ~(1 << v1)
        masks[v1] &= ~(1 << u1)
        count -= 1
        
        # Try to add back plus possibly another edge
        improved = False
        for _ in range(50):
            u2, v2 = random.choice(all_pairs)
            if masks[u2] & (1 << v2):
                continue
            if _count_c6_through_edge(masks, u2, v2) == 0:
                masks[u2] |= (1 << v2)
                masks[v2] |= (1 << u2)
                count += 1
                improved = True
                break
        
        if not improved:
            # Revert
            masks[u1] |= (1 << v1)
            masks[v1] |= (1 << u1)
            count += 1
        
        # Try to add more
        for u, v in all_pairs:
            if time.time() - start_time > time_limit * 0.8:
                break
            if masks[u] & (1 << v):
                continue
            if _count_c6_through_edge(masks, u, v) == 0:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                count += 1
        
        if count > best_count:
            best_count = count
            best_masks = masks[:]
    
    return best_masks, best_count


def _blowup_seed():
    """
    Start from a different structure: 5 groups of 8 vertices.
    Each group is a K8 (complete), giving 5 * 28 = 140 edges... too many.
    Let's do: 4 groups of 10, each K10 = 4*45=180... way too many.
    
    Actually 8 K5s = 100 edges is good. Let's try a different structure:
    5 groups of 8, each group is a random 14-edge graph on 8 vertices that's C6-free.
    Or: use a known C6-free base and blow up.
    
    Let me try: 2 groups of 20. Each group is a C6-free graph on 20 vertices.
    A K5-free structure... 
    
    Actually, let's just use 8 K5s as the seed and rely on SA/local search.
    But to be structurally different, let's start from a sparse random graph and grow.
    """
    # Start with a random sparse graph
    random.seed(42)
    edges = []
    # Add a random spanning tree
    order = list(range(N))
    random.shuffle(order)
    for i in range(1, N):
        edges.append([order[i - 1], order[i]])
    # Add some random extra edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    for u, v in all_pairs:
        if [u, v] in edges or [v, u] in edges:
            continue
        if len(edges) >= 60:
            break
        edges.append([u, v])
    return edges


def _verify_no_c6(masks):
    """Final verification."""
    for u in range(N):
        for v in range(u + 1, N):
            if masks[u] & (1 << v):
                if _count_c6_through_edge(masks, u, v) > 0:
                    return False
    return True


def construct():
    deadline = 130  # Safety margin before 150s kill
    start_time = time.time()
    
    best_masks = None
    best_count = 0
    
    # Strategy 1: Greedy from 8-K5 seed with local search
    seed1_edges = _seed_graph_8K5()
    masks1 = _masks_from_edges(seed1_edges)
    
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 30:
        t1 = min(25, remaining * 0.4)
        masks1, count1 = _local_search_greedy(masks1, t1, seed=1)
        if count1 > best_count:
            best_count = count1
            best_masks = masks1[:]
    
    # Strategy 2: SA from 8-K5
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 20:
        t2 = min(40, remaining * 0.45)
        masks2 = _masks_from_edges(seed1_edges)
        masks2, count2 = _simulated_annealing(masks2, t2, seed=7)
        if count2 > best_count:
            best_count = count2
            best_masks = masks2[:]
    
    # Strategy 3: Local search from random sparse seed
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 20:
        sparse_edges = _blowup_seed()
        masks3 = _masks_from_edges(sparse_edges)
        t3 = min(30, remaining * 0.5)
        masks3, count3 = _local_search_greedy(masks3, t3, seed=42)
        if count3 > best_count:
            best_count = count3
            best_masks = masks3[:]
    
    # Strategy 4: Another SA run with different seed
    remaining = deadline - (time.time() - start_time) - 5
    if remaining > 15 and best_masks is not None:
        t4 = min(25, remaining * 0.6)
        masks4 = best_masks[:]
        masks4, count4 = _simulated_annealing(masks4, t4, seed=123)
        if count4 > best_count:
            best_count = count4
            best_masks = masks4[:]
    
    # Final: verify and return
    if best_masks is None:
        # Fallback
        fallback_edges = _seed_graph_8K5()
        return fallback_edges
    
    # Verify
    if not _verify_no_c6(best_masks):
        # Strip edges until valid
        # This shouldn't happen but just in case
        # Try removing random edges until C6-free
        for _ in range(200):
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.seed(999)
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if best_masks[u] & (1 << v):
                    best_masks[u] &= ~(1 << v)
                    best_masks[v] &= ~(1 << u)
                    if _verify_no_c6(best_masks):
                        break
            if _verify_no_c6(best_masks):
                break
        else:
            return _seed_graph_8K5()
    
    edges = _edges_from_masks(best_masks)
    return edges
# EVOLVE-BLOCK-END
