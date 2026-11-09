import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 145.0

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []

    # Build bitmask adjacency
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    initial_edge_set = set(edges)
    best_partition = None
    best_count = float('inf')

    def _count_common(u, v, cur_mask_list):
        """Count common neighbors of u,v in current graph using bitmasks."""
        common = cur_mask_list[u] & cur_mask_list[v]
        common &= ~(1 << u) & ~(1 << v)
        return bin(common).count('1')

    def _expand_clique(start_u, start_v, cur_mask_list, n, rng, top_k=3):
        """Expand edge (start_u, start_v) to a maximal clique using top-K strategy."""
        clique = [start_u, start_v]

        # Find candidates: vertices connected to both start_u and start_v in current graph
        common = cur_mask_list[start_u] & cur_mask_list[start_v]
        common &= ~(1 << start_u) & ~(1 << start_v)

        candidates = []
        m = common
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            candidates.append(w)
            m ^= lsb

        if not candidates:
            return clique

        # Build adjacency among candidates using cur_mask_list
        while candidates:
            if not _deadline_ok():
                break

            # For each candidate, compute degree within candidates
            # Use top-K strategy: try top candidates by degree, pick best expansion
            cand_set = set(candidates)
            degs = {}
            for w in candidates:
                d = 0
                m2 = cur_mask_list[w]
                # Count neighbors of w that are in candidates
                for x in candidates:
                    if x != w and (m2 >> x) & 1:
                        d += 1
                degs[w] = d

            # Sort candidates by degree descending
            sorted_cands = sorted(candidates, key=lambda w: degs[w], reverse=True)

            # Try top-K starting vertices, pick the one yielding largest clique
            best_w = None
            best_clique_size = -1
            best_remaining = None

            k = min(top_k, len(sorted_cands))
            for i in range(k):
                w = sorted_cands[i]
                # Simulate adding w: new candidates must be connected to w
                new_cands = []
                for x in candidates:
                    if x != w and (cur_mask_list[w] >> x) & 1:
                        new_cands.append(x)
                # The clique size would be len(clique) + 1 + (max further expansion)
                # For simplicity, estimate by |new_cands| + 1
                est = 1 + len(new_cands)
                if est > best_clique_size:
                    best_clique_size = est
                    best_w = w
                    best_remaining = new_cands

            if best_w is None:
                break

            clique.append(best_w)
            candidates = best_remaining

        return clique

    def _expand_clique_simple(start_u, start_v, cur_mask_list, n, rng):
        """Simpler greedy expansion for speed."""
        clique = [start_u, start_v]
        common = cur_mask_list[start_u] & cur_mask_list[start_v]
        common &= ~(1 << start_u) & ~(1 << start_v)

        candidates = []
        m = common
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            candidates.append(w)
            m ^= lsb

        while candidates:
            if not _deadline_ok():
                break

            # Pick candidate with max degree in candidate subgraph
            best_w = None
            best_deg = -1
            for w in candidates:
                d = 0
                mw = cur_mask_list[w]
                for x in candidates:
                    if x != w and (mw >> x) & 1:
                        d += 1
                if d > best_deg:
                    best_deg = d
                    best_w = w

            if best_w is None:
                break

            clique.append(best_w)
            # Filter
            new_cands = []
            for x in candidates:
                if x != best_w and (cur_mask_list[best_w] >> x) & 1:
                    new_cands.append(x)
            candidates = new_cands

        return clique

    def _greedy_partition(cur_mask_list, n, rng, strategy):
        """Run one greedy pass. Returns (partition, cur_mask_list)."""
        cur = [m for m in cur_mask_list]
        partition = []

        # Build edge set from cur masks for iteration
        while True:
            if not _deadline_ok():
                break

            # Find all edges in current graph
            current_edges = []
            for u in range(n):
                m = cur[u] & ~((1 << (u + 1)) - 1)  # only v > u
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    current_edges.append((u, v))
                    m ^= lsb

            if not current_edges:
                break

            if strategy == 0:
                # Min common neighbors
                min_c = float('inf')
                cands = []
                for u, v in current_edges:
                    c = _count_common(u, v, cur)
                    if c < min_c:
                        min_c = c
                        cands = [(u, v)]
                    elif c == min_c:
                        cands.append((u, v))
            elif strategy == 1:
                # Max common neighbors
                max_c = -1
                cands = []
                for u, v in current_edges:
                    c = _count_common(u, v, cur)
                    if c > max_c:
                        max_c = c
                        cands = [(u, v)]
                    elif c == max_c:
                        cands.append((u, v))
            elif strategy == 2:
                # Clique density: ratio of common neighbors to degree
                # Higher density = more likely to form large clique
                best_density = -1
                cands = []
                for u, v in current_edges:
                    c = _count_common(u, v, cur)
                    deg_u = bin(cur[u]).count('1')
                    deg_v = bin(cur[v]).count('1')
                    # Density: common / (deg_u + deg_v - common)
                    denom = deg_u + deg_v - c
                    density = c / denom if denom > 0 else 0
                    if density > best_density + 1e-9:
                        best_density = density
                        cands = [(u, v)]
                    elif abs(density - best_density) < 1e-9:
                        cands.append((u, v))
            else:
                cands = current_edges

            if not cands:
                break

            u, v = rng.choice(cands)

            if strategy == 3:
                clique = _expand_clique(u, v, cur, n, rng, top_k=3)
            else:
                clique = _expand_clique_simple(u, v, cur, n, rng)

            partition.append(clique)

            # Remove clique edges from cur
            for a, b in combinations(clique, 2):
                cur[a] &= ~(1 << b)
                cur[b] &= ~(1 << a)

        # Handle remaining edges as K2
        for u in range(n):
            m = cur[u] & ~((1 << (u + 1)) - 1)
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                partition.append([u, v])
                cur[u] &= ~(1 << v)
                cur[v] &= ~(1 << u)
                m ^= lsb

        return partition

    def _local_improvement(partition, cur_mask_list, n, rng):
        """Try to merge adjacent cliques to reduce count."""
        if not partition:
            return partition

        # Build edge-to-clique mapping
        edge_to_clique = {}
        for idx, clique in enumerate(partition):
            for a, b in combinations(clique, 2):
                key = (min(a, b), max(a, b))
                edge_to_clique[key] = idx

        improved = True
        while improved:
            if not _deadline_ok():
                break
            improved = False

            # Try merging pairs of cliques
            for i in range(len(partition)):
                if not _deadline_ok():
                    break
                for j in range(i + 1, len(partition)):
                    # Check if union is a clique
                    clique_i = set(partition[i])
                    clique_j = set(partition[j])
                    merged = clique_i | clique_j

                    # Check all pairs in merged are edges
                    is_clique = True
                    for a, b in combinations(merged, 2):
                        if not ((cur_mask_list[a] >> b) & 1):
                            is_clique = False
                            break

                    if is_clique:
                        # Merge
                        new_clique = list(merged)
                        partition[i] = new_clique
                        partition.pop(j)
                        improved = True
                        break
                if improved:
                    break

        return partition

    # Run multiple strategies
    for run_seed in range(60):
        if not _deadline_ok():
            break

        rng = random.Random(seed * 1000 + run_seed)
        strategy = run_seed % 4

        partition = _greedy_partition(adj_mask, n, rng, strategy)

        # Local improvement: try to merge cliques
        if _deadline_ok():
            partition = _local_improvement(partition, adj_mask, n, rng)

        # Validate: every edge covered exactly once, every clique valid
        valid = True
        seen_edges = set()
        for clique in partition:
            if len(clique) < 2:
                valid = False
                break
            for a, b in combinations(clique, 2):
                key = (min(a, b), max(a, b))
                if key in seen_edges:
                    valid = False
                    break
                seen_edges.add(key)
                # Check it's actually an edge
                if not ((adj_mask[a] >> b) & 1):
                    valid = False
                    break
            if not valid:
                break

        if valid and seen_edges == initial_edge_set:
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        elif valid:
            # Even if not all edges covered, still track (shouldn't happen)
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition

    if best_partition is None:
        return [[u, v] for u, v in edges]

    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
