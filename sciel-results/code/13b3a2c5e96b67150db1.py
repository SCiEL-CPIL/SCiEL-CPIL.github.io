"""Combined: bitset conflict graph + branch-and-bound MWIS with matching bound + ILS fallback."""
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    # Perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Diagonal constraint: 2*v must not be a perfect square
    forbidden = [False] * N
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True

    valid_vertices = [i for i in range(N) if not forbidden[i]]
    M = len(valid_vertices)
    vertex_map = valid_vertices  # index -> 0-based vertex id

    # Build adjacency as bitmasks over the M valid vertices
    # adj[i] has bit j set iff vertex_map[i] and vertex_map[j] conflict
    adj = [0] * M
    for i in range(M):
        vi = vertex_map[i] + 1  # 1-based value
        for j in range(i + 1, M):
            vj = vertex_map[j] + 1
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Precompute: for each vertex, the mask of valid vertices it can coexist with
    compatible_with = [0] * M
    all_valid_mask = (1 << M) - 1
    for i in range(M):
        compatible_with[i] = all_valid_mask & ~adj[i] & ~(1 << i)

    # ---------- Bitset-based greedy and local search (fast, for initialization) ----------

    def greedy_min_degree():
        """Greedy: repeatedly pick vertex with fewest conflicts among available."""
        available = all_valid_mask
        selected = 0
        while available:
            best_v = -1
            min_deg = M + 1
            m = available
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected |= (1 << best_v)
            # Remove best_v and all its neighbors from available
            available &= ~((1 << best_v) | (adj[best_v] & available))
        return selected

    def local_add(mask):
        """Add any vertex that doesn't conflict with current set."""
        improved = True
        while improved:
            improved = False
            not_in = all_valid_mask & ~mask
            m = not_in
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def swap_improve(mask, max_rounds=200):
        """1-1 swap local search: remove one, add one (possibly more via local_add)."""
        for _ in range(max_rounds):
            if elapsed() > DEADLINE:
                break
            mask = local_add(mask)
            improved = False
            m = mask
            while m:
                lsb_u = m & -m
                u = lsb_u.bit_length() - 1
                m ^= lsb_u
                mask_wo_u = mask ^ lsb_u
                # Vertices that conflict with u but not with mask_wo_u can be added
                # Actually: v can replace u if (adj[v] & mask_wo_u) == 0 and v not in mask
                # But v must be in adj[u] to be a "swap" (otherwise just add)
                # More generally: any v not in mask with (adj[v] & mask_wo_u) == 0
                potential = all_valid_mask & ~mask
                pm = potential
                while pm:
                    lsb_v = pm & -pm
                    v = lsb_v.bit_length() - 1
                    pm ^= lsb_v
                    if (adj[v] & mask_wo_u) == 0:
                        new_mask = mask_wo_u | lsb_v
                        new_mask = local_add(new_mask)
                        if new_mask.bit_count() > mask.bit_count():
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return mask

    # ---------- Branch and Bound for Maximum Independent Set ----------
    # Using a matching-based upper bound (Lovasz/Edmonds style) for pruning.
    # The bound: for a graph G, the maximum independent set size <=
    #   |V| - (size of maximum matching in G)
    # This is because each matched edge can "cover" at most 2 vertices,
    # and an independent set can take at most 1 from each matched edge.
    # So alpha(G) <= |V| - |maximum matching|.

    def max_matching(mask):
        """Find maximum cardinality matching in the subgraph induced by mask.
        Uses a simple augmenting path approach (BFS-based).
        mask is a bitmask over M vertices."""
        # Extract vertex list
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb

        n = len(verts)
        if n == 0:
            return 0

        # Build local adjacency for these vertices
        # local_adj[i] = set of local indices j (j > i) connected to i
        local_adj = [set() for _ in range(n)]
        idx_map = {v: i for i, v in enumerate(verts)}
        for i in range(n):
            vi = verts[i]
            # neighbors of vi in the mask
            neigh = adj[vi] & mask
            nm = neigh
            while nm:
                lsb = nm & -nm
                vj = lsb.bit_length() - 1
                nm ^= lsb
                if vj != vi:
                    local_adj[i].add(idx_map[vj])

        # Edmonds' blossom algorithm is complex; use a greedy + augmenting path
        # matching for small graphs. For larger, use greedy heuristic.
        if n > 80:
            # Greedy matching heuristic for large subgraphs
            match = [-1] * n
            used = [False] * n
            for i in range(n):
                if used[i]:
                    continue
                for j in local_adj[i]:
                    if not used[j]:
                        match[i] = j
                        match[j] = i
                        used[i] = True
                        used[j] = True
                        break
            return sum(1 for i in range(n) if match[i] != -1 and i < match[i])

        # For small graphs, use BFS augmenting path matching
        match = [-1] * n

        def try_augment(u, visited):
            for v in local_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    if match[v] == -1 or try_augment(match[v], visited):
                        match[u] = v
                        match[v] = u
                        return True
            return False

        matching_size = 0
        for u in range(n):
            if match[u] == -1:
                visited = [False] * n
                if try_augment(u, visited):
                    matching_size += 1
        return matching_size

    def upper_bound(mask):
        """Upper bound on maximum independent set size in subgraph induced by mask."""
        n = mask.bit_count()
        if n <= 1:
            return n
        # Matching-based bound: alpha <= n - max_matching
        mm = max_matching(mask)
        return n - mm

    def branch_and_bound(mask, current_size, best_size, time_limit=8.0):
        """Branch and bound to find maximum independent set in subgraph induced by mask.
        Returns (best_mask, best_size) for the subproblem."""
        start_bb = time.time()
        best_mask = 0
        best_sz = 0

        # Sort vertices by degree (descending) for better branching
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            verts.append(v)

        # Order by degree in the induced subgraph, descending
        verts.sort(key=lambda v: (adj[v] & mask).bit_count(), reverse=True)

        # Recursive B&B with explicit stack to avoid recursion depth issues
        # State: (remaining_mask, selected_mask, selected_count)
        # We use a priority queue approach: explore most promising branches first

        # Actually, let's use a simpler recursive approach with memoization-like pruning
        def bb_search(rem_mask, sel_mask, sel_count):
            nonlocal best_mask, best_sz

            if time.time() - start_bb > time_limit:
                return

            if rem_mask == 0:
                if sel_count > best_sz:
                    best_sz = sel_count
                    best_mask = sel_mask
                return

            # Upper bound check
            ub = sel_count + upper_bound(rem_mask)
            if ub <= best_sz:
                return

            # Pick a vertex to branch on (highest degree in rem_mask)
            # Find vertex with max degree in rem_mask
            best_v = -1
            max_deg = -1
            rm = rem_mask
            while rm:
                lsb = rm & -rm
                v = lsb.bit_length() - 1
                rm ^= lsb
                deg = (adj[v] & rem_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v

            if best_v == -1:
                return

            v_bit = 1 << best_v

            # Branch 1: Include best_v
            new_rem = rem_mask & ~(v_bit | adj[best_v])
            bb_search(new_rem, sel_mask | v_bit, sel_count + 1)

            # Branch 2: Exclude best_v
            bb_search(rem_mask & ~v_bit, sel_mask, sel_count)

        bb_search(mask, 0, 0)
        return best_mask, best_sz

    # ---------- Main: ILS with B&B refinement ----------

    # Initial solution via greedy
    best_mask = greedy_min_degree()
    best_mask = local_add(best_mask)
    best_size = best_mask.bit_count()

    # Try to improve with swap
    best_mask = swap_improve(best_mask, max_rounds=100)
    best_size = best_mask.bit_count()

    # ILS loop
    while elapsed() < DEADLINE - 5.0:
        # Perturbation: remove 1-3 random vertices from best
        current = best_mask
        n_remove = random.randint(1, 3)
        for _ in range(n_remove):
            if current == 0:
                break
            # Pick random set bit
            r = random.randint(0, M - 1)
            if current & (1 << r):
                current ^= (1 << r)

        # Local improvement
        current = local_add(current)
        current = swap_improve(current, max_rounds=80)
        cur_size = current.bit_count()

        if cur_size > best_size:
            best_size = cur_size
            best_mask = current

        # Occasionally try a fresh greedy with randomization
        if random.random() < 0.3:
            available = all_valid_mask
            sel = 0
            while available:
                # Weighted random selection
                verts_avail = []
                am = available
                while am:
                    lsb = am & -am
                    v = lsb.bit_length() - 1
                    am ^= lsb
                    deg = (adj[v] & available).bit_count()
                    verts_avail.append((deg + random.uniform(0, 1.0), v))
                if not verts_avail:
                    break
                verts_avail.sort()
                best_v = verts_avail[0][1]
                sel |= (1 << best_v)
                available &= ~((1 << best_v) | (adj[best_v] & available))
            sel = local_add(sel)
            sel = swap_improve(sel, max_rounds=50)
            if sel.bit_count() > best_size:
                best_size = sel.bit_count()
                best_mask = sel

    # Final attempt: try B&B on the current best to see if we can prove or improve
    # B&B on the full graph is too expensive, but we can try on a restricted subgraph
    # Or: try B&B to see if we can find a larger set by exploring from best
    if elapsed() < DEADLINE - 3.0:
        # Try to add more vertices via B&B on the complement
        # Find vertices not in best_mask that could potentially be added
        not_in_best = all_valid_mask & ~best_mask
        if not_in_best:
            # Try B&B on a small subproblem: can we find a better set?
            # Limit: only try if the gap is small
            ub_full = upper_bound(all_valid_mask)
            if ub_full > best_size:
                # Try B&B with limited time
                time_for_bb = min(5.0, DEADLINE - elapsed() - 2.0)
                if time_for_bb > 1.0:
                    bb_mask, bb_sz = branch_and_bound(all_valid_mask, 0, best_size, time_limit=time_for_bb)
                    if bb_sz > best_size:
                        best_size = bb_sz
                        best_mask = bb_mask

    # Final local improvement
    best_mask = local_add(best_mask)

    # Convert to 1-based list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(vertex_map[v] + 1)
        m ^= lsb

    return sorted(result)
