"""Kissing number in dimension 11: maximum independent set in conflict graph of norm-4 integer vectors."""
import numpy as np
import itertools
import time
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s

    # Generate all candidate vectors with norm^2 = 4
    # Type A: weight 4, entries in {-1,0,1}
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    n4 = len(v4_list)

    # Type B: weight 1, entry ±2
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    n1 = len(v1_list)

    n_total = n4 + n1
    all_vectors = np.array(v4_list + v1_list, dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2 (since norm^2=4, threshold = 4/2 = 2)
    dots = all_vectors @ all_vectors.T
    conflicts = (dots > 2).astype(np.int32)
    np.fill_diagonal(conflicts, 0)

    # Separate subgraphs
    # W4-W4 conflicts (indices 0..n4-1)
    # W1-W1 conflicts (indices n4..n4+n1-1)
    # W4-W1 conflicts

    # Build W4-W4 conflict graph
    w4_conf = conflicts[:n4, :n4]
    w1_conf = conflicts[n4:, n4:]
    w4_w1_conf = conflicts[:n4, n4:]  # n4 x n1

    # --- Phase 1: Solve W4-W4 independent set via bipartite matching (if bipartite) ---
    # Check if W4-W4 graph is bipartite using BFS 2-coloring
    color = np.full(n4, -1, dtype=np.int32)
    is_bipartite = True
    for start in range(n4):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            # Get neighbors of u in W4-W4 graph
            nbrs = np.where(w4_conf[u] == 1)[0]
            for v in nbrs:
                if color[v] == -1:
                    color[v] = color[u] ^ 1
                    queue.append(v)
                elif color[v] == color[u]:
                    is_bipartite = False
                    break
            if not is_bipartite:
                break
        if not is_bipartite:
            break

    # If bipartite, use Hopcroft-Karp to find maximum matching
    # Left side: color 0, Right side: color 1
    if is_bipartite:
        left_nodes = np.where(color == 0)[0]
        right_nodes = np.where(color == 1)[0]
        n_left = len(left_nodes)
        n_right = len(right_nodes)

        # Build adjacency: for each left node, which right nodes it connects to
        # left_to_right[i] = list of right indices j such that w4_conf[left_nodes[i], right_nodes[j]] == 1
        left_to_right = []
        for i in range(n_left):
            u = left_nodes[i]
            nbrs = np.where(w4_conf[u] == 1)[0]
            # Filter to right-side nodes
            right_nbrs = [np.where(right_nodes == v)[0][0] for v in nbrs if color[v] == 1]
            left_to_right.append(right_nbrs)

        # Hopcroft-Karp
        def hopcroft_karp(adj, n_l, n_r):
            pair_u = [-1] * n_l
            pair_v = [-1] * n_r
            dist = [0] * n_l

            def bfs():
                from collections import deque
                q = deque()
                for u in range(n_l):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        q.append(u)
                    else:
                        dist[u] = -1
                found = False
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            q.append(pu)
                return found

            def dfs(u):
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = -1
                return False

            matching = 0
            while bfs():
                for u in range(n_l):
                    if pair_u[u] == -1:
                        if dfs(u):
                            matching += 1
            return matching, pair_u, pair_v

        matching, pair_u, pair_v = hopcroft_karp(left_to_right, n_left, n_right)
        # Maximum independent set in bipartite graph = n_total - minimum vertex cover = n_total - maximum matching
        # But we want the actual set of vertices in the independent set
        # Minimum vertex cover: for each matched edge, pick one endpoint
        # Standard: Z = (unmatched left) U (left reachable from unmatched left via alternating paths)
        # Min vertex cover = (Left \ Z) U (Right ∩ Z)
        # Independent set = complement of min vertex cover

        # Find Z: vertices reachable from unmatched left vertices via alternating paths
        # Start from unmatched left vertices, follow unmatched edges to right, matched edges to left
        visited_left = [False] * n_left
        visited_right = [False] * n_r
        q = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                visited_left[u] = True
                q.append(('L', u))
        while q:
            typ, idx = q.popleft()
            if typ == 'L':
                # From left node, follow unmatched edges to right
                for v in left_to_right[idx]:
                    if not visited_right[v]:
                        visited_right[v] = True
                        q.append(('R', v))
            else:
                # From right node, follow matched edge to left (if exists)
                if pair_v[idx] != -1:
                    u = pair_v[idx]
                    if not visited_left[u]:
                        visited_left[u] = True
                        q.append(('L', u))

        # Min vertex cover = (Left \ Z) U (Right ∩ Z)
        # Independent set = (Left ∩ Z) U (Right \ Z)
        indep_left = []
        for u in range(n_left):
            if visited_left[u]:
                indep_left.append(left_nodes[u])
        indep_right = []
        for v in range(n_r):
            if not visited_right[v]:
                indep_right.append(right_nodes[v])

        w4_indep_set = set(indep_left + indep_right)
        w4_indep_size = len(w4_indep_set)
    else:
        w4_indep_set = set()
        w4_indep_size = 0

    # --- Phase 2: Multi-start local search on full graph ---
    # Use bitset for speed
    # conflicts as bitsets: each row is a Python int (bitset)
    # N is at most ~3544, so we need ~56 64-bit words or use Python's arbitrary precision ints

    # Convert conflict matrix to list of Python ints (bitsets)
    # conflicts[i] as a big integer where bit j is set if i conflicts with j
    conflict_bits = []
    for i in range(n_total):
        bits = 0
        row = conflicts[i]
        # Build bitset from boolean array
        for j in range(n_total):
            if row[j]:
                bits |= (1 << j)
        conflict_bits.append(bits)

    # Precompute: for each vertex, its neighbors as a set (for quick lookup)
    # Actually, bitsets are enough.

    # Local search using bitsets
    rng = np.random.default_rng(42)

    def bit_count(x):
        return bin(x).count('1')

    def local_search(init_set_bits, max_time=2.0):
        """
        Local search on the full conflict graph using bitsets.
        init_set_bits: Python int representing the initial independent set
        Returns: (best_set_bits, best_size)
        """
        current = init_set_bits
        current_size = bit_count(current)
        best = current
        best_size = current_size

        # Precompute: for each vertex not in set, its conflict count with set
        # conflict_count[i] = number of neighbors of i that are in current set
        # = bit_count(conflict_bits[i] & current)

        ls_deadline = time.time() + max_time
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0

        while time.time() < ls_deadline and T > T_min:
            improved = False

            # Phase 1: Add all vertices with 0 conflicts
            # For each vertex not in set, check if conflict_bits[v] & current == 0
            not_in_set_mask = ((1 << n_total) - 1) ^ current
            # Find vertices with 0 conflicts
            zero_conf = []
            # Iterate over set bits in not_in_set_mask
            mask = not_in_set_mask
            while mask:
                # Get lowest set bit
                v = (mask & (-mask)).bit_length() - 1
                if (conflict_bits[v] & current) == 0:
                    zero_conf.append(v)
                mask &= mask - 1  # Clear lowest set bit

            if zero_conf:
                for v in zero_conf:
                    current |= (1 << v)
                    current_size += 1
                improved = True
                if current_size > best_size:
                    best_size = current_size
                    best = current
                no_improve = 0

            # Phase 2: Swaps
            if time.time() < ls_deadline and current_size > 0:
                # Find vertices v not in set with exactly 1 conflict in set
                # These can be swapped with their unique neighbor u in set
                not_in_set_mask = ((1 << n_total) - 1) ^ current
                swap_candidates = []
                mask = not_in_set_mask
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    cc = bit_count(conflict_bits[v] & current)
                    if cc == 1:
                        # Find the unique neighbor u in set
                        # u is the only bit set in (conflict_bits[v] & current)
                        u = ((conflict_bits[v] & current) & (-(conflict_bits[v] & current))).bit_length() - 1
                        swap_candidates.append((v, u))
                    mask &= mask - 1

                if swap_candidates:
                    # Try a random swap
                    v, u = swap_candidates[rng.integers(len(swap_candidates))]
                    # Perform swap: remove u, add v
                    new_set = (current & ~(1 << u)) | (1 << v)
                    # Accept with probability T
                    if rng.random() < T:
                        current = new_set
                        # Size unchanged
                        improved = True
                    else:
                        pass  # revert (we didn't change current)

            if not improved:
                no_improve += 1
                if no_improve > 30:
                    # Perturb: remove 2-3 random vertices
                    set_mask = current
                    set_vertices = []
                    mask = set_mask
                    while mask:
                        v = (mask & (-mask)).bit_length() - 1
                        set_vertices.append(v)
                        mask &= mask - 1

                    if len(set_vertices) >= 2:
                        n_remove = min(3, len(set_vertices))
                        to_remove = rng.choice(len(set_vertices), size=n_remove, replace=False)
                        for idx in to_remove:
                            v = set_vertices[idx]
                            current &= ~(1 << v)
                            current_size -= 1
                        no_improve = 0
                    else:
                        break
            else:
                no_improve = 0

            T *= cooling

        # Final polish
        not_in_set_mask = ((1 << n_total) - 1) ^ current
        mask = not_in_set_mask
        while mask:
            v = (mask & (-mask)).bit_length() - 1
            if (conflict_bits[v] & current) == 0:
                current |= (1 << v)
                current_size += 1
            mask &= mask - 1

        if current_size > best_size:
            best_size = current_size
            best = current

        return best, best_size

    def greedy_init():
        """Greedy min-degree initialization using bitsets."""
        in_set = 0
        alive = (1 << n_total) - 1
        # degrees[i] = number of alive neighbors of i
        # Initialize
        degrees = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            degrees[i] = bit_count(conflict_bits[i] & alive)

        while alive and time.time() < deadline - 5:
            # Find min degree among alive vertices
            alive_list = []
            mask = alive
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                alive_list.append(v)
                mask &= mask - 1

            if not alive_list:
                break

            min_deg = min(degrees[v] for v in alive_list)
            cands = [v for v in alive_list if degrees[v] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            in_set |= (1 << chosen)
            alive &= ~(1 << chosen)

            # Remove neighbors of chosen from alive
            nbrs = conflict_bits[chosen]
            alive &= ~nbrs

            # Update degrees
            # For each alive vertex i, degrees[i] -= number of neighbors of i in {chosen} ∪ nbrs
            # Since chosen is not a neighbor of i (i is alive), just subtract overlap with nbrs
            mask = alive
            while mask:
                i = (mask & (-mask)).bit_length() - 1
                degrees[i] -= bit_count(conflict_bits[i] & nbrs)
                mask &= mask - 1

        return in_set

    # --- Main search loop ---
    best_bits = 0
    best_size = 0

    # Start with the bipartite solution as initial best
    if w4_indep_size > 0:
        # Convert w4_indep_set to bitset
        init_bits = 0
        for v in w4_indep_set:
            init_bits |= (1 << v)
        # Try to add W1 vectors
        current = init_bits
        current_size = w4_indep_size
        # Try adding W1 vectors
        for v in range(n4, n_total):
            if (conflict_bits[v] & current) == 0:
                current |= (1 << v)
                current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_bits = current

    # Multi-start local search
    n_restarts = 0
    max_restarts = 500

    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1

        # Greedy initialization
        init_bits = greedy_init()
        init_size = bit_count(init_bits)

        if init_size > best_size:
            best_size = init_size
            best_bits = init_bits

        # Local search
        remaining = deadline - 8 - time.time()
        if remaining < 1.0:
            break

        ls_time = min(remaining * 0.3, 2.5)
        if ls_time < 0.3:
            ls_time = 0.3

        ls_result, ls_size = local_search(init_bits, max_time=ls_time)

        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_result

        if time.time() > deadline - 10:
            break

    # Also try local search starting from the bipartite solution
    if w4_indep_size > 0 and time.time() < deadline - 5:
        init_bits = 0
        for v in w4_indep_set:
            init_bits |= (1 << v)
        # Add W1 vectors greedily
        current = init_bits
        for v in range(n4, n_total):
            if (conflict_bits[v] & current) == 0:
                current |= (1 << v)
        ls_result, ls_size = local_search(current, max_time=min(2.0, deadline - 5 - time.time()))
        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_result

    # Convert best_bits to index array
    final_indices = []
    mask = best_bits
    while mask:
        v = (mask & (-mask)).bit_length() - 1
        final_indices.append(v)
        mask &= mask - 1

    if not final_indices:
        return np.array([], dtype=np.int64)

    result_vectors = all_vectors[final_indices]
    return result_vectors
