import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets (Python integers)
    # adj_mask[i] has bit j set if (i+1)+(j+1) is a perfect square
    adj_mask = [0] * N

    # Identify vertices with self-loops (2*v is a square)
    forbidden = set()

    for i in range(N):
        v = i + 1
        # Check self-loop
        if 2 * v in squares:
            forbidden.add(i)

    # Build adjacency masks
    for i in range(N):
        v = i + 1
        mask = 0
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                mask |= (1 << j)
        adj_mask[i] = mask

    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    n_cand = len(candidates)

    # Map candidate index to vertex index
    cand_to_vertex = {cand: idx for idx, cand in enumerate(candidates)}
    vertex_to_cand = {idx: cand for cand, idx in cand_to_vertex.items()}

    # Build adjacency among candidates using bitsets over candidate indices
    # cand_adj[i] has bit j set if candidates[i] and candidates[j] conflict
    cand_adj = [0] * n_cand
    for i, vi in enumerate(candidates):
        mask = 0
        for j, vj in enumerate(candidates):
            if i != j and ((1 << vj) & adj_mask[vi]):
                mask |= (1 << j)
        cand_adj[i] = mask

    def bit_count(x):
        return bin(x).count('1')

    def greedy_color_upper_bound(vertices_mask):
        """
        Greedy coloring of the subgraph induced by vertices_mask.
        Returns the number of colors used, which is an upper bound on alpha.
        """
        # Get list of vertices
        verts = []
        for i in range(n_cand):
            if (vertices_mask >> i) & 1:
                verts.append(i)

        if not verts:
            return 0

        # Greedy coloring: assign colors in order of decreasing degree
        deg = {}
        for v in verts:
            # Count neighbors within verts
            mask = 0
            for u in verts:
                if u != v and ((cand_adj[v] >> u) & 1):
                    mask |= (1 << u)
            deg[v] = bit_count(mask)

        # Sort by degree descending
        verts_sorted = sorted(verts, key=lambda x: -deg[x])

        colors = {}
        color_classes = []  # list of sets of vertices with same color

        for v in verts_sorted:
            # Find neighbors already colored
            neighbor_colors = set()
            for u in verts:
                if u != v and ((cand_adj[v] >> u) & 1) and u in colors:
                    neighbor_colors.add(colors[u])

            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)

        return len(color_classes)

    def greedy_independent_set(vertices_mask):
        """
        Greedy: repeatedly pick vertex with minimum degree in remaining subgraph.
        """
        result = 0
        remaining = vertices_mask
        while remaining:
            # Find vertex with minimum degree in remaining
            best_v = -1
            best_deg = float('inf')
            for i in range(n_cand):
                if (remaining >> i) & 1:
                    # Count neighbors in remaining
                    neigh = cand_adj[i] & remaining
                    d = bit_count(neigh)
                    if d < best_deg:
                        best_deg = d
                        best_v = i

            if best_v == -1:
                break

            result |= (1 << best_v)
            # Remove best_v and its neighbors from remaining
            remaining &= ~(cand_adj[best_v] | (1 << best_v))

        return result

    def local_improve(s_mask):
        """Try to add vertices that don't conflict with current set."""
        s = s_mask
        improved = True
        while improved:
            improved = False
            # Find all vertices not in s that have no neighbors in s
            for i in range(n_cand):
                if not ((s >> i) & 1):
                    if not (cand_adj[i] & s):
                        s |= (1 << i)
                        improved = True
        return s

    def branch_and_bound(vertices_mask, current_size, current_set_mask):
        """
        Branch and bound for maximum independent set.
        Uses greedy coloring as upper bound.
        """
        if not vertices_mask:
            return current_size, current_set_mask

        # Upper bound via greedy coloring
        ub = greedy_color_upper_bound(vertices_mask) + current_size
        if ub <= best_size[0]:
            return current_size, current_set_mask

        # Pick a vertex to branch on
        # Use vertex with highest degree for better pruning
        best_v = -1
        best_deg = -1
        for i in range(n_cand):
            if (vertices_mask >> i) & 1:
                neigh = cand_adj[i] & vertices_mask
                d = bit_count(neigh)
                if d > best_deg:
                    best_deg = d
                    best_v = i

        if best_v == -1:
            return current_size, current_set_mask

        # Branch 1: include best_v
        new_set = current_set_mask | (1 << best_v)
        # Remove best_v and its neighbors
        new_mask = vertices_mask & ~(cand_adj[best_v] | (1 << best_v))
        s1, set1 = branch_and_bound(new_mask, current_size + 1, new_set)
        if s1 > best_size[0]:
            best_size[0] = s1
            best_set[0] = set1

        # Branch 2: exclude best_v
        new_mask2 = vertices_mask & ~(1 << best_v)
        s2, set2 = branch_and_bound(new_mask2, current_size, current_set_mask)
        if s2 > best_size[0]:
            best_size[0] = s2
            best_set[0] = set2

        return max(s1, s2), (set1 if s1 >= s2 else set2)

    best_size = [0]
    best_set = [0]

    # Strategy 1: Greedy + local improve with random restarts
    def greedy_solve(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        order = candidates[:]
        random.shuffle(order)

        current_set = 0
        for v in order:
            if not (cand_adj[v] & current_set):
                current_set |= (1 << v)

        return current_set

    # Strategy 2: Greedy by degree (min degree first)
    def greedy_by_degree():
        result = 0
        remaining = (1 << n_cand) - 1
        while remaining:
            best_v = -1
            best_deg = float('inf')
            for i in range(n_cand):
                if (remaining >> i) & 1:
                    neigh = cand_adj[i] & remaining
                    d = bit_count(neigh)
                    if d < best_deg:
                        best_deg = d
                        best_v = i
            if best_v == -1:
                break
            result |= (1 << best_v)
            remaining &= ~(cand_adj[best_v] | (1 << best_v))
        return result

    # Strategy 3: Simulated annealing
    def simulated_annealing(initial_mask, max_iters=50000):
        s = initial_mask
        current_size = bit_count(s)
        best_s = s
        best_sz = current_size
        temp = 10.0
        cooling = 0.9999

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 2:
                break
            # Pick a random vertex to flip
            i = random.randint(0, n_cand - 1)
            if (s >> i) & 1:
                # Try removing i
                new_s = s & ~(1 << i)
                new_sz = current_size - 1
                # Check if we can add more
                new_s = local_improve(new_s)
                new_sz = bit_count(new_s)
                delta = new_sz - current_size
            else:
                # Try adding i
                if cand_adj[i] & s:
                    continue  # Can't add
                new_s = s | (1 << i)
                new_sz = current_size + 1
                new_s = local_improve(new_s)
                new_sz = bit_count(new_s)
                delta = new_sz - current_size

            if delta >= 0 or random.random() < math.exp(delta / temp):
                s = new_s
                current_size = new_sz
                if current_size > best_sz:
                    best_sz = current_size
                    best_s = s

            temp *= cooling

        return best_s

    # Run multiple strategies
    # First, get a good initial solution
    init = greedy_by_degree()
    init = local_improve(init)
    if bit_count(init) > best_size[0]:
        best_size[0] = bit_count(init)
        best_set[0] = init

    # Random restart greedy + local improve
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        s = greedy_solve(random_seed=r * 12345 + 67890)
        s = local_improve(s)
        sz = bit_count(s)
        if sz > best_size[0]:
            best_size[0] = sz
            best_set[0] = s

    # Branch and bound on the full problem (may be too slow, but try with time limit)
    if time.time() - start_time < DEADLINE - 10:
        # Try B&B
        full_mask = (1 << n_cand) - 1
        s, set_mask = branch_and_bound(full_mask, 0, 0)
        if s > best_size[0]:
            best_size[0] = s
            best_set[0] = set_mask

    # Simulated annealing from best solution
    if time.time() - start_time < DEADLINE - 3:
        sa_result = simulated_annealing(best_set[0], max_iters=100000)
        if bit_count(sa_result) > best_size[0]:
            best_size[0] = bit_count(sa_result)
            best_set[0] = sa_result

    # More random restarts with different seeds
    for r in range(num_restarts, 400):
        if time.time() - start_time > DEADLINE - 2:
            break
        s = greedy_solve(random_seed=r * 777 + 111)
        s = local_improve(s)
        sz = bit_count(s)
        if sz > best_size[0]:
            best_size[0] = sz
            best_set[0] = s

    # Final local improve on best
    final = local_improve(best_set[0])
    if bit_count(final) > best_size[0]:
        best_size[0] = bit_count(final)
        best_set[0] = final

    # Convert back to 1-based values
    result = []
    for i in range(n_cand):
        if (best_set[0] >> i) & 1:
            result.append(candidates[i] + 1)

    return sorted(result)
