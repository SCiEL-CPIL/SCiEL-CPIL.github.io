import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch_max_cliques(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivoting."""
        cliques = []

        def expand(R, P, X):
            if time.time() > deadline:
                return
            if P == 0 and X == 0:
                if R != 0:
                    cliques.append(R)
                return
            if P == 0:
                return

            # Choose pivot: vertex in P|X with most neighbors in P
            pivot_set = P | X
            best_pivot = -1
            best_count = -1
            ps = pivot_set
            while ps:
                v = (ps & -ps).bit_length() - 1
                count = (P & adj[v]).bit_count()
                if count > best_count:
                    best_count = count
                    best_pivot = v
                ps &= ps - 1

            if best_pivot >= 0:
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P

            cs = candidates
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                new_R = R | v_bit
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                P &= ~v_bit
                X |= v_bit
                cs &= ~v_bit

        all_vertices = (1 << n) - 1
        expand(0, all_vertices, 0)
        return cliques

    def greedy_hitting_set(cliques, n):
        """Greedy hitting set: repeatedly pick the vertex that covers the most uncovered cliques."""
        if not cliques:
            return []

        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return []

        m = len(cliques)
        # For each vertex, which cliques it's in
        vertex_to_cliques = [0] * n  # bitset of clique indices
        for ci, clique in enumerate(cliques):
            bit = 1 << ci
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v] |= bit
                cs &= ~v_bit

        uncovered = (1 << m) - 1
        hitting_set = set()

        while uncovered:
            best_v = -1
            best_count = -1
            for v in range(n):
                if v in hitting_set:
                    continue
                count = (vertex_to_cliques[v] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_v = v

            if best_v == -1 or best_count == 0:
                break

            hitting_set.add(best_v)
            uncovered &= ~vertex_to_cliques[best_v]

        return sorted(hitting_set)

    def branch_and_bound_hitting_set(cliques, n, time_limit, initial_solution=None):
        """
        Branch and bound for minimum hitting set using bitsets.
        """
        if not cliques:
            return []

        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return []

        m = len(cliques)

        # Sort cliques by size (smaller first for better pruning)
        cliques.sort(key=lambda c: c.bit_count())
        m = len(cliques)

        # Precompute vertex-to-cliques mapping as bitsets
        vertex_to_cliques = [0] * n
        for ci, clique in enumerate(cliques):
            bit = 1 << ci
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v] |= bit
                cs &= ~v_bit

        # Count occurrences
        vertex_count = [0] * n
        for v in range(n):
            vertex_count[v] = vertex_to_cliques[v].bit_count()

        # Initial upper bound
        if initial_solution is not None:
            best_size = len(initial_solution)
            best_solution = list(initial_solution)
        else:
            greedy = greedy_hitting_set(cliques, n)
            best_size = len(greedy)
            best_solution = greedy

        # Convert initial solution to bitset
        initial_bitset = 0
        for v in best_solution:
            initial_bitset |= (1 << v)

        # Track which cliques are covered by the initial solution
        covered_by_initial = 0
        for v in best_solution:
            covered_by_initial |= vertex_to_cliques[v]

        # Remaining cliques not covered by initial solution
        initial_remaining = ((1 << m) - 1) & ~covered_by_initial

        def solve(remaining, current_size, current_set):
            nonlocal best_size, best_solution

            if time.time() > time_limit:
                return

            if not remaining:
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_set)
                return

            if current_size >= best_size:
                return

            # Lower bound: number of cliques / max coverage per vertex
            # For each clique, at least one vertex must be added
            # Simple lower bound: count of remaining cliques that are singletons in terms of uncovered vertices
            # Better: find cliques that can only be hit by few vertices

            # Greedy lower bound: repeatedly pick the clique with fewest available vertices
            # and count how many vertices we'd need
            lb_remaining = remaining
            lb_count = 0
            temp_covered = 0
            while lb_remaining and lb_count < best_size - current_size:
                # Find clique with fewest vertices not in current_set
                best_ci = -1
                best_min = n + 1
                rs = lb_remaining
                while rs:
                    ci_bit = rs & -rs
                    ci = ci_bit.bit_length() - 1
                    clique = cliques[ci]
                    count = 0
                    cs = clique
                    while cs:
                        v_bit = cs & -cs
                        v = v_bit.bit_length() - 1
                        if v not in current_set:
                            count += 1
                        cs &= ~v_bit
                    if count == 0:
                        return  # impossible
                    if count < best_min:
                        best_min = count
                        best_ci = ci
                    rs &= ~ci_bit

                if best_ci == -1:
                    break

                lb_count += 1
                # Remove all cliques that share a vertex with cliques[best_ci]
                clique = cliques[best_ci]
                shared = 0
                cs = clique
                while cs:
                    v_bit = cs & -cs
                    v = v_bit.bit_length() - 1
                    shared |= vertex_to_cliques[v]
                    cs &= ~v_bit
                lb_remaining &= ~shared

            if current_size + lb_count >= best_size:
                return

            # Pick the most constrained clique
            best_ci = -1
            best_min_count = n + 1
            rs = remaining
            while rs:
                ci_bit = rs & -rs
                ci = ci_bit.bit_length() - 1
                clique = cliques[ci]
                count = 0
                cs = clique
                while cs:
                    v_bit = cs & -cs
                    v = v_bit.bit_length() - 1
                    if v not in current_set:
                        count += 1
                    cs &= ~v_bit
                if count == 0:
                    return
                if count < best_min_count:
                    best_min_count = count
                    best_ci = ci
                rs &= ~ci_bit

            if best_ci == -1:
                return

            clique = cliques[best_ci]
            cs = clique
            vertices_in_clique = []
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                if v not in current_set:
                    vertices_in_clique.append(v)
                cs &= ~v_bit

            # Sort vertices by frequency (most frequent first)
            vertices_in_clique.sort(key=lambda v: -vertex_count[v])

            for v in vertices_in_clique:
                if time.time() > time_limit:
                    return
                if current_size >= best_size:
                    return

                current_set.add(v)
                # Remove all cliques containing v from remaining
                new_remaining = remaining & ~vertex_to_cliques[v]

                solve(new_remaining, current_size + 1, current_set)
                current_set.discard(v)

        solve(initial_remaining, len(best_solution), set(best_solution))

        if best_solution is not None:
            return best_solution
        return greedy_hitting_set(cliques, n)

    def local_search_improve(hitting_set, cliques, n, max_iters=200):
        """
        Local search: try removing vertices, then 1-swap improvements.
        """
        if not cliques:
            return hitting_set

        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return hitting_set

        m = len(cliques)

        # For each vertex, which cliques it covers
        vertex_to_cliques = [0] * n
        for ci, clique in enumerate(cliques):
            bit = 1 << ci
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v] |= bit
                cs &= ~v_bit

        current = set(hitting_set)

        def covered_cliques(hs):
            cov = 0
            for v in hs:
                cov |= vertex_to_cliques[v]
            return cov

        def is_valid(hs):
            cov = covered_cliques(hs)
            return cov == ((1 << m) - 1)

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try removing each vertex in current
            for v in list(current):
                new_current = current - {v}
                if is_valid(new_current):
                    current = new_current
                    improved = True
                    break

            if improved:
                continue

            # Try 1-swap: remove v_out, add v_in
            for v_out in list(current):
                for v_in in range(n):
                    if v_in in current:
                        continue
                    new_current = (current - {v_out}) | {v_in}
                    if is_valid(new_current) and len(new_current) < len(current):
                        current = new_current
                        improved = True
                        break
                if improved:
                    break

        return sorted(current)

    def local_search_2swap(hitting_set, cliques, n, max_iters=50):
        """Try 2-swap improvements for further optimization."""
        if not cliques:
            return hitting_set

        cliques = [c for c in cliques if c.bit_count() >= 2]
        if not cliques:
            return hitting_set

        m = len(cliques)
        all_cliques_bitset = (1 << m) - 1

        vertex_to_cliques = [0] * n
        for ci, clique in enumerate(cliques):
            bit = 1 << ci
            cs = clique
            while cs:
                v_bit = cs & -cs
                v = v_bit.bit_length() - 1
                vertex_to_cliques[v] |= bit
                cs &= ~v_bit

        current = set(hitting_set)

        def is_valid(hs):
            cov = 0
            for v in hs:
                cov |= vertex_to_cliques[v]
            return cov == all_cliques_bitset

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try 2-swap: remove v1, v2, add w1, w2 (if it reduces size)
            # Only try if we can reduce size (remove 1, add 0 is already handled)
            # For 2-swap to reduce size: remove 2, add 1
            for v1 in list(current):
                for v2 in list(current):
                    if v1 >= v2:
                        continue
                    for w in range(n):
                        if w in current:
                            continue
                        new_current = (current - {v1, v2}) | {w}
                        if len(new_current) < len(current) and is_valid(new_current):
                            current = new_current
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return sorted(current)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices with degree > 0
            adj = generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                if adj[u]:
                    touched.add(u)
            results.append(sorted(touched))
            continue

        adj = generate_graph(n, p, seed)

        # Find all maximal cliques
        maximal_cliques = bron_kerbosch_max_cliques(adj, n)

        # Filter to cliques with at least 2 vertices
        maximal_cliques = [c for c in maximal_cliques if c.bit_count() >= 2]

        if not maximal_cliques:
            results.append([])
            continue

        # Time budget for this instance
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed
        time_limit = min(remaining_time, 100.0)
        if time_limit < 1.0:
            time_limit = 1.0

        # Get greedy solution as initial upper bound
        greedy_solution = greedy_hitting_set(maximal_cliques, n)

        # Try exact B&B with time limit
        exact_solution = branch_and_bound_hitting_set(
            maximal_cliques, n, time_limit, initial_solution=greedy_solution
        )

        best = exact_solution

        # Local search improvement
        improved = local_search_improve(best, maximal_cliques, n, max_iters=100)

        # 2-swap local search
        improved2 = local_search_2swap(improved, maximal_cliques, n, max_iters=30)

        results.append(improved2)

    return results
