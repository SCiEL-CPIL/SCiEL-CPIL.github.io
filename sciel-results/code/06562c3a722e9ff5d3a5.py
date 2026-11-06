import time
import random
from itertools import combinations


def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                if a + S_list[j] in Sset:
                    return False
        return True

    def get_modular_subset(V, M):
        """Subset of V whose residues mod M fall in [M/3+1, 2M/3] (sum-free mod M)."""
        L = M // 3 + 1
        R = (2 * M) // 3
        S = set()
        for x in V:
            r = x % M
            if L <= r <= R:
                S.add(x)
        return S

    def build_conflict_graph(V):
        """
        Conflict graph: edge (i,j) if some k in V has V[i]+V[j]==V[k] (i<=j).
        A set S is sum-free iff its index set is an independent set in this graph.
        """
        n = len(V)
        Vset = set(V)
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if V[i] + V[j] in Vset:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def greedy_seed(V, adj):
        """Greedy maximal independent set with degree-based ordering."""
        n = len(V)
        remaining = set(range(n))
        chosen = []
        # Order by degree ascending (pick low-conflict first)
        order = sorted(remaining, key=lambda v: len(adj[v]))
        for v in order:
            if v not in remaining:
                continue
            chosen.append(v)
            remaining.discard(v)
            for u in adj[v]:
                remaining.discard(u)
        return set(chosen)

    def modular_seed(V, M):
        idx = {x: i for i, x in enumerate(V)}
        S = get_modular_subset(V, M)
        return set(idx[x] for x in S)

    def par_seed(V):
        idx = {x: i for i, x in enumerate(V)}
        return set(idx[x] for x in V if x % 2 != 0)

    def sign_seed(V, positive):
        idx = {x: i for i, x in enumerate(V)}
        return set(idx[x] for x in V if (x > 0) == positive)

    def max_independent_set(adj, time_limit, start_time, init=None):
        """
        Branch-and-bound exact MIS solver with greedy upper bound.
        Returns the largest independent set (as a set of indices) found
        within time_limit (best-so-far is returned on timeout).
        """
        n = len(adj)
        best = list(init) if init else []
        best_set = set(best)

        # Precompute adjacency as sorted lists for speed
        adj_list = [sorted(adj[i]) for i in range(n)]

        # Global time guard
        def timed_out():
            return (time.time() - start_time) > time_limit

        # Greedy upper bound on a candidate set (bitset)
        def greedy_ub(cand):
            """Greedy maximal independent set size on cand (upper bound)."""
            remaining = set(cand)
            count = 0
            # Sort by degree within remaining
            while remaining:
                # pick vertex of min degree in remaining
                v = min(remaining, key=lambda x: len(adj_list[x] & remaining))
                count += 1
                remaining.discard(v)
                for u in adj_list[v]:
                    remaining.discard(u)
            return count

        # Fast upper bound: sum of 1/(deg_in_cand+1) style is loose; use a
        # cheap packing bound: number of vertices minus forced exclusions.
        # We use greedy UB but memoize on frozenset for small sets.
        ub_cache = {}

        def ub(cand):
            key = frozenset(cand)
            if key in ub_cache:
                return ub_cache[key]
            val = greedy_ub(cand)
            if len(ub_cache) < 200000:
                ub_cache[key] = val
            return val

        # Branching: choose vertex to branch on (high degree for tighter pruning)
        stack = [(set(range(n)), set())]  # (candidates, chosen)
        # We do DFS with explicit stack to allow clean timeout return.

        # To keep it tractable, use a best-first-ish DFS with pruning.
        # Sort initial candidates by degree to branch on "hard" vertices.

        # We'll do a recursive DFS but with manual stack to avoid recursion limits.
        # Each frame: (cand_set, chosen_set)
        # Branch on a vertex v in cand: include v (remove v and its neighbors)
        #                          or exclude v (remove v only).

        # Choose pivot: vertex with max degree in cand (reduces branching).
        def pick_pivot(cand):
            best_v = None
            best_d = -1
            for v in cand:
                d = len(adj_list[v] & cand)
                if d > best_d:
                    best_d = d
                    best_v = v
            return best_v, best_d

        # DFS
        # Use a list as stack of (cand, chosen)
        # We want to find large sets, so explore "include" branch first.
        # Order: include high-degree vertices first? Actually including a high-degree
        # vertex removes many, so to maximize size we may want to include low-degree.
        # But for pruning, branch on high-degree. We'll branch on high-degree pivot
        # but explore include-branch first only if it's promising.

        # Simpler robust approach: branch on pivot, explore both, prune by UB.
        # Track best globally.

        # To manage time, limit stack depth and use time checks.
        visited_states = set()

        # We'll do iterative DFS.
        # For correctness of "best found", we update best whenever chosen is larger
        # and is an independent set (it always is by construction).

        # Start
        stack = [(set(range(n)), set())]
        # Precompute: we want to explore include-branch for low-degree pivots to grow.
        # Let's branch on a pivot and recurse.

        # Because pure DFS can be slow, we interleave: at each node compute UB;
        # if UB <= len(chosen), prune. Else branch.

        # We'll implement with explicit stack and a node counter for time slicing.
        node_count = 0

        while stack:
            if timed_out():
                break
            cand, chosen = stack.pop()
            if len(chosen) + ub(cand) <= len(best_set):
                continue
            if not cand:
                if len(chosen) > len(best_set):
                    best_set = set(chosen)
                    best = list(best_set)
                continue
            # If current chosen plus all cand is independent (no edges in cand), take all
            # Check if cand is independent:
            indep = True
            for v in cand:
                if adj_list[v] & cand:
                    indep = False
                    break
            if indep:
                new_set = chosen | cand
                if len(new_set) > len(best_set):
                    best_set = set(new_set)
                    best = list(best_set)
                continue

            v, deg = pick_pivot(cand)
            # Branch: include v
            cand_incl = cand - {v} - set(adj_list[v])
            chosen_incl = chosen | {v}
            # Branch: exclude v
            cand_excl = cand - {v}
            chosen_excl = chosen

            # Push both; explore the more promising (larger UB) first
            ub_incl = len(chosen_incl) + ub(cand_incl)
            ub_excl = len(chosen_excl) + ub(cand_excl)
            if ub_excl > ub_incl:
                stack.append((cand_incl, chosen_incl))
                stack.append((cand_excl, chosen_excl))
            else:
                stack.append((cand_excl, chosen_excl))
                stack.append((cand_incl, chosen_incl))

            node_count += 1
            # Periodic safety
            if node_count % 100000 == 0 and timed_out():
                break

        return best

    def solve_instance(V, time_limit, start_time):
        n = len(V)
        min_v = min(V)
        max_v = max(V)
        span = max_v - min_v

        # Build conflict graph
        adj = build_conflict_graph(V)

        # Generate multiple seeds
        seeds = []
        # Modular seeds over a range of moduli
        candidate_mods = set()
        candidate_mods.update(range(7, 60))
        for factor in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
            m = int(span * factor)
            if m > 5:
                candidate_mods.update([m - 1, m, m + 1])
        for factor in [0.5, 1.0, 1.5, 2.0, 3.0]:
            m = int(n * factor)
            if m > 5:
                candidate_mods.update([m - 1, m, m + 1])
        candidate_mods.update(range(100, 400, 10))
        candidate_mods.update(range(500, 3000, 100))
        candidate_mods = [m for m in candidate_mods if m > 5]

        for M in candidate_mods:
            if (time.time() - start_time) > time_limit:
                break
            seeds.append(modular_seed(V, M))

        # Parity seed
        seeds.append(par_seed(V))
        # Sign seeds
        seeds.append(sign_seed(V, True))
        seeds.append(sign_seed(V, False))
        # Greedy seed
        seeds.append(greedy_seed(V, adj))

        # Pick the best seed by size
        best_seed = max(seeds, key=len)

        # Run exact branch-and-bound from this seed
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 1:
            return sorted(V[i] for i in best_seed)

        result_idx = max_independent_set(adj, remaining, start_time, init=best_seed)
        return sorted(V[i] for i in result_idx)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed

        if remaining_total < 8:
            # Fallback to parity (always sum-free)
            odds = [x for x in V if x % 2 != 0]
            results.append(sorted(odds))
            continue

        instances_left = 3 - len(results)
        time_per_instance = (remaining_total / instances_left) - 2
        if time_per_instance < 2:
            time_per_instance = 2

        S_best = solve_instance(V, time_per_instance, start)

        if not is_sum_free(S_best):
            odds = [x for x in V if x % 2 != 0]
            S_best = sorted(odds)

        results.append(S_best)

    return results


if __name__ == "__main__":
    res = construct()
    print([len(r) for r in res])
