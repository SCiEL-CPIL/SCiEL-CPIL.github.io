"""Combined program: ILP-optimal with spectral/local-search fallback."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    N = 128

    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Build conflict matrix: conflict[a][b] = bitmask of c values where {a,b,c} is a bad triple
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    # Also build edge weights for spectral methods
    edge_weight = [[0] * (N + 1) for _ in range(N + 1)]
    # Node degree (number of bad triples each node participates in)
    node_degree = [0] * (N + 1)

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    edge_weight[a][b] += 1
                    edge_weight[b][a] += 1
                    edge_weight[a][c] += 1
                    edge_weight[c][a] += 1
                    edge_weight[b][c] += 1
                    edge_weight[c][b] += 1
                    node_degree[a] += 1
                    node_degree[b] += 1
                    node_degree[c] += 1

    # Helper: check if adding x to a set (represented by mask) creates a bad triple
    def can_add(x, mask):
        """Check if x can be added to the set represented by mask without creating a bad triple."""
        for y in range(1, N + 1):
            if (mask >> y) & 1:
                if conflict[x][y] & mask:
                    return False
        return True

    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # --- Try ILP approach first ---
    def try_ilp():
        """Try to solve via ILP using scipy.optimize.milp."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
        except ImportError:
            return None

        # Variables: x_i = 1 if i is in the set
        # Maximize sum(x_i)
        # Subject to: for each bad triple (a,b,c), x_a + x_b + x_c <= 2
        # x_i in {0, 1}

        # Collect all bad triples
        bad_triples = []
        for a in range(1, N + 1):
            for b in range(a + 1, N + 1):
                for c in range(b + 1, N + 1):
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        bad_triples.append((a, b, c))

        if not bad_triples:
            return list(range(1, N + 1))

        n_vars = N
        n_constraints = len(bad_triples)

        # Objective: maximize sum(x_i), so minimize -sum(x_i)
        c = np.ones(n_vars)

        # Constraints: for each triple (a,b,c): x_a + x_b + x_c <= 2
        A = np.zeros((n_constraints, n_vars))
        for idx, (a, b, c_) in enumerate(bad_triples):
            A[idx, a - 1] = 1
            A[idx, b - 1] = 1
            A[idx, c_ - 1] = 1

        constraints = [
            LinearConstraint(A, -np.inf, 2),
            Bounds(lb=0, ub=1)
        ]

        integrality = np.ones(n_vars)

        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                options={"time_limit": 50}
            )
            if result.success:
                sol = [i + 1 for i in range(n_vars) if result.x[i] > 0.5]
                if sol:
                    return sol
        except Exception:
            pass

        return None

    # --- Spectral embedding for ordering ---
    def spectral_embed():
        """Find leading eigenvector of conflict graph via power iteration."""
        rng_spec = random.Random(123)
        v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            v[i] = rng_spec.uniform(-1, 1)

        for _ in range(30):
            new_v = [0.0] * (N + 1)
            for i in range(1, N + 1):
                s = 0.0
                for j in range(1, N + 1):
                    if edge_weight[i][j] > 0:
                        s += edge_weight[i][j] * v[j]
                new_v[i] = s
            norm = math.sqrt(sum(x * x for x in new_v[1:]))
            if norm < 1e-10:
                break
            for i in range(1, N + 1):
                new_v[i] /= norm
            v = new_v
        return v

    # --- Greedy with conflict matrix ---
    def greedy(order, start_mask=0):
        mask = start_mask
        for x in order:
            if not (mask >> x) & 1:
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    # --- Local search: remove k, re-add greedily ---
    def local_search(current_mask, eigvec, rng, max_rounds=200):
        best_mask = current_mask
        best_size = bin(current_mask).count("1")
        improved = True
        rounds = 0

        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1

            for k in [1, 2, 3, 4, 5, 8]:
                if time.time() > deadline - 5.0:
                    break

                set_bits = mask_to_list(best_mask)
                if len(set_bits) < k:
                    continue

                # Remove k elements, preferring high-conflict
                weights = [node_degree[x] + 1 for x in set_bits]
                total_w = sum(weights)
                removed = []
                available = set(set_bits)
                for _ in range(min(k, len(set_bits))):
                    r = rng.random() * total_w
                    cum = 0
                    chosen = None
                    for x in available:
                        cum += node_degree[x] + 1
                        if cum >= r:
                            chosen = x
                            break
                    if chosen is None:
                        chosen = next(iter(available))
                    removed.append(chosen)
                    available.discard(chosen)
                    total_w -= node_degree[chosen] + 1

                new_mask = best_mask
                for x in removed:
                    new_mask &= ~(1 << x)

                # Re-add greedily using spectral order
                candidates = [x for x in range(1, N + 1) if not (new_mask >> x) & 1]
                candidates.sort(key=lambda x: eigvec[x])
                m = greedy(candidates, new_mask)
                sz = bin(m).count("1")

                if sz > best_size:
                    best_size = sz
                    best_mask = m
                    improved = True
                    break

        return best_mask, best_size

    # --- 1-for-2 swap local search ---
    def swap_search(best_mask, eigvec, rng, max_rounds=100):
        improved = True
        rounds = 0
        current_mask = best_mask
        current_size = bin(current_mask).count("1")

        while improved and rounds < max_rounds and time.time() < deadline - 3.0:
            improved = False
            rounds += 1

            set_bits = mask_to_list(current_mask)
            not_in = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]

            # Try 1-for-2: remove 1, add up to 2
            for rem_x in set_bits:
                if time.time() > deadline - 3.0:
                    break
                trial_mask = current_mask & ~(1 << rem_x)
                added = 0
                # Sort candidates by spectral preference
                not_in_sorted = sorted(not_in, key=lambda x: eigvec[x])
                for add_x in not_in_sorted:
                    if time.time() > deadline - 3.0 or added >= 2:
                        break
                    if can_add(add_x, trial_mask):
                        trial_mask |= (1 << add_x)
                        added += 1
                if added >= 1:
                    sz = bin(trial_mask).count("1")
                    if sz > current_size:
                        current_size = sz
                        current_mask = trial_mask
                        improved = True
                        break

        return current_mask, current_size

    # --- Main search ---
    best_mask = 0
    best_size = 0

    # Try ILP first
    ilp_result = try_ilp()
    if ilp_result is not None:
        best_mask = list_to_mask(ilp_result)
        best_size = len(ilp_result)
    else:
        # Fallback: multi-phase heuristic search
        rng = random.Random(42)

        # Spectral embedding
        eigvec = spectral_embed()

        # Phase 1: Spectral-guided greedy
        spec_order = sorted(range(1, N + 1), key=lambda x: eigvec[x])
        m = greedy(spec_order)
        sz = bin(m).count("1")
        if sz > best_size:
            best_size = sz
            best_mask = m

        spec_order_rev = sorted(range(1, N + 1), key=lambda x: -eigvec[x])
        m = greedy(spec_order_rev)
        sz = bin(m).count("1")
        if sz > best_size:
            best_size = sz
            best_mask = m

        # Phase 2: Standard greedy orderings
        orders = [
            list(range(1, N + 1)),
            list(range(N, 0, -1)),
            sorted(range(1, N + 1), key=lambda x: node_degree[x]),
            sorted(range(1, N + 1), key=lambda x: -node_degree[x]),
        ]

        for order in orders:
            m = greedy(order)
            sz = bin(m).count("1")
            if sz > best_size:
                best_size = sz
                best_mask = m

        # Phase 3: Random restarts
        for trial in range(50000):
            if time.time() > deadline - 10.0:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            m = greedy(order)
            sz = bin(m).count("1")
            if sz > best_size:
                best_size = sz
                best_mask = m

        # Phase 4: Local search
        best_mask, best_size = local_search(best_mask, eigvec, rng)

        # Phase 5: Swap search
        best_mask, best_size = swap_search(best_mask, eigvec, rng)

    # Final validation
    result = mask_to_list(best_mask)
    if not result:
        return [1]

    # Verify no bad triples
    valid = True
    for a, b, c in combinations(result, 3):
        if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
            valid = False
            break

    if not valid:
        # Fallback: rebuild greedily
        m = greedy(list(range(1, N + 1)))
        result = mask_to_list(m)
        if not result:
            return [1]

    return result
# EVOLVE-BLOCK-END
