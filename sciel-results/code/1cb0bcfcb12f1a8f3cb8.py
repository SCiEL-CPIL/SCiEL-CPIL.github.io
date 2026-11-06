"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_one(S, seed):
        """Find a large sum-free subset of S using bitmask-based local search."""
        n = len(S)
        if n == 0:
            return []

        # Precompute violation triples: for each j, list of (a,b) such that
        # adding j to a set containing both a and b would create a violation.
        # Violation: x_a + x_b = x_j, or x_j + x_a = x_b, or x_j + x_b = x_a
        # i.e., one element is the sum of the other two.
        pairs = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = S[i] + S[j]
                if s in S:
                    k = S.index(s)  # slow, but n<=240 so OK
                    # Triple (i, j, k): x_i + x_j = x_k
                    # For each element in the triple, the other two block it
                    pairs[i].append((j, k))
                    pairs[j].append((i, k))
                    pairs[k].append((i, j))
        # Also handle 2*x_i = x_j (i.e., x_i + x_i = x_j)
        # This is already covered if we consider i=j in the above? No, we have i<j.
        # So we need to handle 2*x_i = x_j separately.
        for i in range(n):
            s = 2 * S[i]
            if s in S:
                j = S.index(s)
                if i != j:
                    # x_i + x_i = x_j, so triple (i, i, j)
                    # For element i: having i and j is a violation (i+i=j)
                    # For element j: having i and i is a violation, but i is only once
                    # Actually, the violation is: if both i and j are in set, then x_i + x_i = x_j
                    # So for i: if j is in set, i cannot be added (because i+i=j)
                    # For j: if i is in set, j cannot be added (because i+i=j)
                    # This is a pair constraint, not a triple. Let me handle it separately.
                    pass
        # Handle 2*x_i = x_j as a pair constraint
        pair_block = [0] * n  # bitmask of elements that block i
        for i in range(n):
            s = 2 * S[i]
            if s in S:
                j = S.index(s)
                if i != j:
                    pair_block[i] |= (1 << j)
                    pair_block[j] |= (1 << i)

        # Build a value-to-index map for faster lookup
        val_to_idx = {v: i for i, v in enumerate(S)}

        # Redo triple computation with dict lookup
        pairs = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = S[i] + S[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    pairs[i].append((j, k))
                    pairs[j].append((i, k))
                    pairs[k].append((i, j))

        def can_add_bitmask(j, T):
            """Check if element j can be added to bitmask set T."""
            # Check pair blocks
            if T & pair_block[j]:
                return False
            # Check triple blocks
            for a, b in pairs[j]:
                if (T & (1 << a)) and (T & (1 << b)):
                    return False
            return True

        def greedy_build_bitmask(order_indices):
            """Greedy construction using bitmask."""
            T = 0
            for j in order_indices:
                if can_add_bitmask(j, T):
                    T |= (1 << j)
            return T

        def local_search_bitmask(T, max_rounds=100):
            """Local search: try to add elements that fit."""
            for _ in range(max_rounds):
                if time.time() > deadline:
                    break
                added = False
                for j in range(n):
                    if not (T & (1 << j)):
                        if can_add_bitmask(j, T):
                            T |= (1 << j)
                            added = True
                if not added:
                    break
            return T

        def bitmask_to_list(T):
            """Convert bitmask to list of elements."""
            return [S[j] for j in range(n) if (T >> j) & 1]

        def list_to_bitmask(lst):
            """Convert list of elements to bitmask."""
            T = 0
            for x in lst:
                T |= (1 << val_to_idx[x])
            return T

        best = 0

        # Strategy 1: All odd numbers
        odds_indices = [j for j in range(n) if S[j] % 2 != 0]
        T_odds = greedy_build_bitmask(odds_indices)
        best = max(best, T_odds, key=lambda T: bin(T).count('1'))

        # Strategy 2: Greedy by absolute value
        order_abs = sorted(range(n), key=lambda j: abs(S[j]))
        T_abs = greedy_build_bitmask(order_abs)
        if bin(T_abs).count('1') > bin(best).count('1'):
            best = T_abs

        # Strategy 3: Greedy by value ascending
        order_asc = list(range(n))  # S is sorted
        T_asc = greedy_build_bitmask(order_asc)
        if bin(T_asc).count('1') > bin(best).count('1'):
            best = T_asc

        # Strategy 4: Greedy by value descending
        order_desc = list(range(n - 1, -1, -1))
        T_desc = greedy_build_bitmask(order_desc)
        if bin(T_desc).count('1') > bin(best).count('1'):
            best = T_desc

        # Local search from each initial
        for T_init in [T_odds, T_abs, T_asc, T_desc]:
            if time.time() < deadline - 5:
                T_ls = local_search_bitmask(T_init, max_rounds=200)
                if bin(T_ls).count('1') > bin(best).count('1'):
                    best = T_ls

        # Random restarts
        rng = Random(seed)
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            p = 0.3 + rng.random() * 0.3
            random_indices = [j for j in range(n) if rng.random() < p]
            T_rand = greedy_build_bitmask(random_indices)
            T_ls = local_search_bitmask(T_rand, max_rounds=100)
            if bin(T_ls).count('1') > bin(best).count('1'):
                best = T_ls

        # Swap improvement: remove one, add up to two
        def swap_improve_bitmask(T, max_rounds=50):
            for _ in range(max_rounds):
                if time.time() > deadline:
                    break
                improved = False
                for rem in range(n):
                    if not (T & (1 << rem)):
                        continue
                    T_temp = T & ~(1 << rem)
                    added = 0
                    for j in range(n):
                        if added >= 2:
                            break
                        if not (T_temp & (1 << j)):
                            if can_add_bitmask(j, T_temp):
                                T_temp |= (1 << j)
                                added += 1
                    if added > 1:
                        T = T_temp
                        improved = True
                        break
                if not improved:
                    break
            return T

        if time.time() < deadline - 3:
            T_swapped = swap_improve_bitmask(best, max_rounds=30)
            if bin(T_swapped).count('1') > bin(best).count('1'):
                best = T_swapped
            if time.time() < deadline - 2:
                T_final = local_search_bitmask(best, max_rounds=100)
                if bin(T_final).count('1') > bin(best).count('1'):
                    best = T_final

        # 2-opt: remove 2, add up to 3
        def two_opt_bitmask(T, max_rounds=30):
            for _ in range(max_rounds):
                if time.time() > deadline:
                    break
                improved = False
                indices_in_T = [j for j in range(n) if (T >> j) & 1]
                for i in range(len(indices_in_T)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(indices_in_T)):
                        if time.time() > deadline:
                            break
                        a, b = indices_in_T[i], indices_in_T[j]
                        T_temp = T & ~(1 << a) & ~(1 << b)
                        added = 0
                        for k in range(n):
                            if added >= 3:
                                break
                            if not (T_temp & (1 << k)):
                                if can_add_bitmask(k, T_temp):
                                    T_temp |= (1 << k)
                                    added += 1
                        if added > 2:
                            T = T_temp
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return T

        if time.time() < deadline - 2:
            T_2opt = two_opt_bitmask(best, max_rounds=20)
            if bin(T_2opt).count('1') > bin(best).count('1'):
                best = T_2opt
            if time.time() < deadline - 1:
                T_final2 = local_search_bitmask(best, max_rounds=100)
                if bin(T_final2).count('1') > bin(best).count('1'):
                    best = T_final2

        # Perturbation: remove random elements, local search
        if time.time() < deadline - 5:
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                T_pert = best
                indices_in_T = [j for j in range(n) if (T_pert >> j) & 1]
                rng.shuffle(indices_in_T)
                remove_count = max(1, int(len(indices_in_T) * (0.1 + rng.random() * 0.1)))
                for j in indices_in_T[:remove_count]:
                    T_pert &= ~(1 << j)
                T_pert_ls = local_search_bitmask(T_pert, max_rounds=150)
                if bin(T_pert_ls).count('1') > bin(best).count('1'):
                    best = T_pert_ls

        return bitmask_to_list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            if not is_sum_free(result):
                result = [x for x in S if x % 2 != 0]
            results.append(result)

    return results
# EVOLVE-BLOCK-END
