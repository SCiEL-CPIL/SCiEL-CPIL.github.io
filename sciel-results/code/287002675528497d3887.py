import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 112.0  # Safety margin (evaluator kills at 150s, but keep conservative)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S):
        """Check if x can be added to sum-free set S."""
        if x + x in S:
            return False
        for a in S:
            if x + a in S:
                return False
            if x - a in S:
                return False
            if a - x in S:
                return False
        return True

    def greedy_add(V, S, order):
        """Add elements from V in given order to S if sum-free constraint holds."""
        S_set = set(S)
        added = []
        for x in order:
            if x in S_set:
                continue
            if can_add(x, S_set):
                S_set.add(x)
                added.append(x)
        return added

    def hill_climb(V, S, time_limit):
        """
        Local search: try to add any element, remove+swap, to grow the set.
        Uses deterministic ordering with randomization for restarts.
        """
        S = set(S)
        V_set = set(V)
        end_time = start_time + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            # Try adding each element not in S
            candidates = [x for x in V if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() >= end_time:
                    break
                if can_add(x, S):
                    S.add(x)
                    improved = True
            # Try swaps: remove a, add b
            if not improved:
                S_list = list(S)
                random.shuffle(S_list)
                for a in S_list:
                    if time.time() >= end_time:
                        break
                    S.remove(a)
                    S_after_remove = set(S)
                    # Try to add multiple elements
                    candidates = [x for x in V if x not in S_after_remove]
                    random.shuffle(candidates)
                    added_count = 0
                    for x in candidates:
                        if can_add(x, S_after_remove):
                            S_after_remove.add(x)
                            added_count += 1
                    if added_count > 1:
                        S = S_after_remove
                        improved = True
                    else:
                        S.add(a)
        return S

    def solve_instance(V, time_limit):
        """
        Solve one instance using multiple strategies:
        1. Greedy with different orderings
        2. Local search / hill climbing
        3. Randomized restarts
        """
        n = len(V)
        end_time = start_time + time_limit
        best_S = set()
        best_size = 0

        # Strategy 1: Multiple greedy initializations with different orderings
        strategies = [
            lambda: sorted(V),
            lambda: sorted(V, reverse=True),
            lambda: sorted(V, key=abs),
            lambda: sorted(V, key=lambda x: -abs(x)),
            lambda: list(random.sample(V, n)),
        ]

        for _ in range(8):
            if time.time() >= end_time:
                break
            # Pick a random strategy
            order = random.choice(strategies)()
            S = set()
            for x in order:
                if can_add(x, S):
                    S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # Strategy 2: Hill climbing from best greedy solution
        if time.time() < end_time - 1:
            S = hill_climb(V, best_S, min(2.0, time_limit * 0.1))
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # Strategy 3: Randomized local search with add/remove/swap moves
        # Use bitset representation for speed
        V_set = set(V)
        val_to_idx = {v: i for i, v in enumerate(V)}
        n = len(V)

        # Precompute conflict masks
        # conflict[i] = bitmask of j such that V[i] and V[j] "conflict" in the sense
        # that they cannot both be in a sum-free set with some third element.
        # Actually, for sum-free: a+b=c means all three can't be in S.
        # We'll precompute for each pair (i,j), whether V[i]+V[j] is in V (and which index).
        sum_idx = [[-1]*n for _ in range(n)]  # sum_idx[i][j] = k if V[i]+V[j]=V[k]
        for i in range(n):
            for j in range(n):
                s = V[i] + V[j]
                if s in val_to_idx:
                    sum_idx[i][j] = val_to_idx[s]

        # Now do a more aggressive local search
        best_mask = 0
        for i in best_S:
            best_mask |= (1 << val_to_idx[i])

        T = 5.0
        cooling = 0.999
        min_T = 0.1
        current_mask = best_mask

        while time.time() < end_time and T > min_T:
            # Generate a move
            move_type = random.random()
            new_mask = current_mask

            if move_type < 0.25:
                # Add an element
                # Find elements not in current_mask
                available = [i for i in range(n) if not (current_mask >> i) & 1]
                if available:
                    i = random.choice(available)
                    # Check if can add
                    can_add_i = True
                    V_i = V[i]
                    for j in range(n):
                        if (current_mask >> j) & 1:
                            V_j = V[j]
                            if (V_i + V_j) in val_to_idx:
                                k = val_to_idx[V_i + V_j]
                                if (current_mask >> k) & 1:
                                    can_add_i = False
                                    break
                            if (V_i - V_j) in val_to_idx:
                                k = val_to_idx[V_i - V_j]
                                if (current_mask >> k) & 1:
                                    can_add_i = False
                                    break
                            if (V_j - V_i) in val_to_idx:
                                k = val_to_idx[V_j - V_i]
                                if (current_mask >> k) & 1:
                                    can_add_i = False
                                    break
                    if can_add_i:
                        new_mask |= (1 << i)
            elif move_type < 0.5:
                # Remove an element
                in_set = [i for i in range(n) if (current_mask >> i) & 1]
                if in_set:
                    i = random.choice(in_set)
                    new_mask &= ~(1 << i)
            elif move_type < 0.8:
                # Swap: remove one, add one
                in_set = [i for i in range(n) if (current_mask >> i) & 1]
                out_set = [i for i in range(n) if not (current_mask >> i) & 1]
                if in_set and out_set:
                    i = random.choice(in_set)
                    j = random.choice(out_set)
                    new_mask &= ~(1 << i)
                    # Check if j can be added to new_mask
                    can_add_j = True
                    V_j = V[j]
                    for k in range(n):
                        if (new_mask >> k) & 1:
                            V_k = V[k]
                            if (V_j + V_k) in val_to_idx:
                                m = val_to_idx[V_j + V_k]
                                if (new_mask >> m) & 1:
                                    can_add_j = False
                                    break
                            if (V_j - V_k) in val_to_idx:
                                m = val_to_idx[V_j - V_k]
                                if (new_mask >> m) & 1:
                                    can_add_j = False
                                    break
                            if (V_k - V_j) in val_to_idx:
                                m = val_to_idx[V_k - V_j]
                                if (new_mask >> m) & 1:
                                    can_add_j = False
                                    break
                    if can_add_j:
                        new_mask |= (1 << j)
                    else:
                        new_mask |= (1 << i)  # revert
            else:
                # 2-swap: remove two, add two
                in_set = [i for i in range(n) if (current_mask >> i) & 1]
                out_set = [i for i in range(n) if not (current_mask >> i) & 1]
                if len(in_set) >= 2 and len(out_set) >= 2:
                    i1, i2 = random.sample(in_set, 2)
                    j1, j2 = random.sample(out_set, 2)
                    new_mask &= ~(1 << i1)
                    new_mask &= ~(1 << i2)
                    # Try to add j1 and j2
                    can_add = True
                    for j in [j1, j2]:
                        V_j = V[j]
                        for k in range(n):
                            if (new_mask >> k) & 1:
                                V_k = V[k]
                                if (V_j + V_k) in val_to_idx:
                                    m = val_to_idx[V_j + V_k]
                                    if (new_mask >> m) & 1:
                                        can_add = False
                                        break
                                if (V_j - V_k) in val_to_idx:
                                    m = val_to_idx[V_j - V_k]
                                    if (new_mask >> m) & 1:
                                        can_add = False
                                        break
                                if (V_k - V_j) in val_to_idx:
                                    m = val_to_idx[V_k - V_j]
                                    if (new_mask >> m) & 1:
                                        can_add = False
                                        break
                        if not can_add:
                            break
                    if can_add:
                        new_mask |= (1 << j1)
                        new_mask |= (1 << j2)
                    else:
                        new_mask |= (1 << i1)
                        new_mask |= (1 << i2)  # revert

            # Evaluate
            new_size = bin(new_mask).count('1')
            cur_size = bin(current_mask).count('1')

            if new_size > cur_size:
                current_mask = new_mask
                if new_size > best_size:
                    best_size = new_size
                    for i in range(n):
                        if (new_mask >> i) & 1:
                            best_S.add(V[i])
                        else:
                            best_S.discard(V[i])
                    best_S = set(best_S)
            elif new_size < cur_size:
                delta = new_size - cur_size
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_mask = new_mask
            # else: same size, accept with some probability
            else:
                if random.random() < 0.1:
                    current_mask = new_mask

            T *= cooling

        return list(best_S)

    results = []

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)

        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 1:
            # Emergency fallback: greedy
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            results.append(list(S))
            continue

        # Allocate time based on instance size
        if size <= 60:
            t_alloc = min(25.0, remaining * 0.35)
        elif size <= 120:
            t_alloc = min(45.0, remaining * 0.35)
        else:
            t_alloc = min(85.0, remaining * 0.55)

        # Run multiple restarts
        best_res = []
        best_len = 0
        restarts = 0
        num_restarts = 4 if size <= 60 else (3 if size <= 120 else 2)
        per_restart = t_alloc / num_restarts

        for r in range(num_restarts):
            if time.time() >= start_time + DEADLINE - 3:
                break
            res = solve_instance(V, per_restart)
            if len(res) > best_len:
                best_len = len(res)
                best_res = res
            restarts += 1

        # Verify
        if not is_sum_free(best_res):
            S = set()
            for x in V:
                if can_add(x, S):
                    S.add(x)
            best_res = list(S)

        results.append(best_res)

    return results
