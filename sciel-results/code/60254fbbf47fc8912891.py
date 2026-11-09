import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set

    # Precompute LCM table for O(1) lookup
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            lcm_table[a][b] = a * b // gcd(a, b)
            lcm_table[b][a] = lcm_table[a][b]

    # Precompute conflict masks: conflict[a][b] is a bitmask of all c such that
    # {a, b, c} forms a bad triple (all three pairwise LCMs equal)
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]

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
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0

    # Build adjacency for conflict graph
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))

    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w

    # Fast greedy with bitmask
    def weighted_greedy(order):
        S = []
        S_mask = 0
        for x in order:
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S

    # Try several greedy orderings
    best_set = []
    best_size = 0

    # Natural order
    S = weighted_greedy(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Reverse order
    S = weighted_greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(range(1, N + 1), key=lambda x: conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Sort by conflict score (descending)
    order = sorted(range(1, N + 1), key=lambda x: -conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    order = sorted(range(1, N + 1), key=num_prime_factors)
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Spectral ordering: use power iteration on (D - W) to get ranking
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()

    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm

    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]

        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v

    spectral_order = sorted(range(1, N + 1), key=lambda x: -v[x])
    S = weighted_greedy(spectral_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    spectral_order_asc = sorted(range(1, N + 1), key=lambda x: v[x])
    S = weighted_greedy(spectral_order_asc)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Primes and prime powers first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    primes = [x for x in range(1, N + 1) if is_prime(x)]
    prime_powers = [x for x in range(1, N + 1) if is_prime_power(x) and x not in primes]
    others = [x for x in range(1, N + 1) if x not in primes and x not in prime_powers]

    order_pp = primes + prime_powers + others
    S = weighted_greedy(order_pp)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # Local search polish: try to improve best_set by swapping or adding elements
    def polish(S):
        S = S[:]
        S_set = set(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)

        improved = True
        while improved and time.time() - start_time < deadline - 1:
            improved = False
            # Try adding each excluded element
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                compatible = True
                for s in S:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    S.append(x)
                    S_set.add(x)
                    S_mask |= (1 << x)
                    improved = True

            # Try swapping: remove one element, add one that was blocked
            for i in range(len(S)):
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    # Remove S[i]
                    removed = S[i]
                    S_set.remove(removed)
                    S_mask ^= (1 << removed)
                    S[i] = x
                    S_set.add(x)
                    S_mask |= (1 << x)

                    # Check if valid
                    valid = True
                    for j in range(len(S)):
                        if j == i:
                            continue
                        if conflict[S[j]][x] & S_mask:
                            valid = False
                            break

                    if valid:
                        improved = True
                    else:
                        # Revert
                        S_set.remove(x)
                        S_mask ^= (1 << x)
                        S[i] = removed
                        S_set.add(removed)
                        S_mask |= (1 << removed)

        return S

    polished = polish(best_set)
    if len(polished) > best_size:
        best_size = len(polished)
        best_set = polished

    # Simulated annealing to escape local optima
    def simulated_annealing(initial_set, time_budget=30.0):
        """
        Treat the set as a binary vector. At each step, either:
        - Remove a random element (to explore)
        - Add a random compatible element (to improve)
        - Swap: remove one, try to add another
        Accept moves probabilistically based on energy (number of conflicts avoided).
        """
        random.seed(12345)
        S = initial_set[:]
        S_set = set(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)

        def energy(s_set):
            """Energy = number of bad triples within the set (should be 0 for valid sets)"""
            # For a valid set, energy is always 0. We use a surrogate:
            # energy = -|S| (we want to maximize size)
            return -len(s_set)

        current_energy = energy(S)
        best_energy = current_energy
        best_S = S[:]

        T = 10.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.995
        max_steps = 500000

        step = 0
        while time.time() - start_time < deadline - time_budget and step < max_steps:
            step += 1

            # Choose a move
            move_type = random.random()
            new_S = S[:]
            new_S_set = S_set.copy()
            new_S_mask = S_mask

            if move_type < 0.4:
                # Remove a random element
                if not S:
                    continue
                idx = random.randint(0, len(S) - 1)
                x = S[idx]
                new_S_set.discard(x)
                new_S_mask &= ~(1 << x)
                new_S.pop(idx)
            elif move_type < 0.7:
                # Add a random compatible element
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)
                # Check if x is compatible
                compatible = True
                for s in S:
                    if conflict[s][x] & S_mask:
                        compatible = False
                        break
                if compatible:
                    new_S_set.add(x)
                    new_S_mask |= (1 << x)
                    new_S.append(x)
                else:
                    # Force add (will create conflicts, but SA can handle)
                    new_S_set.add(x)
                    new_S_mask |= (1 << x)
                    new_S.append(x)
            else:
                # Swap: remove one, add one
                if not S:
                    continue
                idx = random.randint(0, len(S) - 1)
                removed = S[idx]
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                if not candidates:
                    continue
                x = random.choice(candidates)

                new_S_set.discard(removed)
                new_S_mask &= ~(1 << removed)
                new_S_set.add(x)
                new_S_mask |= (1 << x)
                new_S[idx] = x

            new_energy = energy(new_S)
            delta_e = new_energy - current_energy

            # Accept if better or with probability exp(-delta_e / T)
            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                S = new_S
                S_set = new_S_set
                S_mask = new_S_mask
                current_energy = new_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_S = S[:]

            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_S

    # Run simulated annealing from best_set
    if time.time() - start_time < deadline - 25:
        sa_result = simulated_annealing(best_set, time_budget=25.0)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result[:]

    # Polish the SA result
    if time.time() - start_time < deadline - 2:
        polished = polish(best_set)
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = polished

    # Branch and bound with improved upper bound
    def solve(selected, available):
        nonlocal best_size, best_set

        if time.time() - start_time > deadline:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]

        if not available:
            return

        avail_count = available.bit_count()
        if len(selected) + avail_count <= best_size:
            return

        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low

        if not avail_list:
            return

        # Pick element with highest conflict degree in available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x

        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]

        solve(new_selected, new_avail)

        if time.time() - start_time > deadline:
            return

        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))

    # Start search with remaining time
    if time.time() - start_time < deadline - 5:
        solve([], full_mask)

    # Final polish
    polished = polish(best_set)
    if len(polished) > best_size:
        best_size = len(polished)
        best_set = polished

    return best_set
