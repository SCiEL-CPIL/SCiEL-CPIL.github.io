def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 128

    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict structure:
    # For each x, find all pairs (a,b) with a<b, a!=x, b!=x such that
    # gcd(x,a) == gcd(x,b) == gcd(a,b)
    CONFLICT = {x: [] for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if gcd_table[x][b] == g and gcd_table[a][b] == g:
                    CONFLICT[x].append((a, b))

    def is_valid_set(s):
        """Check if set s has no three distinct elements with all pairwise gcds equal."""
        elems = sorted(s)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, n):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    def count_bad_triples(s):
        """Count number of bad triples in set s."""
        elems = sorted(s)
        n = len(elems)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, n):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        count += 1
        return count

    def can_add(x, s):
        """Check if x can be added to set s without creating a bad triple."""
        for a, b in CONFLICT[x]:
            if a in s and b in s:
                return False
        return True

    def swap_valid(s, r, x):
        """Check if s - {r} + {x} is valid."""
        new_s = s - {r} | {x}
        return is_valid_set(new_s)

    # Strategy: Multiple greedy initializations with different orderings,
    # then simulated annealing local search with conflict-based can_add

    # Ordering 1: numbers with large prime factors first
    def largest_prime_factor(n):
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp

    # Ordering 2: primes first
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]

    orderings = [
        sorted(range(1, 129), key=lambda x: (-largest_prime_factor(x), x)),
        sorted(range(1, 129), key=lambda x: (0 if x in primes else 1, x)),
        list(range(1, 129)),
        list(range(128, 0, -1)),
    ]

    # Add several random orderings
    random.seed(42)
    for _ in range(10):
        order = list(range(1, 129))
        random.shuffle(order)
        orderings.append(order)

    best_set = set()
    best_size = 0

    # Greedy construction with multiple orderings
    for order in orderings:
        if elapsed() > DEADLINE - 30:
            break
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        if len(s) > best_size:
            best_set = s.copy()
            best_size = len(s)

    # Simulated annealing local search
    current_set = set(best_set)
    current_bad = 0  # should be 0 since we started from valid sets

    T = 5.0
    T_min = 0.01
    cooling = 0.9995

    while elapsed() < DEADLINE - 5:
        # Try random moves
        move_type = random.random()

        if move_type < 0.3:
            # Try to add an element
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add(x, current_set):
                current_set.add(x)
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
            else:
                # Allow adding with some probability (accepting violations temporarily)
                new_set = current_set | {x}
                new_bad = count_bad_triples(new_set)
                if new_bad <= current_bad or random.random() < __import__('math').exp(-new_bad / T):
                    current_set = new_set
                    current_bad = new_bad

        elif move_type < 0.6:
            # Try to remove an element
            if not current_set:
                continue
            x = random.choice(list(current_set))
            current_set.remove(x)
            current_bad = max(0, current_bad - 1)  # approximate

        else:
            # Try 1-for-1 swap
            in_set = list(current_set)
            out_set = [x for x in range(1, N + 1) if x not in current_set]
            if not in_set or not out_set:
                continue
            x_out = random.choice(in_set)
            x_in = random.choice(out_set)
            new_set = (current_set - {x_out}) | {x_in}
            if is_valid_set(new_set):
                current_set = new_set
                current_bad = 0
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
            else:
                new_bad = count_bad_triples(new_set)
                if new_bad <= current_bad or random.random() < __import__('math').exp(-new_bad / T):
                    current_set = new_set
                    current_bad = new_bad

        T *= cooling
        if T < T_min:
            T = T_min

    # Final cleanup: ensure best_set is valid and try to add more
    if elapsed() < DEADLINE:
        # Verify best_set is valid
        if not is_valid_set(best_set):
            # Remove elements until valid
            s = set(best_set)
            while s and not is_valid_set(s):
                # Find a bad triple and remove one element
                elems = sorted(s)
                n = len(elems)
                removed = False
                for i in range(n):
                    for j in range(i + 1, n):
                        g_ij = gcd_table[elems[i]][elems[j]]
                        for k in range(j + 1, n):
                            if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                                s.remove(elems[i])
                                removed = True
                                break
                        if removed:
                            break
                    if removed:
                        break
                if not removed:
                    break
            best_set = s

        # Try to add more elements
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            for x in range(1, 129):
                if x in best_set:
                    continue
                if can_add(x, best_set):
                    best_set.add(x)
                    improved = True
                    break

    if not best_set:
        best_set = {1}

    return list(best_set)
