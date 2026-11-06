def construct():
    import time
    import random
    from math import gcd

    deadline = time.time() + 80  # safety margin

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

    def is_valid_triple_free(selected_set):
        """Check if selected_set has no bad triple. Returns True if valid."""
        elems = sorted(selected_set)
        n = len(elems)
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, n):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        return False
        return True

    def count_bad_triples(selected_set):
        """Count number of bad triples in selected_set."""
        elems = sorted(selected_set)
        n = len(elems)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[elems[i]][elems[j]]
                for k in range(j + 1, n):
                    if gcd_table[elems[i]][elems[k]] == g_ij and gcd_table[elems[j]][elems[k]] == g_ij:
                        count += 1
        return count

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a, b in CONFLICT[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    def swap_valid(selected_set, r, x):
        """Check if selected_set - {r} + {x} is valid."""
        temp_set = selected_set - {r}
        for a, b in CONFLICT[x]:
            if a in temp_set and b in temp_set:
                return False
        return True

    # Greedy initialization
    random.seed(42)
    order = list(range(1, N + 1))
    random.shuffle(order)

    best_set = set()
    for x in order:
        if can_add(x, best_set):
            best_set.add(x)

    best_size = len(best_set)

    # Local search with simulated annealing
    current_set = set(best_set)
    current_bad = count_bad_triples(current_set)  # should be 0

    T = 5.0
    T_min = 0.01
    cooling = 0.999

    for iteration in range(500000):
        if time.time() > deadline:
            break

        # Try random moves
        move_type = random.random()

        if move_type < 0.3:
            # Try to add an element
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_set = current_set | {x}
            new_bad = count_bad_triples(new_set)
            old_bad = current_bad
            if new_bad <= old_bad:
                current_set = new_set
                current_bad = new_bad
                if len(current_set) > best_size and is_valid_triple_free(current_set):
                    best_set = set(current_set)
                    best_size = len(current_set)
            else:
                delta = new_bad - old_bad
                if random.random() < __import__('math').exp(-delta / T):
                    current_set = new_set
                    current_bad = new_bad

        elif move_type < 0.6:
            # Try to remove an element
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            new_bad = count_bad_triples(new_set)
            old_bad = current_bad
            if new_bad <= old_bad:
                current_set = new_set
                current_bad = new_bad
            else:
                delta = new_bad - old_bad
                if random.random() < __import__('math').exp(-delta / T):
                    current_set = new_set
                    current_bad = new_bad

        else:
            # Try 1-for-1 swap
            in_set = list(current_set)
            out_set = [x for x in range(1, N + 1) if x not in current_set]
            if not in_set or not out_set:
                continue
            x_out = random.choice(in_set)
            x_in = random.choice(out_set)
            new_set = (current_set - {x_out}) | {x_in}
            new_bad = count_bad_triples(new_set)
            old_bad = current_bad
            if new_bad <= old_bad:
                current_set = new_set
                current_bad = new_bad
                if len(current_set) > best_size and is_valid_triple_free(current_set):
                    best_set = set(current_set)
                    best_size = len(current_set)
            else:
                delta = new_bad - old_bad
                if random.random() < __import__('math').exp(-delta / T):
                    current_set = new_set
                    current_bad = new_bad

        T *= cooling
        if T < T_min:
            T = T_min

    # Final check
    if is_valid_triple_free(best_set):
        return list(best_set)
    else:
        # Fallback: return a known valid small set
        return [1]
