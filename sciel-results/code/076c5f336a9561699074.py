import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 512

    # Step 1: Build compatibility graph using bitsets
    # compat[i] is a bitmask where bit j is set if lcm(i,j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l <= N:
                mask |= (1 << j)
        compat[i] = mask
        if time.time() > deadline:
            break

    if time.time() > deadline:
        return list(range(1, N + 1))

    # Step 2: Greedy construction of initial clique
    def get_divisors(m):
        divs = []
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
            i += 1
        return divs

    # Find M with maximum divisors for initial clique
    best_m = 1
    max_divs = 0
    for m in range(1, N + 1):
        d = get_divisors(m)
        if len(d) > max_divs:
            max_divs = len(d)
            best_m = m

    best_clique = set(get_divisors(best_m))

    # Greedy improvement
    candidates = [x for x in range(1, N + 1) if x not in best_clique]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates.sort(key=lambda x: sum(1 for y in best_clique if (x * y) // gcd(x, y) > N))
        for x in list(candidates):
            if x in best_clique:
                continue
            valid = True
            for y in best_clique:
                if (x * y) // gcd(x, y) > N:
                    valid = False
                    break
            if valid:
                best_clique.add(x)
                candidates.remove(x)
                changed = True

    # Step 3: Bron-Kerbosch with pivot using bitsets
    best_size = len(best_clique)
    best_set = set(best_clique)

    def popcount(x):
        return bin(x).count('1')

    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        r_size = len(R)
        p_size = popcount(P)

        if r_size > best_size:
            best_size = r_size
            best_set = set(R)

        if p_size == 0:
            return

        if r_size + p_size <= best_size:
            return

        # Pivot selection: maximize |P ∩ N(u)| over u in P ∪ X
        pivot_candidates = P | X
        if pivot_candidates == 0:
            return

        pivot = None
        max_count = -1
        pc = pivot_candidates
        while pc:
            lsb = pc & (-pc)
            u = lsb.bit_length() - 1
            count = popcount(P & compat[u])
            if count > max_count:
                max_count = count
                pivot = u
            pc ^= lsb

        if pivot is None:
            return

        candidates = P & ~compat[pivot]

        c = candidates
        while c:
            lsb = c & (-c)
            v = lsb.bit_length() - 1
            R.add(v)
            bron_kerbosch(R, P & compat[v], X & compat[v])
            R.remove(v)
            P &= ~lsb
            X |= lsb
            c ^= lsb

            if time.time() > deadline:
                return

    # Run BK with initial clique as starting point
    R = set(best_clique)
    P = 0
    for i in range(1, N + 1):
        if i not in best_clique:
            P |= (1 << i)
    X = 0
    bron_kerbosch(R, P, X)

    # Step 4: Local search polish - try to add elements by removing blockers
    if time.time() < deadline:
        current_set = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                # Find blockers
                blockers = [y for y in current_set if (x * y) // gcd(x, y) > N]
                if len(blockers) <= 2:
                    # Try removing 1 or 2 blockers and adding x
                    new_set = set(current_set)
                    for b in blockers:
                        new_set.discard(b)
                    new_set.add(x)
                    if len(new_set) > len(current_set):
                        # Verify validity
                        valid = True
                        for i in range(len(new_set)):
                            for j in range(i + 1, len(new_set)):
                                a, b = list(new_set)[i], list(new_set)[j]
                                if (a * b) // gcd(a, b) > N:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            current_set = new_set
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = set(current_set)
                            improved = True

    # Final validation
    result = list(best_set)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) // gcd(a, b) > N:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return list(best_clique)

    return result
