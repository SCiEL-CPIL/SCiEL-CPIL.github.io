import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 512

    # Step 1: Find M with maximum divisors -> initial clique
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

    best_m = 1
    max_divs = 0
    for m in range(1, N + 1):
        d = get_divisors(m)
        if len(d) > max_divs:
            max_divs = len(d)
            best_m = m

    best_clique = set(get_divisors(best_m))

    # Step 2: Greedy improvement
    def conflicts_with_set(x, S):
        count = 0
        for y in S:
            if (x * y) // gcd(x, y) > N:
                count += 1
        return count

    candidates = [x for x in range(1, N + 1) if x not in best_clique]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates.sort(key=lambda x: conflicts_with_set(x, best_clique))
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

    # Step 3: Build compatibility graph using bitsets
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
        return list(best_clique)

    # Verify best_clique is a clique using bitsets
    for a in best_clique:
        for b in best_clique:
            if a != b and not (compat[a] >> b) & 1:
                return get_divisors(best_m)

    # Step 4: Bron-Kerbosch with pivot using bitsets
    best_size = len(best_clique)
    best_set = set(best_clique)

    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        r_size = len(R)
        p_size = P.bit_count()

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
            count = (P & compat[u]).bit_count()
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

    # Run BK
    R = set()
    P = 0
    for i in range(1, N + 1):
        P |= (1 << i)
    X = 0
    bron_kerbosch(R, P, X)

    # Step 5: Local search polish - try to improve by swapping elements
    result_set = set(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each element not in result_set
        for x in range(1, N + 1):
            if x in result_set:
                continue
            # Check if x is compatible with all elements in result_set
            can_add = True
            for y in result_set:
                if not (compat[x] >> y) & 1:
                    can_add = False
                    break
            if can_add:
                result_set.add(x)
                improved = True
                break

        if not improved:
            # Try swapping: remove one element, add two
            for a in list(result_set):
                for x in range(1, N + 1):
                    if x in result_set or x == a:
                        continue
                    # Check if x is compatible with all elements except a
                    can_add_x = True
                    for y in result_set:
                        if y == a:
                            continue
                        if not (compat[x] >> y) & 1:
                            can_add_x = False
                            break
                    if can_add_x:
                        # Check if there's another element z we can also add
                        for z in range(1, N + 1):
                            if z in result_set or z == a or z == x:
                                continue
                            # Check if z is compatible with all elements except a
                            can_add_z = True
                            for y in result_set:
                                if y == a:
                                    continue
                                if not (compat[z] >> y) & 1:
                                    can_add_z = False
                                    break
                            if can_add_z and (compat[x] >> z) & 1:
                                result_set.remove(a)
                                result_set.add(x)
                                result_set.add(z)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break

    # Final validation
    result = list(result_set)
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
        return get_divisors(best_m)

    return result
