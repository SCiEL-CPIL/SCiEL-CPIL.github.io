import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

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

    # Step 3: Build compatibility graph
    neighbors = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= N:
                neighbors[i].add(j)
                neighbors[j].add(i)
        if time.time() > deadline:
            break

    if time.time() > deadline:
        return list(best_clique)

    # Verify best_clique is a clique
    for a in best_clique:
        for b in best_clique:
            if a < b and b not in neighbors[a]:
                return get_divisors(best_m)

    # Step 4: Bron-Kerbosch with pivot
    best_size = len(best_clique)
    best_set = set(best_clique)

    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set

        if time.time() > deadline:
            return

        if len(R) > best_size:
            best_size = len(R)
            best_set = set(R)

        if not P:
            return

        if len(R) + len(P) <= best_size:
            return

        # Pivot selection: maximize |P ∩ N(u)| over u in P ∪ X
        pivot_candidates = P | X
        if not pivot_candidates:
            return

        pivot = None
        max_count = -1
        for u in pivot_candidates:
            count = len(P & neighbors[u])
            if count > max_count:
                max_count = count
                pivot = u

        if pivot is None:
            return

        candidates = P - neighbors[pivot]

        for v in list(candidates):
            R.add(v)
            bron_kerbosch(R, P & neighbors[v], X & neighbors[v])
            R.remove(v)
            P.remove(v)
            X.add(v)

            if time.time() > deadline:
                return

    # Run BK
    R = set()
    P = set(range(1, N + 1))
    X = set()
    bron_kerbosch(R, P, X)

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
        return get_divisors(best_m)

    return result
