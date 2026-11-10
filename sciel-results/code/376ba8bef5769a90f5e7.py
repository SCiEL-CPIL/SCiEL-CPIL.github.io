import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    N = 180

    # Precompute forbidden triples: (a,b,c) with a<b, a<c, (b-a)(c-a)=a^2
    # For each a, find all (b,c)
    triples_by_root = {}
    all_triples = []
    for a in range(1, N+1):
        a2 = a*a
        # divisors d of a2, d < a (so d < a2/d, giving b=a+d < c=a+a2/d)
        pairs = []
        for d in range(1, a):
            if a2 % d == 0:
                e = a2 // d  # e = a2/d > a
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c:
                    pairs.append((b,c))
        if pairs:
            triples_by_root[a] = pairs
            for (b,c) in pairs:
                all_triples.append((a,b,c))

    # A number is a "root" if it's the smallest element of some forbidden triple
    roots = set(triples_by_root.keys())

    # Strategy 1: all non-roots (guaranteed valid)
    base = set(range(1, N+1)) - roots

    def is_valid(S):
        Sset = set(S)
        for (a,b,c) in all_triples:
            if a in Sset and b in Sset and c in Sset:
                return False
        return True

    def conflicts_with(S, x):
        """Return set of elements in S that form a forbidden triple with x."""
        Sset = set(S)
        conf = set()
        for (a,b,c) in all_triples:
            # x must be one of a,b,c
            if x == a:
                if b in Sset and c in Sset:
                    conf.add(b); conf.add(c)
            elif x == b:
                if a in Sset and c in Sset:
                    conf.add(a); conf.add(c)
            elif x == c:
                if a in Sset and b in Sset:
                    conf.add(a); conf.add(b)
        return conf

    def is_safe(S, x):
        Sset = set(S)
        for (a,b,c) in all_triples:
            if x == a and b in Sset and c in Sset:
                return False
            if x == b and a in Sset and c in Sset:
                return False
            if x == c and a in Sset and b in Sset:
                return False
        return True

    # Start from base (all non-roots), which is valid
    S = set(base)

    # Try to add roots greedily (smallest conflicts first)
    root_list = sorted(roots)
    for x in root_list:
        if time.time() > deadline:
            break
        if is_safe(S, x):
            S.add(x)

    # Local search: swap - remove one element, add more
    improved = True
    best_S = set(S)
    best_size = len(S)
    while improved and time.time() < deadline:
        improved = False
        for rem in sorted(best_S):
            if time.time() > deadline:
                break
            S2 = best_S - {rem}
            # greedily add all safe
            for x in range(1, N+1):
                if time.time() > deadline:
                    break
                if x not in S2 and is_safe(S2, x):
                    S2.add(x)
            if len(S2) > best_size:
                best_S = S2
                best_size = len(S2)
                improved = True

    # Random restarts with perturbation
    for _ in range(200):
        if time.time() > deadline - 3:
            break
        S2 = set(best_S)
        k = random.randint(2, 6)
        for x in random.sample(sorted(S2), min(k, len(S2))):
            S2.discard(x)
        for x in range(1, N+1):
            if x not in S2 and is_safe(S2, x):
                S2.add(x)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)

    return list(best_S)
