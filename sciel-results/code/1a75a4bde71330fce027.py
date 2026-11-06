import time, random, math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 85.0
    N = 180

    # Build all triples (a<b<c) with 1/a = 1/b + 1/c <=> (b-a)(c-a)=a^2
    triples = []
    seen = set()
    for a in range(1, N + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d != e:
                    b = a + d
                    c = a + e
                    if b <= N and c <= N and b != c:
                        t = tuple(sorted((a, b, c)))
                        if t not in seen:
                            seen.add(t)
                            triples.append(t)
            d += 1

    # Build adjacency: for each vertex, list of (other1, other2) pairs in triples
    adj = [[] for _ in range(N + 1)]
    for (a, b, c) in triples:
        adj[a].append((b, c))
        adj[b].append((a, c))
        adj[c].append((a, b))

    def count_active(p):
        cnt = 0
        for (a, b, c) in triples:
            if p[a] and p[b] and p[c]:
                cnt += 1
        return cnt

    def can_add(p, v):
        """Check if we can add vertex v to present set p without creating an active triple."""
        for (u, w) in adj[v]:
            if p[u] and p[w]:
                return False
        return True

    def greedy_grow(p):
        """Greedily add elements that don't create conflicts, prioritizing those with fewest conflicts."""
        p = p[:]
        improved = True
        while improved:
            improved = False
            # Try to add elements, prioritize those with fewer conflicts
            candidates = []
            for v in range(1, N + 1):
                if not p[v]:
                    if can_add(p, v):
                        # Count how many potential conflicts this would create
                        conflicts = 0
                        for (u, w) in adj[v]:
                            if p[u] or p[w]:
                                conflicts += 1
                        candidates.append((conflicts, v))
            if candidates:
                candidates.sort()
                # Add the best candidate
                _, v = candidates[0]
                p[v] = True
                improved = True
        return p

    def local_search(p, max_iter=10000):
        """Local search: try single flips and double flips to improve size while maintaining validity."""
        p = p[:]
        cur_active = count_active(p)
        cur_size = sum(p[1:])
        best_p = p[:]
        best_size = cur_size
        best_active = cur_active

        # Single flip moves
        for iteration in range(max_iter):
            if time.time() > DEADLINE:
                break
            improved = False
            # Try adding
            for v in range(1, N + 1):
                if not p[v] and can_add(p, v):
                    p[v] = True
                    cur_size += 1
                    improved = True
            # Try removing elements that are in active triples to free up space
            if cur_active > 0:
                for (a, b, c) in triples:
                    if p[a] and p[b] and p[c]:
                        # Remove the element with the most connections to try to break more triples
                        scores = {}
                        for v in [a, b, c]:
                            scores[v] = len(adj[v])
                        v = min([a, b, c], key=lambda x: scores[x])
                        # Actually, let's remove the one that breaks the most active triples
                        for v in [a, b, c]:
                            break_count = 0
                            for (x, y) in adj[v]:
                                if p[x] and p[y]:
                                    break_count += 1
                            scores[v] = break_count
                        v = max([a, b, c], key=lambda x: scores[x])
                        if p[v]:
                            p[v] = False
                            cur_size -= 1
                            cur_active = count_active(p)
                            improved = True
                            break
            if not improved:
                break
            if cur_active == 0 and cur_size > best_size:
                best_size = cur_size
                best_p = p[:]
                best_active = cur_active

        return best_p, best_size

    def simulated_annealing(p, T_init=5.0, T_min=0.1, cool=0.999, max_steps=50000):
        """Simulated annealing to find larger valid sets."""
        p = p[:]
        cur_active = count_active(p)
        cur_size = sum(p[1:])
        best_p = p[:]
        best_size = cur_size
        best_active = cur_active
        T = T_init

        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            if T < T_min:
                break

            v = random.randint(1, N)
            old = p[v]

            # Calculate change in active triples
            dact = 0
            for (u, w) in adj[v]:
                if p[u] and p[w]:
                    if not old:
                        dact += 1
                    else:
                        dact -= 1

            dsize = 1 if not old else -1
            new_size = cur_size + dsize
            new_active = cur_active + dact

            # We prefer valid sets (active=0) and larger size
            # Energy: -size + big_penalty * active
            old_energy = -cur_size + 10000 * cur_active
            new_energy = -new_size + 10000 * new_active
            delta_e = new_energy - old_energy

            if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
                p[v] = not old
                cur_active = new_active
                cur_size = new_size
                if cur_active == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_p = p[:]
                    best_active = cur_active

            T *= cool

        return best_p, best_size

    # Multiple strategies
    best_set = None
    best_size = 0

    # Strategy 1: Start from upper half and grow
    p0 = [False] * (N + 1)
    for v in range(91, N + 1):
        p0[v] = True
    p1 = greedy_grow(p0)
    if count_active(p1) == 0:
        if sum(p1[1:]) > best_size:
            best_size = sum(p1[1:])
            best_set = p1[:]

    # Strategy 2: Start from a different range
    for lo in [80, 85, 90, 95, 100]:
        p0 = [False] * (N + 1)
        for v in range(lo, N + 1):
            p0[v] = True
        p1 = greedy_grow(p0)
        if count_active(p1) == 0 and sum(p1[1:]) > best_size:
            best_size = sum(p1[1:])
            best_set = p1[:]

    # Strategy 3: Random initial sets + local search + SA
    random.seed(42)
    for restart in range(30):
        if time.time() > DEADLINE:
            break

        # Random initial set
        p0 = [False] * (N + 1)
        for v in range(1, N + 1):
            if random.random() < 0.55:
                p0[v] = True

        # Greedy cleanup: remove elements in active triples
        p1 = p0[:]
        improved = True
        while improved:
            improved = False
            for (a, b, c) in triples:
                if p1[a] and p1[b] and p1[c]:
                    # Remove one
                    v = random.choice([a, b, c])
                    p1[v] = False
                    improved = True

        # Greedy grow
        p1 = greedy_grow(p1)

        if count_active(p1) == 0 and sum(p1[1:]) > best_size:
            best_size = sum(p1[1:])
            best_set = p1[:]

        # Local search
        p2, sz2 = local_search(p1, max_iter=2000)
        if count_active(p2) == 0 and sz2 > best_size:
            best_size = sz2
            best_set = p2[:]

        # Simulated annealing
        p3, sz3 = simulated_annealing(p2, T_init=3.0, T_min=0.05, cool=0.9995, max_steps=30000)
        if count_active(p3) == 0 and sz3 > best_size:
            best_size = sz3
            best_set = p3[:]

    # Final greedy grow on best set
    if best_set is not None and time.time() < DEADLINE:
        p_final = greedy_grow(best_set)
        if count_active(p_final) == 0 and sum(p_final[1:]) > best_size:
            best_size = sum(p_final[1:])
            best_set = p_final[:]

    # Convert to list
    if best_set is None:
        res = list(range(91, N + 1))
    else:
        res = [v for v in range(1, N + 1) if best_set[v]]

    if not res:
        res = list(range(91, N + 1))

    return res
