import time
import random
import math
from collections import deque
import sys

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def precompute_inverses(q):
        inv = [0] * q
        if q > 1:
            inv[1] = 1
            for i in range(2, q):
                inv[i] = (q - (q // i) * inv[q % i]) % q
        return inv

    def precompute_quadratic_residues(q):
        residues = [False] * q
        residues[0] = True
        for x in range(1, q):
            residues[(x * x) % q] = True
        return residues

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:  # g = 2r + 1
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:  # g = 2r
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- graph validation ----------
    def check_girth_bfs(adj, n, g):
        """Optimized BFS from each vertex, checking for cycles shorter than g."""
        half_g = g // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = deque([u])
            while queue:
                x = queue.popleft()
                dx = dist[x]
                if dx >= half_g:
                    continue
                for y in adj[x]:
                    if dist[y] == -1:
                        dist[y] = dx + 1
                        parent[y] = x
                        if dist[y] < g:
                            queue.append(y)
                    else:
                        # If y is already visited and not the parent of x, 
                        # and x is not the parent of y, we found a cycle.
                        if y != parent[x]:
                            cycle_len = dx + dist[y] + 1
                            if cycle_len < g:
                                return False
        return True

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
            if i in adj[i]:
                return False
            if len(adj[i]) != len(set(adj[i])):
                return False
        # Connectivity
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if count != n:
            return False
        # Girth
        if not check_girth_bfs(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q, inv_table):
        if x == q:
            if c == 0:
                return q
            return (a * inv_table[c]) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        return (num * inv_table[den]) % q

    def is_derangement(a, b, c, d, q, is_res_table):
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_res_table[disc]:
            return False
        return True

    def build_from_gens(gens, q, k, g, inv_table, is_res_table):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q, inv_table)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                if j in adj[i]:
                    return None # Duplicate edge
                adj[i].add(j)
                adj[j].add(i)
        return validate(adj, n, k, g)

    # ---------- Enumerate involutory derangements ----------
    def enumerate_involutions(q, is_res_table):
        """
        Enumerates all involutory derangements in PGL(2, q).
        An involution in PGL(2, q) is represented by (a, b, c, -a) with a^2 - bc being a non-square.
        It is a derangement if it has no fixed points, which means the discriminant of the fixed point equation is a non-square.
        For (a, b, c, -a), the fixed points satisfy cx^2 + (2a)x - b = 0? No, (ax+b)/(cx-a) = x => cx^2 - 2ax - b = 0.
        Discriminant D = 4a^2 + 4bc = 4(a^2 + bc).
        So we need a^2 - bc to be a non-square (for involution) and a^2 + bc to be a non-square (for derangement).
        """
        invols = []
        for a in range(q):
            for b in range(q):
                for c in range(1, q):
                    val1 = (a*a - b*c) % q
                    if is_res_table[val1]:
                        continue
                    val2 = (a*a + b*c) % q
                    if is_res_table[val2]:
                        continue
                    # Ensure we don't add duplicates due to scaling? 
                    # PGL(2,q) matrices are projective. (a,b,c,-a) is equivalent to (ka, kb, kc, -ka).
                    # To get unique representatives, we can fix c=1 if possible? 
                    # Or just check if it's a derangement and involution.
                    # Actually, the condition for involution is trace^2 - 4det = 0? 
                    # In PGL, trace is not well-defined. 
                    # Let's stick to the property: M^2 = lambda I.
                    # (a b; c -a)^2 = (a^2+bc, 0; 0, a^2+bc). So it is an involution if a^2+bc != 0? No, it's a scalar matrix if a^2+bc is a square? 
                    # Wait, in PGL, M is an involution if M^2 = I. 
                    # M^2 = (a^2+bc) I. So we need a^2+bc to be a square? 
                    # If a^2+bc = s^2, then (1/s M)^2 = I. 
                    # So a^2+bc must be a quadratic residue.
                    # And for it to be a derangement, the equation (ax+b)/(cx-a) = x => cx^2 - 2ax - b = 0 must have no solutions.
                    # Discriminant D = 4a^2 + 4bc = 4(a^2+bc).
                    # So a^2+bc must be a non-residue?
                    # Contradiction: 
                    # 1. Involution: a^2+bc is a square (so that M^2 is scalar and can be normalized to I).
                    # 2. Derangement: a^2+bc is a non-square (so no fixed points).
                    # This implies there are NO involutory derangements in PGL(2,q) for odd q?
                    # Let's re-read the seed code.
                    # Seed code:
                    # val = (a*a - b*c) % q
                    # if is_res_table[val]: continue  # a^2 - bc is non-square
                    # if (a*a + b*c) % q == 0: continue
                    # d = -a
                    # This logic in seed code seems to be checking for something else.
                    # Let's check the seed code's `is_derangement`.
                    # It checks: c != 0, disc != 0, and disc is non-square.
                    # disc = (d-a)^2 + 4bc.
                    # For d=-a, disc = (-2a)^2 + 4bc = 4(a^2+bc).
                    # So seed requires a^2+bc to be non-square.
                    # And seed requires a^2-bc to be non-square? 
                    # Why a^2-bc? 
                    # Maybe it's not an involution in the strict sense of M^2=I, but an element of order 2 in the permutation group?
                    # If M^2 = lambda I, then M is an involution in PGL iff lambda is a square.
                    # So a^2+bc must be a square.
                    # But seed requires a^2+bc non-square for derangement.
                    # This means the seed code is looking for elements of order 2 that are derangements?
                    # If M^2 = lambda I with lambda non-square, then M^4 = lambda^2 I = I (if lambda^2=1? No, lambda is in F_q).
                    # If lambda is non-square, M^2 is not I. M^4 = lambda^2 I. 
                    # If lambda^2 = 1, then M is order 4?
                    # This is confusing. Let's just trust the seed's `is_derangement` and `sample_involutions` logic might be flawed or I am misinterpreting.
                    # Actually, let's just use the `is_derangement` check on all elements of PGL(2,q) and filter for involutions.
                    pass
        return invols

    # ---------- Main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # 2. Search PGL(2, q) for q < best_n
    # We will try to find k derangements in PGL(2,q) that generate a graph of girth g.
    # Since the seed's random search failed, we will try a deterministic approach for small q.
    
    # Note: The seed code's `sample_involutions` logic was:
    # a^2 - bc is non-square, a^2 + bc != 0.
    # Let's stick to the seed's definition of "involution" for consistency, 
    # but perhaps the search strategy was the issue.
    # I will implement a backtracking search over the generated candidates.

    candidates = []
    for q in range(max(3, n_min - 1), best_n - 1):
        if is_prime_power(q):
            candidates.append(q)

    candidates.sort()

    for q in candidates:
        if time.time() > deadline - 10.0:
            break
        n = q + 1
        if n >= best_n:
            break

        inv_table = precompute_inverses(q)
        is_res_table = precompute_quadratic_residues(q)

        # Generate a pool of potential generators
        # We want derangements.
        # Let's collect all derangements that are involutions (M^2 = I in PGL)
        # Condition for M = (a,b,c,d) to be involution in PGL:
        # M^2 = lambda I.
        # For d = -a, M^2 = (a^2+bc) I.
        # So we need a^2+bc to be a square.
        # Condition for derangement:
        # Fixed points: cx^2 + (d-a)x - b = 0.
        # Disc = (d-a)^2 + 4bc.
        # For d=-a, Disc = 4(a^2+bc).
        # So we need a^2+bc to be a non-square.
        # Contradiction: a^2+bc cannot be both square and non-square.
        # Thus, there are NO involutory derangements in PGL(2,q) for odd q.
        # The seed code was using `a^2 - bc` non-square.
        # Let's check what kind of elements those are.
        # If a^2 - bc is non-square, and d = -a.
        # M^2 = (a^2+bc) I.
        # If a^2+bc is a square, it's an involution.
        # If a^2-bc is non-square, does that imply a^2+bc is a square?
        # (a^2+bc)(a^2-bc) = a^4 - b^2c^2 = (a^2-bc)(a^2+bc).
        # Not necessarily.
        
        # Let's just collect ALL derangements and pick k of them.
        # This is a huge set. 
        # Maybe we should just try to find ANY k-regular graph using PGL(2,q) actions.
        # The seed code tried random sampling.
        # Let's try a greedy approach:
        # 1. Collect a set of derangements.
        # 2. Try to add them one by one to the generator set, checking girth.
        
        derangements = []
        # Limit the number of derangements we consider to avoid memory/time issues
        max_derangements = 200
        for a in range(q):
            for b in range(q):
                for c in range(1, q):
                    d = (-a) % q # Try involutions first, even if logic above says impossible, maybe I missed something
                    # Actually, let's just check is_derangement
                    if is_derangement(a, b, c, d, q, is_res_table):
                        derangements.append((a, b, c, d))
                        if len(derangements) >= max_derangements:
                            break
                if len(derangements) >= max_derangements:
                    break
            if len(derangements) >= max_derangements:
                break
            
        if len(derangements) < k:
            continue
            
        # Backtracking search
        # This might be slow, so we limit the depth or width.
        # Since k=5, depth is 5.
        # Width is len(derangements).
        # We can prune if adding a generator creates a short cycle.
        
        def backtrack(selected, idx):
            if len(selected) == k:
                res = build_from_gens(selected, q, k, g, inv_table, is_res_table)
                return res
            if time.time() > deadline - 2.0:
                return None
            
            for i in range(idx, len(derangements)):
                gen = derangements[i]
                # Check if gen is already in selected? 
                # They are distinct matrices, but might act same? 
                # Assume distinct matrices are distinct permutations.
                selected.append(gen)
                res = backtrack(selected, i + 1)
                if res:
                    return res
                selected.pop()
            return None

        # To avoid exponential explosion, we can limit the number of starting derangements
        # or use a heuristic.
        # Let's just try the first 10 derangements as the first generator.
        start_t = time.time()
        budget = 5.0
        for start_gen in derangements[:10]:
            if time.time() - start_t > budget:
                break
            res = backtrack([start_gen], 1)
            if res:
                if res[0] < best_n:
                    best_n = res[0]
                    best_result = res
                break # Found one for this q, move to next q (which is smaller? No, q increases)
                # Actually, we want smallest n. q+1 = n.
                # If we find one, it's valid. Since we iterate q in increasing order, 
                # the first valid q gives the smallest n in this family.
                break

    return best_result

# EVOLVE-BLOCK-END
