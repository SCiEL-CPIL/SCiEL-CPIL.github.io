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
        """BFS from each vertex, checking for cycles shorter than g."""
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
                        if y != parent[x] and parent[y] != x:
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
                    return None  # Duplicate edge
                adj[i].add(j)
                adj[j].add(i)
        return validate(adj, n, k, g)

    # ---------- Enumerate all involutory derangements for small q ----------
    def enumerate_involutions(q, is_res_table):
        """Enumerate all involutory derangements in PGL(2,q)."""
        invols = []
        for a in range(q):
            for b in range(q):
                for c in range(1, q):  # c != 0 for derangement
                    d = (-a) % q
                    # Check if it's an involution: matrix squared = identity
                    # [[a,b],[c,d]]^2 = [[a^2+bc, b(a+d)], [c(a+d), bc+d^2]]
                    # For involution: a+d=0 (which we have: d=-a), and a^2+bc=1, bc+d^2=1
                    # With d=-a: a^2+bc=1 and bc+a^2=1 (same condition)
                    if (a * a + b * c) % q != 1:
                        continue
                    # Check if it's a derangement (no fixed points)
                    # Fixed point: ax+b = x(cx+d) => (a-c)x = d-b => x = (d-b)/(a-c) if a!=c
                    # Or if a=c, then need d=b for all fixed points (identity) or no fixed points
                    # For derangement: the equation (a-c)x^2 + (d-b)x + ... has no solutions
                    # Actually for PGL(2,q) acting on projective line, fixed points satisfy:
                    # (a-c)x^2 + (d-b)x = 0 in affine, plus check infinity
                    # Let's use the discriminant approach
                    disc = ((d - a) ** 2 + 4 * b * c) % q
                    if disc == 0:
                        continue
                    if is_res_table[disc]:
                        continue
                    gen = (a, b, c, d)
                    invols.append(gen)
        return invols

    # ---------- Local search to improve girth ----------
    def local_search_girth(adj, n, k, g, inv_table, q, gens, is_res_table, max_iterations=20):
        """Try to improve girth by swapping generators."""
        current_girth = g
        best_gens = list(gens)
        
        # Compute current girth
        def compute_girth(adj, n, max_check=15):
            """Compute the actual girth of the graph up to max_check."""
            for target_g in range(3, max_check + 1):
                if check_girth_bfs(adj, n, target_g):
                    return target_g
            return max_check
        
        current_girth_val = compute_girth(adj, n, max_check=g + 5)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            # Try swapping each generator with a random alternative
            for i in range(k):
                if time.time() > deadline - 10.0:
                    break
                
                # Try a few random alternatives
                for attempt in range(5):
                    a = random.randint(0, q - 1)
                    b = random.randint(0, q - 1)
                    c = random.randint(1, q - 1)
                    d = (-a) % q
                    if (a * a + b * c) % q != 1:
                        continue
                    disc = ((d - a) ** 2 + 4 * b * c) % q
                    if disc == 0 or is_res_table[disc]:
                        continue
                    
                    new_gen = (a, b, c, d)
                    if new_gen in best_gens:
                        continue
                    
                    # Build new graph with this swap
                    new_gens = list(best_gens)
                    new_gens[i] = new_gen
                    
                    n_pts = q + 1
                    new_adj = [set() for _ in range(n_pts)]
                    valid = True
                    for vtx in range(n_pts):
                        x = vtx if vtx < q else q
                        for gen in new_gens:
                            y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q, inv_table)
                            if y is None or y == vtx:
                                valid = False
                                break
                            j = y if y < q else q
                            if j in new_adj[vtx]:
                                valid = False
                                break
                            new_adj[vtx].add(j)
                            new_adj[j].add(vtx)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check if all vertices have degree k
                    if any(len(new_adj[v]) != k for v in range(n_pts)):
                        continue
                    
                    # Check connectivity
                    visited = [False] * n_pts
                    visited[0] = True
                    queue = deque([0])
                    count = 0
                    while queue:
                        u = queue.popleft()
                        count += 1
                        for v in new_adj[u]:
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                    if count != n_pts:
                        continue
                    
                    # Check girth
                    new_girth_val = compute_girth(new_adj, n_pts, max_check=current_girth_val + 5)
                    
                    if new_girth_val > current_girth_val:
                        current_girth_val = new_girth_val
                        best_gens = new_gens
                        adj = new_adj
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_gens, adj, current_girth_val

    # ---------- Main ----------
    # 1. deterministic baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)

    # 2. Search PGL(2, q) for q < best_n
    candidates = []
    for q in range(max(3, n_min - 1), best_n - 1):
        if is_prime_power(q):
            candidates.append(q)

    # Sort by q
    candidates.sort()

    for q in candidates:
        if time.time() > deadline - 5.0:
            break
        n = q + 1
        if n >= best_n:
            break

        inv_table = precompute_inverses(q)
        is_res_table = precompute_quadratic_residues(q)

        # Enumerate all involutory derangements
        invols = enumerate_involutions(q, is_res_table)
        if len(invols) < k:
            continue

        random.shuffle(invols)
        
        # Try multiple random subsets
        best_gens_for_q = None
        best_adj_for_q = None
        best_girth_for_q = 0
        
        num_trials = min(50, max(10, int((deadline - time.time()) / 2.0)))
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                break
            
            gens = random.sample(invols, k)
            
            res = build_from_gens(gens, q, k, g, inv_table, is_res_table)
            if res:
                # Found a valid graph with girth >= g
                # Try to improve girth via local search
                n_pts = q + 1
                adj = [set() for _ in range(n_pts)]
                for vtx in range(n_pts):
                    x = vtx if vtx < q else q
                    for gen in gens:
                        y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q, inv_table)
                        j = y if y < q else q
                        adj[vtx].add(j)
                        adj[j].add(vtx)
                
                # Local search to improve girth
                improved_gens, improved_adj, improved_girth = local_search_girth(
                    adj, n_pts, k, g, inv_table, q, gens, is_res_table, max_iterations=10
                )
                
                if improved_girth > best_girth_for_q:
                    best_girth_for_q = improved_girth
                    best_gens_for_q = improved_gens
                    best_adj_for_q = improved_adj
                
                # If we've achieved girth > g, we can potentially use this
                if improved_girth > g:
                    # Verify the graph is valid
                    final_res = validate(improved_adj, n_pts, k, g)
                    if final_res and final_res[0] < best_n:
                        best_n = final_res[0]
                        best_result = final_res
                    break
        
        if best_n < n_luw:
            break

    return best_result

# EVOLVE-BLOCK-END
