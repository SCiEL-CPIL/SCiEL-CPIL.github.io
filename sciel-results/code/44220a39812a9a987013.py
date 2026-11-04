import time
import math
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # --- Finite Field F_q for q = p^m ---
    class FiniteField:
        def __init__(self, p, m, poly):
            self.p = p
            self.m = m
            self.q = p ** m
            # poly is a list of coefficients [c_0, c_1, ..., c_m] for x^m + ... + c_0 = 0
            # We store the reduction polynomial coefficients for x^i (i>=m)
            # Actually, store the polynomial used for reduction: x^m = -sum_{i=0}^{m-1} c_i x^i
            self.poly = poly
            # Precompute multiplication table? No, too big for large q.
            # Precompute powers? Maybe.
            self.inv_cache = {0: None, 1: 1}
            
        def add(self, a, b):
            # a, b are lists of length m
            res = [0] * self.m
            for i in range(self.m):
                res[i] = (a[i] + b[i]) % self.p
            return res

        def mul(self, a, b):
            # Polynomial multiplication mod p
            res = [0] * (2 * self.m - 1)
            for i in range(self.m):
                if a[i] == 0: continue
                for j in range(self.m):
                    if b[j] == 0: continue
                    res[i + j] = (res[i + j] + a[i] * b[j]) % self.p
            # Reduction mod poly
            # poly represents x^m + c_{m-1}x^{m-1} + ... + c_0 = 0
            # So x^m = -(c_{m-1}x^{m-1} + ... + c_0)
            # We reduce from high degree down
            # Our poly input: irreducible polynomial of degree m.
            # Let's assume standard form: x^m + p_{m-1}x^{m-1} + ... + p_0
            # Then x^m = -p_{m-1}x^{m-1} - ... - p_0
            p_coeffs = self.poly # [p_0, ..., p_{m-1}]
            for i in range(2 * self.m - 2, self.m - 1, -1):
                if res[i] != 0:
                    coeff = res[i]
                    for j in range(self.m):
                        # x^i * x^j = x^{i+j}
                        # x^i = x^{i-m} * x^m = x^{i-m} * (-sum p_l x^l)
                        # So we subtract coeff * p_j from res[i-m+j]
                        res[i - self.m + j] = (res[i - self.m + j] - coeff * p_coeffs[j]) % self.p
                    res[i] = 0
            return res[:self.m]

        def inv(self, a):
            # Extended Euclidean or Fermat's little theorem (a^(q-2))
            # Fermat is easier to code correctly for small q
            if a[0] == 0 and all(x == 0 for x in a):
                return None
            if self.inv_cache.get(tuple(a)) is not None:
                return self.inv_cache[tuple(a)]
            
            # a^(q-2)
            res = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # dummy, will be sized properly
            res = [0] * self.m
            res[0] = 1
            base = a
            exp = self.q - 2
            while exp > 0:
                if exp % 2 == 1:
                    res = self.mul(res, base)
                base = self.mul(base, base)
                exp //= 2
            self.inv_cache[tuple(a)] = res
            return res

        def to_int(self, a):
            # Convert field element to integer 0..q-1
            val = 0
            for i in range(self.m):
                val += a[i] * (self.p ** i)
            return val

        def from_int(self, val):
            # Convert integer 0..q-1 to field element
            a = [0] * self.m
            for i in range(self.m):
                a[i] = val % self.p
                val //= self.p
            return a

    def get_prime_power_field(q):
        # Returns FiniteField instance for q
        # Check if q is prime
        if q < 2:
            return None
        # Check prime
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(math.isqrt(n)) + 1):
                if n % i == 0: return False
            return True
        
        if is_prime(q):
            # Field F_p
            class PrimeField:
                def __init__(self, p):
                    self.p = p
                    self.q = p
                def add(self, a, b): return (a + b) % self.p
                def mul(self, a, b): return (a * b) % self.p
                def inv(self, a):
                    if a == 0: return None
                    return pow(a, self.p - 2, self.p)
                def to_int(self, a): return a
                def from_int(self, val): return val
            return PrimeField(q)
            
        # Check prime power p^m
        p = 2
        while p * p <= q:
            if q % p == 0:
                break
            p += 1
        if p * p > q:
            return None # Not a prime power? Or prime handled above.
            
        # Find m such that p^m = q
        m = 0
        temp = q
        while temp % p == 0:
            temp //= p
            m += 1
        if p ** m != q:
            return None
            
        # Find irreducible polynomial of degree m over F_p
        # For small m, we can hardcode or search
        # Search for monic irreducible polynomial
        def is_irreducible(poly, p, m):
            # poly: list of coeffs [c_0, ..., c_m] with c_m=1
            # Check if it has roots in F_p
            for x in range(p):
                val = 0
                for i in range(m, -1, -1):
                    val = (val * x + poly[i]) % p
                if val == 0:
                    return False
            # Check if it shares factor with x^(p^i) - x for i=1..m/2
            # This is complex. For small m (m<=5), we can just try to factor?
            # Simpler: for very small q, we can use known irreducibles.
            # q=4 (p=2, m=2): x^2 + x + 1 -> [1, 1, 1]
            # q=8 (p=2, m=3): x^3 + x + 1 -> [1, 1, 0, 1]
            # q=9 (p=3, m=2): x^2 + 1 -> [1, 0, 1]
            # q=16 (p=2, m=4): x^4 + x + 1 -> [1, 1, 0, 0, 1]
            # q=25 (p=5, m=2): x^2 + 1 -> [1, 0, 1]
            # q=27 (p=3, m=3): x^3 + x^2 + 2 -> [2, 1, 1, 1]
            # q=32 (p=2, m=5): x^5 + x^2 + 1 -> [1, 0, 1, 0, 0, 1]
            # q=49 (p=7, m=2): x^2 + 1 -> [1, 0, 1]
            # q=64 (p=2, m=6): x^6 + x^3 + 1 -> [1, 0, 0, 1, 0, 0, 1]
            
            # For robustness, let's just return False if we don't know, 
            # and rely on the hardcoded list for common small q.
            # If not in hardcoded, we might skip this q.
            return True # Assume true for now if no root, but this is unsafe.
            
        # Hardcoded irreducibles for small q
        irreducibles = {
            (2, 2): [1, 1, 1], # x^2+x+1
            (2, 3): [1, 1, 0, 1], # x^3+x+1
            (3, 2): [1, 0, 1], # x^2+1
            (2, 4): [1, 1, 0, 0, 1], # x^4+x+1
            (5, 2): [1, 0, 1], # x^2+1
            (3, 3): [2, 1, 1, 1], # x^3+x^2+2
            (2, 5): [1, 0, 1, 0, 0, 1], # x^5+x^2+1
            (7, 2): [1, 0, 1], # x^2+1
            (2, 6): [1, 0, 0, 1, 0, 0, 1], # x^6+x^3+1
            (11, 2): [1, 0, 1], # x^2+1
            (13, 2): [1, 0, 1], # x^2+1
            (17, 2): [1, 0, 1], # x^2+1
            (19, 2): [1, 0, 1], # x^2+1
            (23, 2): [1, 0, 1], # x^2+1
            (29, 2): [1, 0, 1], # x^2+1
            (31, 2): [1, 0, 1], # x^2+1
            (37, 2): [1, 0, 1], # x^2+1
            (41, 2): [1, 0, 1], # x^2+1
            (43, 2): [1, 0, 1], # x^2+1
            (47, 2): [1, 0, 1], # x^2+1
            (53, 2): [1, 0, 1], # x^2+1
            (59, 2): [1, 0, 1], # x^2+1
            (61, 2): [1, 0, 1], # x^2+1
            (67, 2): [1, 0, 1], # x^2+1
            (71, 2): [1, 0, 1], # x^2+1
            (73, 2): [1, 0, 1], # x^2+1
            (79, 2): [1, 0, 1], # x^2+1
            (83, 2): [1, 0, 1], # x^2+1
            (89, 2): [1, 0, 1], # x^2+1
            (97, 2): [1, 0, 1], # x^2+1
        }
        
        if (p, m) in irreducibles:
            poly = irreducibles[(p, m)]
            return FiniteField(p, m, poly)
            
        return None

    # --- Girth Verification ---
    def girth_check(adj, n, target_girth):
        # Check if girth >= target_girth
        # Non-backtracking BFS from each vertex
        # State: (v, parent)
        # If we find a cycle of length < target_girth, return False
        # We can prune BFS if depth >= (target_girth - 1) // 2 for odd girth?
        # Actually, if girth is g, the non-backtracking BFS tree from u will have
        # collisions at depth g/2 (even) or (g-1)/2 (odd) ??
        # Let's just run BFS up to depth target_girth - 1.
        # If no cycle found, girth >= target_girth.
        
        max_depth = target_girth - 1
        
        for u in range(n):
            visited = {}
            q = deque()
            for v in adj[u]:
                visited[(v, u)] = 1
                q.append((v, u, 1))
            
            while q:
                v, parent, d = q.popleft()
                if d >= max_depth:
                    continue
                for w in adj[v]:
                    if w == parent:
                        continue
                    if w == u:
                        # Cycle of length d+1
                        if d + 1 < target_girth:
                            return False
                        continue
                    state = (w, v)
                    if state not in visited:
                        visited[state] = d + 1
                        q.append((w, v, d + 1))
                    else:
                        # Collision: path u->...->v->w and u->...->w->v?
                        # This implies a cycle.
                        # Length of cycle = d + 1 + visited[state] ?
                        # Actually, if (w, v) was visited at depth d', 
                        # we have a walk of length d' to (w,v) and d+1 to (w,v) via v?
                        # No, (w,v) means we are at w, came from v.
                        # Previous visit: at w, came from v.
                        # This means the path to w via v is the same?
                        # If we reach (w, v) again, it's a cycle in the state graph.
                        # The length of the cycle in the original graph is d + 1 + d_prev?
                        # Actually, simpler: if state exists, we have a cycle.
                        # Length = d + 1 + visited[state]
                        # Wait, visited[state] is the depth when we first saw (w,v).
                        # Let d_prev = visited[state].
                        # Path 1: u -> ... -> v -> w (length d+1)
                        # Path 2: u -> ... -> w -> v (length d_prev)
                        # This forms a cycle?
                        # Yes, the union of the two paths forms a cycle.
                        # Length <= d + 1 + d_prev.
                        # If d + 1 + d_prev < target_girth, return False.
                        d_prev = visited[state]
                        if d + 1 + d_prev < target_girth:
                            return False
        return True

    def is_valid(k, g, n, edges):
        adj = [set() for _ in range(n)]
        seen = set()
        for (i, j) in edges:
            if i == j: return False
            if i < 0 or i >= n or j < 0 or j >= n: return False
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen: return False
            seen.add((a, b))
            adj[i].add(j)
            adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k: return False
        # Connectivity
        visited = [False] * n
        dq = deque([0])
        visited[0] = True
        cnt = 0
        while dq:
            u = dq.popleft()
            cnt += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dq.append(v)
        if cnt != n: return False
        # Girth
        if not girth_check(adj, n, g):
            return False
        return True

    # --- Morgenstern Construction ---
    def build_morgenstern(q, k, g):
        field = get_prime_power_field(q)
        if field is None:
            return None
            
        n = q + 1
        # Vertices are 0..q. q represents infinity.
        # Action of matrix [[a,b],[c,d]] on x:
        # if x == q:
        #   if c == 0: return q
        #   else: return a/c
        # else:
        #   num = a*x + b
        #   den = c*x + d
        #   if den == 0: return q
        #   else: return num/den
        
        def action(x, a, b, c, d):
            if x == q:
                if c == 0:
                    return q
                # return a * inv(c)
                c_inv = field.inv(field.from_int(c))
                if c_inv is None: return -1
                res = field.mul(field.from_int(a), c_inv)
                return field.to_int(res)
            
            x_f = field.from_int(x)
            num = field.add(field.mul(field.from_int(a), x_f), field.from_int(b))
            den = field.add(field.mul(field.from_int(c), x_f), field.from_int(d))
            
            # Check if den is 0
            if all(v == 0 for v in den):
                return q
            
            den_inv = field.inv(den)
            if den_inv is None: return -1
            res = field.mul(num, den_inv)
            return field.to_int(res)

        # Generate candidate generators
        # We need k generators.
        # Standard Morgenstern uses specific sets.
        # Let's try sets of matrices.
        
        # Helper to get matrix entries as ints
        def get_matrices():
            mats = []
            # Identity-like? No, need distinct.
            # Try random matrices
            rng = random.Random(q)
            for _ in range(100):
                a = rng.randint(0, q-1)
                b = rng.randint(0, q-1)
                c = rng.randint(0, q-1)
                d = rng.randint(0, q-1)
                # Check det != 0
                # det = ad - bc in F_q
                ad = field.mul(field.from_int(a), field.from_int(d))
                bc = field.mul(field.from_int(b), field.from_int(c))
                det = field.sub(ad, bc) # Need sub method
                # Implement sub
                def sub(a, b):
                    res = [0] * field.m
                    for i in range(field.m):
                        res[i] = (a[i] - b[i]) % field.p
                    return res
                det = sub(ad, bc)
                if not all(v == 0 for v in det):
                    mats.append((a, b, c, d))
            return mats

        mats = get_matrices()
        
        # Try combinations of k generators
        # For small k, we can try all combinations? No, too many.
        # Try random subsets of size k from mats.
        
        best_res = None
        for _ in range(50):
            if elapsed() > deadline - 5:
                break
            if len(mats) < k:
                break
            gens = random.sample(mats, k)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n):
                x = i
                for (a, b, c, d) in gens:
                    y = action(x, a, b, c, d)
                    if y == -1:
                        valid = False
                        break
                    if i == y:
                        valid = False
                        break
                    if y in adj[i]:
                        valid = False
                        break
                    adj[i].add(y)
                    adj[y].add(i)
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Check regular
            reg = True
            for i in range(n):
                if len(adj[i]) != k:
                    reg = False
                    break
            if not reg:
                continue
                
            # Check connectivity
            visited = [False] * n
            dq = deque([0])
            visited[0] = True
            cnt = 0
            while dq:
                u = dq.popleft()
                cnt += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            if cnt != n:
                continue
                
            # Check girth
            if girth_check(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

        return None

    # --- LUW Construction ---
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
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

    # --- Main ---
    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            if is_valid(k, g, n, edges):
                best = (n, edges)
                best_n = n
                return True
        return False

    # 1. Try Morgenstern for prime powers q >= n_min - 1
    # n = q + 1, so q >= n_min - 1
    q_start = max(2, n_min - 1)
    q_end = q_start + 1000 # Search a bit above Moore bound
    
    q = q_start
    while q <= q_end and elapsed() < deadline - 20:
        res = build_morgenstern(q, k, g)
        if res:
            consider(res[0], res[1])
        q += 1

    # 2. Try LUW
    if elapsed() < deadline - 5:
        try:
            res = luw_construct(k, g)
            consider(res[0], res[1])
        except Exception:
            pass

    if best is None:
        # Fallback: LUW with larger d
        try:
            q = k
            d = max(5, g)
            if q == 4:
                table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
                add = lambda x, y: x ^ y
                sub = lambda x, y: x ^ y
                mul = lambda x, y: table[x][y]
            else:
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
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
            best = (len(order), edges)
        except Exception:
            # Last resort: Random
            n = n_min + 100
            if n * k % 2 != 0: n += 1
            edges = []
            # Just return a dummy valid graph? No, must be valid.
            # LUW should work.
            pass

    return best

# EVOLVE-BLOCK-END
