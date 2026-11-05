import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin under 120s kill
    rng = random.Random(12345)

    def has_k4(n, A):
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return True
        return False

    def find_k4(n, A):
        """Return a K4 as tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (A[b] >> a) & 1:
                    continue
                for c in range(b + 1, n):
                    if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                        continue
                    for d in range(c + 1, n):
                        if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                            return (a, b, c, d)
        return None

    def find_ind6(n, A):
        """Find an independent set of size 6 using Bron-Kerbosch on complement.
        Returns the set as a tuple of 6 vertices, or None if max independent set < 6."""
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            nonlocal found
            if found[0] is not None:
                return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P:
                return
            # bound
            if len(R) + bin(P).count("1") < 6:
                return
            # pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                u = -1
                candidates = P
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(R, P & comp[v], X & comp[v])
                R.pop()
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], (1 << n) - 1, 0)
        return found[0]

    def max_independent_set_size(n, A):
        """Exact max independent set size via Bron-Kerbosch on complement."""
        comp = [0] * n
        for i in range(n):
            comp[i] = ((1 << n) - 1) & ~(A[i] | (1 << i))
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count("1") <= best[0]:
                return
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], (1 << n) - 1, 0)
        return best[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if max_independent_set_size(n, A) >= 6:
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    # Start with Paley(17)
    q = 17
    squares = {x * x % q for x in range(1, q)}
    n0 = q
    A = [0] * n0
    for i in range(n0):
        for j in range(n0):
            if i != j and (j - i) % q in squares:
                A[i] |= (1 << j)
    assert valid(n0, A), "Paley(17) should be valid"
    best_n = n0
    best_A = list(A)

    def try_grow(cur_n, cur_A, rng, deadline):
        """Try to grow the graph by 1 vertex. Returns (new_n, new_A) or None."""
        new_n = cur_n + 1
        new_A = cur_A + [0]
        
        # Strategy: try several random adjacencies for the new vertex,
        # then repair if needed.
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            cand_A = cur_A + [0]
            # Random adjacency with bias toward making it not create K4 or ind6
            # Try different densities
            density = 0.3 + rng.random() * 0.4
            for i in range(cur_n):
                if rng.random() < density:
                    cand_A[i] |= (1 << new_n - 1)
                    cand_A[new_n - 1] |= (1 << i)
            
            if valid(new_n, cand_A):
                return new_n, cand_A
        
        return None

    def local_repair(cur_n, cur_A, rng, deadline):
        """Targeted repair: fix specific K4s or independent sets."""
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            k4 = find_k4(cur_n, cur_A)
            if k4:
                a, b, c, d = k4
                # Remove a random edge from this K4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                cur_A[u] &= ~(1 << v)
                cur_A[v] &= ~(1 << u)
                continue
            
            ind6 = find_ind6(cur_n, cur_A)
            if ind6:
                # Add a random edge within this independent set to break it
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                cur_A[u] |= (1 << v)
                cur_A[v] |= (1 << u)
                continue
            
            # No violations found
            break
        
        return cur_A

    def random_flips(cur_n, cur_A, rng, deadline, n_flips=20):
        """Apply some random edge flips."""
        for _ in range(n_flips):
            if time.time() > deadline - 10:
                break
            i = rng.randrange(cur_n)
            j = rng.randrange(cur_n)
            if i == j:
                continue
            if (cur_A[j] >> i) & 1:
                cur_A[i] &= ~(1 << j)
                cur_A[j] &= ~(1 << i)
            else:
                cur_A[i] |= (1 << j)
                cur_A[j] |= (1 << i)
        return cur_A

    # Main search loop
    while time.time() < deadline:
        cur_n = best_n
        cur_A = list(best_A)
        
        # Try to grow
        if cur_n < 40 and time.time() < deadline - 20:
            result = try_grow(cur_n, cur_A, rng, deadline)
            if result:
                best_n = result[0]
                best_A = result[1]
                continue
        
        # Local repair on current best
        repaired = local_repair(cur_n, cur_A, rng, deadline)
        if not has_k4(cur_n, repaired) and max_independent_set_size(cur_n, repaired) < 6:
            # Try random perturbations and re-repair
            for _ in range(10):
                if time.time() > deadline - 15:
                    break
                perturbed = random_flips(cur_n, list(repaired), rng, deadline, n_flips=rng.randint(1, 5))
                repaired2 = local_repair(cur_n, perturbed, rng, deadline)
                if not has_k4(cur_n, repaired2) and max_independent_set_size(cur_n, repaired2) < 6:
                    # Check if this helps growth
                    if cur_n < 40:
                        result = try_grow(cur_n, repaired2, rng, deadline)
                        if result:
                            best_n = result[0]
                            best_A = result[1]
                            break
                    # Keep as alternative if same size
                    break

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
# EVOLVE-BLOCK-END
