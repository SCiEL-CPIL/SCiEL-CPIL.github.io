import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 118
    rng = random.Random(12345)

    def has_k4(n, A):
        """Fast K4 check using bitmasks."""
        full = (1 << n) - 1
        for a in range(n):
            for b in range(a + 1, n):
                if not ((A[b] >> a) & 1):
                    continue
                common = A[a] & A[b] & full
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    common2 = common & A[c] & full
                    for d in range(c + 1, n):
                        if (common2 >> d) & 1:
                            return True
        return False

    def has_ind6(n, A):
        """Check if there's an independent set of size 6 by finding clique of size 6 in complement."""
        full = (1 << n) - 1
        # Build complement adjacency
        comp = [0] * n
        for i in range(n):
            comp[i] = full & ~(A[i] | (1 << i))
        
        found = [0]
        
        def bk(R_size, P, X):
            if found[0]:
                return
            if R_size >= 6:
                found[0] = 1
                return
            if not P:
                return
            if R_size + P.bit_count() < 6:
                return
            # Pivot
            if X:
                u = (X & -X).bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
            c = candidates
            while c and not found[0]:
                low = c & -c
                v = low.bit_length() - 1
                bk(R_size + 1, P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk(0, full, 0)
        return found[0]

    def valid(n, A):
        if has_k4(n, A):
            return False
        if has_ind6(n, A):
            return False
        return True

    def to_edges(n, A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (A[j] >> i) & 1:
                    edges.append([i, j])
        return edges

    def build_circulant(n, D):
        """Build circulant graph on n vertices with difference set D."""
        A = [0] * n
        for i in range(n):
            for d in D:
                j = (i + d) % n
                if j > i:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        return A

    def build_paley_circulant(n):
        """Build circulant using quadratic residues mod n (if n is prime power)."""
        # For n=36, use a mixed approach: residues mod 9 for the 9-cycles
        # Actually, let's try: D = {d : d^2 mod n in some set}
        # Better: use the fact that for composite, we can use CRT structure
        # 36 = 4 * 9
        # Try D = {d mod 36 : d mod 4 in QR_4 and d mod 9 in QR_9}
        QR_4 = {1}  # squares mod 4: 0,1 -> nonzero: {1}
        QR_9 = {1, 4, 7}  # squares mod 9: 1,4,7
        D = set()
        for d in range(1, n):
            if d % 4 in QR_4 and d % 9 in QR_9:
                D.add(d)
        return D

    def build_paley_prime(p):
        """Standard Paley graph on p vertices."""
        squares = {x * x % p for x in range(1, p)} - {0}
        return list(squares)

    # Strategy: Try multiple algebraic designs on n=36, then fall back to known-good
    # and try to grow from there.

    best_n = 0
    best_A = None

    # First, try to construct a valid 36-vertex graph using algebraic designs
    n_target = 36
    found_36 = False

    # Design 1: CRT-based circulant (4x9 structure)
    D1 = build_paley_circulant(n_target)
    A1 = build_circulant(n_target, D1)
    if valid(n_target, A1):
        best_n = n_target
        best_A = A1
        found_36 = True
    elif time.time() < deadline - 30:
        # Design 2: Try other CRT-based difference sets
        # QR_4 = {1}, QR_9 = {1,4,7}
        # Try variations
        variants = [
            {d for d in range(1, 36) if d % 4 == 1 and d % 9 in {1, 4, 7}},
            {d for d in range(1, 36) if d % 4 in {1, 2} and d % 9 in {1, 4, 7}},
            {d for d in range(1, 36) if d % 4 in {1, 3} and d % 9 in {1, 4, 7}},
            {d for d in range(1, 36) if d % 4 in {2, 3} and d % 9 in {1, 4, 7}},
            {d for d in range(1, 36) if d % 4 == 2 and d % 9 in {1, 4, 7}},
            {d for d in range(1, 36) if d % 4 == 3 and d % 9 in {1, 4, 7}},
        ]
        for D in variants:
            if time.time() > deadline - 30:
                break
            A = build_circulant(n_target, D)
            if valid(n_target, A):
                best_n = n_target
                best_A = A
                found_36 = True
                break

    # Design 3: Try circulant with D = quadratic residues mod 36 (nonzero squares)
    if not found_36 and time.time() < deadline - 30:
        squares36 = set()
        for x in range(1, 36):
            squares36.add(x * x % 36)
        squares36.discard(0)
        A = build_circulant(n_target, squares36)
        if valid(n_target, A):
            best_n = n_target
            best_A = A
            found_36 = True

    # Design 4: Try a "blow-up" style: take Paley(9) and blow up each vertex to 4
    if not found_36 and time.time() < deadline - 30:
        # Paley(9) on Z_9 with QR = {1,4,7}
        # Blow up: 4 copies of each vertex, connect copies if parent vertices adjacent
        # This gives a 36-vertex graph
        A = [0] * 36
        QR9 = {1, 4, 7}
        for i in range(9):
            for j in range(9):
                if i == j:
                    continue
                if (j - i) % 9 in QR9:
                    # Connect all 4 copies of i to all 4 copies of j
                    for a in range(4):
                        for b in range(4):
                            u = i * 4 + a
                            v = j * 4 + b
                            A[u] |= (1 << v)
                            A[v] |= (1 << u)
        if valid(36, A):
            best_n = 36
            best_A = A
            found_36 = True

    # Design 5: Try circulant with D from a (36, 18, 8) or similar difference set
    # Known: there's a (36,18,8) difference set in Z_36? Let's try D of size 18
    if not found_36 and time.time() < deadline - 30:
        # Try D = {1,2,4,5,8,10,13,14,17,19,22,23,26,28,31,32,34,35} (symmetric)
        # Actually let's just try a few random symmetric sets of size 18
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            # Random symmetric difference set of size 18
            D = set()
            while len(D) < 18:
                d = rng.randint(1, 35)
                D.add(d)
                D.add((36 - d) % 36)
                if (36 - d) % 36 == 0:
                    D.discard(0)
            D.discard(0)
            A = build_circulant(36, D)
            if valid(36, A):
                best_n = 36
                best_A = A
                found_36 = True
                break

    # If we found a 36-vertex graph, great. Otherwise, fall back to constructing
    # the best we can using the old strategy but with faster validation.
    if not found_36:
        # Start with Paley(17) which is known valid
        q = 17
        squares = {x * x % q for x in range(1, q)}
        n0 = q
        A = [0] * n0
        for i in range(n0):
            for j in range(n0):
                if i != j and (j - i) % q in squares:
                    A[i] |= (1 << j)
        if valid(n0, A):
            best_n = n0
            best_A = list(A)
        else:
            # Try small graphs
            for n_try in range(15, 1, -1):
                if time.time() > deadline - 20:
                    break
                D = set(rng.sample(range(1, n_try), n_try // 2))
                A_try = build_circulant(n_try, D)
                if valid(n_try, A_try):
                    best_n = n_try
                    best_A = A_try
                    break
            if best_n == 0:
                # Absolute fallback: empty graph on 5 vertices (no K4, alpha=5<6)
                best_n = 5
                best_A = [0] * 5

    # Now try to grow from best_n using faster methods
    if best_n < 36 and time.time() < deadline - 10:
        cur_n = best_n
        cur_A = list(best_A)
        
        def try_add_vertex_fast(cur_n, cur_A):
            new_n = cur_n + 1
            full_mask = (1 << cur_n) - 1
            
            # Precompute triangles
            triangles = []
            for a in range(cur_n):
                for b in range(a + 1, cur_n):
                    if not ((cur_A[b] >> a) & 1):
                        continue
                    common = cur_A[a] & cur_A[b] & full_mask
                    for c in range(b + 1, cur_n):
                        if (common >> c) & 1:
                            triangles.append((a, b, c))
            
            # Try random neighborhoods
            for _ in range(300):
                target_deg = cur_n // 2 + rng.randint(-cur_n // 4, cur_n // 4)
                target_deg = max(1, min(cur_n - 1, target_deg))
                neighbors = set(rng.sample(range(cur_n), target_deg))
                S = 0
                for v in neighbors:
                    S |= (1 << v)
                
                tri_ok = True
                for (a, b, c) in triangles:
                    if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                        tri_ok = False
                        break
                if not tri_ok:
                    continue
                
                cand_A = list(cur_A)
                cand_A.append(0)
                for i in range(cur_n):
                    if (S >> i) & 1:
                        cand_A[i] |= (1 << new_n - 1)
                        cand_A[new_n - 1] |= (1 << i)
                
                if has_k4(new_n, cand_A):
                    continue
                if has_ind6(new_n, cand_A):
                    continue
                return new_n, cand_A
            return None
        
        while cur_n < 36 and time.time() < deadline - 5:
            result = try_add_vertex_fast(cur_n, cur_A)
            if result:
                cur_n = result[0]
                cur_A = result[1]
                best_n = cur_n
                best_A = cur_A
            else:
                # Try local improvement
                improved = list(cur_A)
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    i = rng.randrange(cur_n)
                    j = rng.randrange(cur_n)
                    if i == j:
                        continue
                    trial = list(improved)
                    if (trial[j] >> i) & 1:
                        trial[i] &= ~(1 << j)
                        trial[j] &= ~(1 << i)
                    else:
                        trial[i] |= (1 << j)
                        trial[j] |= (1 << i)
                    if valid(cur_n, trial):
                        improved = trial
                if improved != cur_A:
                    cur_A = improved
                    result = try_add_vertex_fast(cur_n, cur_A)
                    if result:
                        cur_n = result[0]
                        cur_A = result[1]
                        best_n = cur_n
                        best_A = cur_A
                    else:
                        break
                else:
                    break

    return {'n': best_n, 'edges': to_edges(best_n, best_A)}
