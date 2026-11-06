"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations
    
    deadline = time.time() + 145
    rng = random.Random(42)
    
    def make_triples(n):
        return list(combinations(range(n), 3))
    
    def build_index(n):
        triples = make_triples(n)
        idx = {t: i for i, t in enumerate(triples)}
        return triples, idx
    
    def precompute_k4_k5(n):
        """Precompute index lists for all K4s and K5s."""
        triples, tidx = build_index(n)
        k4_indices = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4_indices.append([tidx[(a,b,c)], tidx[(a,b,d)], tidx[(a,c,d)], tidx[(b,c,d)]])
        k5_indices = []
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            tris = [(a,b,c),(a,b,d),(a,b,e),(a,c,d),(a,c,e),(a,d,e),
                    (b,c,d),(b,c,e),(b,d,e),(c,d,e)]
            k5_indices.append([tidx[t] for t in tris])
        return k4_indices, k5_indices
    
    def check_valid(n, colors, k4_indices, k5_indices):
        ca = np.array(colors, dtype=np.uint8)
        for idxs in k4_indices:
            if ca[idxs[0]] == 0 and ca[idxs[1]] == 0 and ca[idxs[2]] == 0 and ca[idxs[3]] == 0:
                return False
        for idxs in k5_indices:
            if all(ca[i] == 1 for i in idxs):
                return False
        return True
    
    def count_violations(n, colors, k4_indices, k5_indices):
        ca = np.array(colors, dtype=np.uint8)
        count = 0
        for idxs in k4_indices:
            if ca[idxs[0]] == 0 and ca[idxs[1]] == 0 and ca[idxs[2]] == 0 and ca[idxs[3]] == 0:
                count += 1
        for idxs in k5_indices:
            if all(ca[i] == 1 for i in idxs):
                count += 1
        return count
    
    def try_modular(n, m, threshold):
        triples, _ = build_index(n)
        return [0 if (a + b + c) % m < threshold else 1 for a, b, c in triples]
    
    def try_xor(n, bits):
        triples, _ = build_index(n)
        return [0 if (a ^ b ^ c) < (1 << bits) else 1 for a, b, c in triples]
    
    def try_linear(n, m, c1, c2, c3):
        triples, _ = build_index(n)
        return [0 if (c1*a + c2*b + c3*c) % m < m//2 else 1 for a, b, c in triples]
    
    def try_diff(n, m, threshold):
        triples, _ = build_index(n)
        return [0 if ((b-a)*(c-b)) % m < threshold else 1 for a, b, c in triples]
    
    def try_prod(n, m, threshold):
        triples, _ = build_index(n)
        return [0 if (a*b*c) % m < threshold else 1 for a, b, c in triples]
    
    def try_sum_sq(n, m, threshold):
        triples, _ = build_index(n)
        return [0 if (a*a + b*b + c*c) % m < threshold else 1 for a, b, c in triples]
    
    def try_mixed(n, m1, m2, th1, th2):
        triples, _ = build_index(n)
        return [0 if ((a+b+c) % m1 < th1 and (a*b) % m2 < th2) else 1 for a, b, c in triples]
    
    def try_random_structured(n):
        triples, _ = build_index(n)
        m = rng.randint(3, 20)
        c1 = rng.randint(0, m-1)
        c2 = rng.randint(0, m-1)
        c3 = rng.randint(0, m-1)
        th = rng.randint(1, m-1)
        return [0 if (c1*a + c2*b + c3*c) % m < th else 1 for a, b, c in triples]
    
    # Search for valid colorings
    for n in range(34, 10, -1):
        if time.time() > deadline:
            break
        
        k4_indices, k5_indices = precompute_k4_k5(n)
        candidates = []
        
        # Modular
        for m in range(2, min(50, n+10)):
            if time.time() > deadline:
                break
            for th in range(1, m):
                candidates.append(try_modular(n, m, th))
        
        # XOR
        for bits in range(1, 7):
            candidates.append(try_xor(n, bits))
        
        # Linear
        for m in range(2, min(15, n)):
            if time.time() > deadline:
                break
            for c1 in range(m):
                for c2 in range(m):
                    for c3 in range(m):
                        if c1+c2+c3 == 0:
                            continue
                        candidates.append(try_linear(n, m, c1, c2, c3))
        
        # Difference-based
        for m in range(2, min(30, n+5)):
            if time.time() > deadline:
                break
            for th in range(1, m):
                candidates.append(try_diff(n, m, th))
        
        # Product-based
        for m in range(2, min(30, n+5)):
            if time.time() > deadline:
                break
            for th in range(1, m):
                candidates.append(try_prod(n, m, th))
        
        # Sum of squares
        for m in range(2, min(30, n+5)):
            if time.time() > deadline:
                break
            for th in range(1, m):
                candidates.append(try_sum_sq(n, m, th))
        
        # Mixed
        for m1 in range(2, min(15, n)):
            if time.time() > deadline:
                break
            for m2 in range(2, min(15, n)):
                for th1 in range(1, m1):
                    for th2 in range(1, m2):
                        candidates.append(try_mixed(n, m1, m2, th1, th2))
        
        # Random structured
        for i in range(200):
            if time.time() > deadline:
                break
            candidates.append(try_random_structured(n))
        
        # Check all candidates
        for colors in candidates:
            if time.time() > deadline:
                break
            if check_valid(n, colors, k4_indices, k5_indices):
                return {'n': n, 'colors': colors}
        
        # If we get here, try local search on best candidates
        if time.time() < deadline - 25:
            scored = []
            for colors in candidates[:100]:
                if time.time() > deadline:
                    break
                v = count_violations(n, colors, k4_indices, k5_indices)
                scored.append((v, colors))
            scored.sort(key=lambda x: x[0])
            
            for v, colors in scored[:3]:
                if time.time() > deadline:
                    break
                if v == 0:
                    return {'n': n, 'colors': colors}
                if v <= 10:
                    triples, _ = build_index(n)
                    cur = list(colors)
                    best_v = v
                    for step in range(20000):
                        if time.time() > deadline:
                            break
                        idx = rng.randint(0, len(cur)-1)
                        old = cur[idx]
                        cur[idx] = 1 - old
                        nv = count_violations(n, cur, k4_indices, k5_indices)
                        if nv < best_v:
                            best_v = nv
                            if nv == 0:
                                return {'n': n, 'colors': cur}
                        else:
                            cur[idx] = old
    
    # Fallback
    n = 9
    triples = make_triples(n)
    return {'n': n, 'colors': [0 if (a+b+c) % 3 == 0 else 1 for a, b, c in triples]}
# EVOLVE-BLOCK-END
