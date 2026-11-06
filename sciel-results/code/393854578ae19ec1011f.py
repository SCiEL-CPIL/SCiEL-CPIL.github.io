"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _build_index(n):
    """Build mapping from triple (a,b,c) to index in lexicographic order."""
    triples = list(combinations(range(n), 3))
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx

def _check_valid(colors, n, triples, idx):
    """Fast validity check: no red K4, no blue K5."""
    # Check red K4
    for a, b, c, d in combinations(range(n), 4):
        if (colors[idx[(a,b,c)]] == 0 and
            colors[idx[(a,b,d)]] == 0 and
            colors[idx[(a,c,d)]] == 0 and
            colors[idx[(b,c,d)]] == 0):
            return False
    # Check blue K5
    for a, b, c, d, e in combinations(range(n), 5):
        all_blue = True
        for tri in combinations((a,b,c,d,e), 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    return True

def _try_algebraic(n):
    """Try several algebraic constructions for the given n."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    
    constructions = [
        # sum mod 34 even/odd
        lambda a,b,c: 0 if (a+b+c) % 34 % 2 == 0 else 1,
        # sum mod 17 == 0
        lambda a,b,c: 0 if (a+b+c) % 17 == 0 else 1,
        # AP mod 17
        lambda a,b,c: 0 if (2*(sorted([a%17,b%17,c%17])[1]) - sorted([a%17,b%17,c%17])[0] - sorted([a%17,b%17,c%17])[2]) % 17 == 0 else 1,
        # product of diffs mod 17 == 0
        lambda a,b,c: 0 if ((b-a)*(c-b)*(a-c)) % 17 == 0 else 1,
        # sum in [0,16] mod 34
        lambda a,b,c: 0 if (a+b+c) % 34 <= 16 else 1,
        # (a-b)^2+(b-c)^2+(c-a)^2 mod 17 == 0
        lambda a,b,c: 0 if ((b-a)**2 + (c-b)**2 + (a-c)**2) % 17 == 0 else 1,
    ]
    
    for color_func in constructions:
        colors = bytearray(len(triples))
        for a, b, c in triples:
            colors[idx[(a,b,c)]] = color_func(a, b, c)
        if _check_valid(colors, n, triples, idx):
            return list(colors)
    return None

def _local_search(n, deadline):
    """Local search: start random, fix violations by flipping triples in violating sets."""
    triples, idx = _build_index(n)
    m = len(triples)
    
    # Precompute which triples belong to which quad (for red K4)
    # and which quints (for blue K5)
    quads = list(combinations(range(n), 4))
    quints = list(combinations(range(n), 5))
    
    # For each quad, store the 4 triple indices
    quad_triples = []
    for q in quads:
        a,b,c,d = q
        quad_triples.append([idx[(a,b,c)], idx[(a,b,d)], idx[(a,c,d)], idx[(b,c,d)]])
    
    # For each quint, store the 10 triple indices
    quint_triples = []
    for quint in quints:
        tris = [idx[tri] for tri in combinations(quint, 3)]
        quint_triples.append(tris)
    
    # Start with random coloring
    colors = bytearray(random.getrandbits(1) for _ in range(m))
    
    def count_violations():
        rv = 0
        for tris in quad_triples:
            if all(colors[i] == 0 for i in tris):
                rv += 1
        bv = 0
        for tris in quint_triples:
            if all(colors[i] == 1 for i in tris):
                bv += 1
        return rv + bv
    
    best_colors = colors.copy()
    best_viol = count_violations()
    
    if best_viol == 0:
        return best_colors
    
    max_iter = 200000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        viol = count_violations()
        if viol == 0:
            return colors
        if viol < best_viol:
            best_colors = colors.copy()
            best_viol = viol
        
        # Find a violating quad or quint and flip one of its triples
        # Prefer flipping a triple that's in a violation
        flip_idx = None
        for tris in quad_triples:
            if all(colors[i] == 0 for i in tris):
                # Flip a random one to blue
                flip_idx = tris[random.randrange(4)]
                break
        if flip_idx is None:
            for tris in quint_triples:
                if all(colors[i] == 1 for i in tris):
                    # Flip a random one to red
                    flip_idx = tris[random.randrange(10)]
                    break
        if flip_idx is None:
            # No violations found (shouldn't happen), flip random
            flip_idx = random.randrange(m)
        
        colors[flip_idx] = 1 - colors[flip_idx]
    
    if best_viol == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try algebraic constructions first (fast)
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        result = _try_algebraic(n)
        if result is not None:
            return {'n': n, 'colors': result}
    
    # Local search for n=34 down to n=20
    for n in range(34, 19, -1):
        if time.time() > deadline:
            break
        for restart in range(3):
            if time.time() > deadline:
                break
            random.seed(12345 + n * 100 + restart)
            result = _local_search(n, deadline)
            if result is not None:
                return {'n': n, 'colors': list(result)}
    
    # Fallback: small n with simple modular coloring
    for n in range(19, 8, -1):
        if time.time() > deadline:
            break
        triples, idx = _build_index(n)
        # Try a few simple colorings
        for mod in [3, 4, 5]:
            colors = bytearray(len(triples))
            for a, b, c in triples:
                colors[idx[(a,b,c)]] = 0 if (a+b+c) % mod == 0 else 1
            if _check_valid(colors, n, triples, idx):
                return {'n': n, 'colors': list(colors)}
    
    # Final fallback: n=9
    n = 9
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    for a, b, c in triples:
        colors[idx[(a,b,c)]] = 0 if (a+b+c) % 3 == 0 else 1
    return {'n': n, 'colors': list(colors)}
# EVOLVE-BLOCK-END
