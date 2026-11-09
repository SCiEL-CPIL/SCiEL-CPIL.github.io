import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 138.0  # safety margin before 150s kill

# ---------------- Bitset-based verification helpers (exact, fast) ----------------

def _has_K4_bitset(n, bits):
    """Check for K4 using bitsets. bits[i] has bit j set iff edge (i,j)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                if common2:
                    return True
    return False

def _has_indep_6_bitset(n, bits):
    """Check for independent set of size 6 (clique of size 6 in complement)."""
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        if common4:
                            return True
    return False

def _is_valid(n, edges):
    if n < 1:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    bits = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        bits[i] = mask
    if _has_K4_bitset(n, bits):
        return False
    if _has_indep_6_bitset(n, bits):
        return False
    return True

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Baseline: 17-vertex Paley graph (known valid) ----------------

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

_BASELINE = _paley_17()

# ---------------- Spectral / structured construction ----------------

def _hadamard_bits(n, rng):
    """Generate a graph by binarizing signs of a random Hadamard-like symmetric matrix."""
    # Use a random symmetric matrix with entries ±1 (Hadamard-like structure)
    M = np.ones((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            val = 1 if rng.random() < 0.5 else -1
            M[i, j] = val
            M[j, i] = val
    # Binarize: edge if M[i,j] > 0
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] > 0:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _paley_circulant_bits(n, rng):
    """Generate a graph using a Paley-type circulant structure with random prime modulus."""
    # Find a prime p near n where p ≡ 1 (mod 4) for Paley construction
    candidates = []
    for p in range(max(5, n - 10), n + 15):
        if p % 2 == 0 and p != 2:
            continue
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime and p % 4 == 1:
            candidates.append(p)
    if not candidates:
        # Fallback: random graph
        bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        return bits
    
    p = rng.choice(candidates)
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _eigenvector_sign_bits(n, rng, k):
    """Generate a graph by taking top-k eigenvectors of a random symmetric matrix and binarizing signs."""
    A = np.random.randn(n, n)
    A = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A)
    # Take top k eigenvectors (largest eigenvalues)
    idx = np.argsort(eigvals)[::-1][:k]
    V = eigvecs[:, idx]
    # Project onto first principal direction and binarize
    signs = np.sign(V[:, 0])
    signs[signs == 0] = 1
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if signs[i] != signs[j]:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _count_violations(n, bits):
    """Count K4s and independent 6-sets (approximate for ranking)."""
    k4_count = 0
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    ind6_count = 0
    
    # Count K4s
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                k4_count += bin(common2).count('1')
    
    # Count independent 6-sets (cliques of size 6 in complement)
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        ind6_count += bin(common4).count('1')
    
    return k4_count, ind6_count

def _targeted_repair(n, bits, rng, max_iters=500):
    """Targeted edge-flip repair: only flip edges that are part of violations."""
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        k4 = _find_K4_fast(n, bits)
        if k4 is None:
            indep = _find_indep_6_fast(n, bits)
            if indep is None:
                return bits
            # Add a random edge within the independent set
            verts = list(indep)
            u, v = verts[rng.randint(0, 5)], verts[rng.randint(0, 5)]
            if u != v:
                if not (bits[u] >> v) & 1:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
        else:
            # Remove a random edge from the K4
            a, b, c, d = k4
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            u, v = pairs[rng.randint(0, 5)]
            if (bits[u] >> v) & 1:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
    
    return bits

def _find_K4_fast(n, bits):
    """Return a K4 as tuple (a,b,c,d) or None."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if (common2 >> d) & 1:
                        return (a, b, c, d)
    return None

def _find_indep_6_fast(n, bits):
    """Return an independent 6-set as tuple or None."""
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        for f in range(e + 1, n):
                            if (common4 >> f) & 1:
                                return (a, b, c, d, e, f)
    return None

def _spectral_search():
    """Spectral construction: generate candidates from structured sources, rank by violations, repair top ones."""
    best = _BASELINE
    rng = random.Random(42)
    
    # Generate candidates from multiple structured sources
    candidates = []
    for target_n in [20, 25, 30, 35, 36, 37, 38, 39, 40]:
        if time.time() > _DEADLINE - 15:
            break
        
        # Hadamard-like
        if time.time() < _DEADLINE - 10:
            bits = _hadamard_bits(target_n, rng)
            candidates.append((target_n, bits, 'hadamard'))
        
        # Paley circulant
        if time.time() < _DEADLINE - 10:
            bits = _paley_circulant_bits(target_n, rng)
            candidates.append((target_n, bits, 'paley'))
        
        # Eigenvector sign patterns
        if time.time() < _DEADLINE - 10:
            for k in [1, 2, 3]:
                bits = _eigenvector_sign_bits(target_n, rng, k)
                candidates.append((target_n, bits, f'eig_k{k}'))
    
    # Rank candidates by violation count (lower is better)
    scored = []
    for n, bits, source in candidates:
        if time.time() > _DEADLINE - 20:
            break
        k4c, ind6c = _count_violations(n, bits)
        scored.append((k4c + ind6c, n, bits, source))
    
    scored.sort(key=lambda x: x[0])
    
    # Repair top candidates
    for score, n, bits, source in scored[:10]:
        if time.time() > _DEADLINE - 5:
            break
        repaired = _targeted_repair(n, bits[:], rng, max_iters=300)
        if not _has_K4_bitset(n, repaired) and not _has_indep_6_bitset(n, repaired):
            edges = _edges_from_bits(n, repaired)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    best = _BASELINE
    
    try:
        improved = _spectral_search()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        best = _BASELINE
    
    return best
