import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

# ---------------- Bitset-based verification helpers (exact, fast) ----------------

def _has_K4_bitset(n, bits):
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
        if u == v:
            return False
        if u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    bits = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        bits[i] = m
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

# ---------------- Spectral construction: binarize eigenvectors of random symmetric matrices ----------------

def _spectral_candidate(n, rng, time_limit=5.0):
    """
    Generate candidate graphs by taking top eigenvectors of random symmetric matrices,
    binarizing signs to form adjacency, then repairing violations.
    """
    start_time = time.time()
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random symmetric matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2.0
        
        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
        except:
            continue
        
        # Use top k eigenvectors (k chosen to give moderate edge density)
        # For Ramsey graphs, we want density ~0.5
        k = max(1, n // 4)
        
        # Combine top eigenvectors
        combined = np.zeros(n)
        for i in range(k):
            idx = n - 1 - i  # largest eigenvalues
            combined += eigenvectors[:, idx]
        
        # Binarize: positive -> edge, negative -> no edge
        bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if combined[i] * combined[j] > 0:  # same sign -> edge
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        
        # Check validity
        if not _has_K4_bitset(n, bits) and not _has_indep_6_bitset(n, bits):
            return bits
        
        # If close, try local repair
        if time.time() - start_time > time_limit - 2.0:
            continue
        
        bits2 = _repair_violations(n, bits, rng, time_limit=1.5)
        if bits2 is not None and not _has_K4_bitset(n, bits2) and not _has_indep_6_bitset(n, bits2):
            return bits2
    
    return None

# ---------------- Local repair: fix K4/Ind6 violations via targeted edge flips ----------------

def _repair_violations(n, bits, rng, time_limit=3.0):
    """
    Repair K4 and Ind6 violations by flipping edges that are part of violations.
    """
    start_time = time.time()
    bits = list(bits)
    
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        has_k4 = _has_K4_bitset(n, bits)
        has_ind6 = _has_indep_6_bitset(n, bits)
        
        if not has_k4 and not has_ind6:
            return bits
        
        # Find a violating clique or independent set and flip an edge in it
        if has_k4:
            # Find a K4 and remove one edge
            k4 = _find_k4(n, bits)
            if k4:
                a, b, c, d = k4
                # Randomly pick an edge to remove
                edge_idx = rng.randint(0, 5)
                pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = pairs[edge_idx]
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
                continue
        
        if has_ind6:
            # Find an independent set of 6 and add one edge
            ind6 = _find_indep_6(n, bits)
            if ind6:
                # Randomly pick two vertices to connect
                u, v = rng.sample(ind6, 2)
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
                continue
        
        # If no specific violation found, do random flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        had_edge = (bits[u] >> v) & 1
        if had_edge:
            bits[u] &= ~(1 << v)
            bits[v] &= ~(1 << u)
        else:
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
    
    return bits

def _find_k4(n, bits):
    """Find a K4 and return its vertices, or None."""
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

def _find_indep_6(n, bits):
    """Find an independent set of 6 and return its vertices, or None."""
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

# ---------------- Structured spectral candidates: use Paley-type matrices ----------------

def _structured_spectral_candidate(n, rng, time_limit=5.0):
    """
    Use structured matrices (Paley-type) to generate spectral candidates.
    """
    start_time = time.time()
    
    for attempt in range(30):
        if time.time() - start_time > time_limit:
            break
        
        # Create a structured symmetric matrix based on quadratic residues
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Use a hash-like function to create structure
                val = ((i * 7 + j * 13) % 17)
                A[i, j] = 1.0 if val % 2 == 0 else -1.0
                A[j, i] = A[i, j]
        
        # Add some randomness
        noise = np.random.randn(n, n) * 0.1
        noise = (noise + noise.T) / 2.0
        A += noise
        
        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(A)
        except:
            continue
        
        # Use top k eigenvectors
        k = max(1, n // 5)
        combined = np.zeros(n)
        for i in range(k):
            idx = n - 1 - i
            combined += eigenvectors[:, idx]
        
        # Binarize
        bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if combined[i] * combined[j] > 0:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        
        # Check and repair
        if not _has_K4_bitset(n, bits) and not _has_indep_6_bitset(n, bits):
            return bits
        
        if time.time() - start_time > time_limit - 2.0:
            continue
        
        bits2 = _repair_violations(n, bits, rng, time_limit=1.5)
        if bits2 is not None and not _has_K4_bitset(n, bits2) and not _has_indep_6_bitset(n, bits2):
            return bits2
    
    return None

# ---------------- Main search strategy ----------------

def _search_spectral():
    best = {'n': 0, 'edges': []}
    rng = random.Random(42)
    np.random.seed(42)
    
    # Try spectral construction for various n
    for n in [36, 37, 38, 39, 40]:
        if time.time() > _DEADLINE - 15:
            break
        
        # Try random spectral
        bits = _spectral_candidate(n, rng, time_limit=8.0)
        if bits is not None:
            edges = _edges_from_bits(n, bits)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
                break  # Found a large valid graph
        
        # Try structured spectral
        if time.time() > _DEADLINE - 10:
            break
        bits = _structured_spectral_candidate(n, rng, time_limit=8.0)
        if bits is not None:
            edges = _edges_from_bits(n, bits)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
                break
    
    # If no large graph found, try smaller n with more attempts
    if best['n'] < 35:
        for n in [33, 34, 35]:
            if time.time() > _DEADLINE - 5:
                break
            
            for attempt in range(20):
                if time.time() > _DEADLINE - 3:
                    break
                
                bits = _spectral_candidate(n, rng, time_limit=2.0)
                if bits is not None:
                    edges = _edges_from_bits(n, bits)
                    if n > best['n']:
                        best = {'n': n, 'edges': edges}
                        break
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    
    # Start with a known valid baseline (Paley 17)
    best = {'n': 17, 'edges': []}
    bits17 = _paley_graph(17)
    if bits17 is not None and not _has_K4_bitset(17, bits17) and not _has_indep_6_bitset(17, bits17):
        best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
    
    # Try spectral search
    try:
        improved = _search_spectral()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        # Fallback to Paley 17
        bits17 = _paley_graph(17)
        if bits17 is not None:
            best = {'n': 17, 'edges': _edges_from_bits(17, bits17)}
        else:
            best = {'n': 1, 'edges': []}
    
    return best

# ---------------- Paley graph construction (for baseline) ----------------

def _paley_graph(p):
    """Paley graph on p vertices where p is a prime congruent to 1 mod 4."""
    if p < 2 or p % 4 != 1:
        return None
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits
