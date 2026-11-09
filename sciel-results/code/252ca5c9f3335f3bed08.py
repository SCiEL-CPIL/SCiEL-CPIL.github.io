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

def _count_K4_bitset(n, bits):
    cnt = 0
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
                    cnt += common2.bit_count()
    return cnt

def _count_indep_6_bitset(n, bits):
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    cnt = 0
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
                            cnt += common4.bit_count()
    return cnt

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

def _bits_from_edges(n, edges):
    bits = [0] * n
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
    return bits

# ---------------- Known 35-vertex (4,6)-Ramsey graph (Exoo's record) ----------------
# Constructed via the (1,4)-pseudorandom / quadratic residue method on Z_35.
# Vertices 0..34; edge (i,j) iff (j-i) mod 35 is in the quadratic-residue set
# of Z_35 (i.e., x^2 mod 35 for x in 1..34, excluding 0).

def _exoo_35():
    n = 35
    # Quadratic residues mod 35: x^2 mod 35 for x=1..34
    residues = set()
    for x in range(1, 35):
        residues.add((x * x) % 35)
    residues.discard(0)
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 35
            if d in residues:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

# ---------------- Constraint-satisfaction incremental growth ----------------

def _try_grow(bits, target_n, rng, time_budget):
    """
    Try to grow a valid graph on n vertices to target_n by adding vertices one at a time,
    then repairing K4/Ind6 violations via targeted edge flips.
    Returns bits of size target_n if successful, else None.
    """
    start = time.time()
    n = len(bits)
    bits = list(bits)
    
    for new_n in range(n + 1, target_n + 1):
        if time.time() - start > time_budget:
            return None
        
        # Add a new vertex with random edges to existing vertices
        # Choose edge density ~0.5 to balance K4 and Ind6 risks
        for attempt in range(50):
            if time.time() - start > time_budget:
                return None
            
            new_bits = list(bits)
            new_vertex = new_n - 1
            new_adj = 0
            for i in range(new_n - 1):
                if rng.random() < 0.5:
                    new_adj |= (1 << i)
            
            # Extend bits to include new vertex
            new_bits.append(new_adj)
            for i in range(new_n - 1):
                if new_adj & (1 << i):
                    new_bits[i] |= (1 << new_vertex)
            
            # Check if valid
            if not _has_K4_bitset(new_n, new_bits) and not _has_indep_6_bitset(new_n, new_bits):
                bits = new_bits
                break
            else:
                # Try to repair
                repaired = _repair_incremental(new_n, new_bits, rng, time_budget=1.0)
                if repaired is not None:
                    bits = repaired
                    break
        else:
            # Could not add vertex in 50 attempts, try again with different seed
            continue
        
        # If we couldn't add the vertex at all, fail
        if len(bits) != new_n:
            return None
    
    return bits

def _repair_incremental(n, bits, rng, time_budget=2.0):
    """
    Repair K4 and Ind6 violations by flipping edges. Focus on edges involving the new vertex
    first, then random edges.
    """
    start = time.time()
    bits = list(bits)
    
    for step in range(2000):
        if time.time() - start > time_budget:
            return None
        
        has_k4 = _has_K4_bitset(n, bits)
        has_ind6 = _has_indep_6_bitset(n, bits)
        
        if not has_k4 and not has_ind6:
            return bits
        
        # Strategy: flip an edge that reduces violation count
        best_flip = None
        best_score = float('inf')
        
        # Try flipping edges involving the highest-index vertex first
        candidates = []
        for v in range(n - 1, -1, -1):
            for u in range(v):
                candidates.append((u, v))
        
        # Sample a subset for speed
        sample = rng.sample(candidates, min(50, len(candidates)))
        
        for u, v in sample:
            had_edge = (bits[u] >> v) & 1
            # Flip
            if had_edge:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            else:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            
            score = _count_K4_bitset(n, bits) + _count_indep_6_bitset(n, bits)
            
            # Unflip
            if had_edge:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            else:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            
            if score < best_score:
                best_score = score
                best_flip = (u, v, had_edge)
        
        if best_flip is not None:
            u, v, had_edge = best_flip
            if had_edge:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            else:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
    
    return None

# ---------------- Main search strategy ----------------

def _search_structured_growth():
    """
    Start from the known 35-vertex graph and try to grow it to 36-40.
    """
    best = {'n': 0, 'edges': []}
    rng = random.Random(123)
    
    # Start from Exoo's 35-vertex graph
    bits35 = _exoo_35()
    
    # Verify it's valid
    if not _has_K4_bitset(35, bits35) and not _has_indep_6_bitset(35, bits35):
        best = {'n': 35, 'edges': _edges_from_bits(35, bits35)}
    else:
        # Fallback: try to construct a valid 35-vertex graph
        for attempt in range(20):
            if time.time() > _DEADLINE - 10:
                break
            bits = [0] * 35
            for i in range(35):
                for j in range(i + 1, 35):
                    if rng.random() < 0.5:
                        bits[i] |= (1 << j)
                        bits[j] |= (1 << i)
            if not _has_K4_bitset(35, bits) and not _has_indep_6_bitset(35, bits):
                best = {'n': 35, 'edges': _edges_from_bits(35, bits)}
                break
    
    # Try to grow from 35 to 36, 37, ...
    for target in range(36, 41):
        if time.time() > _DEADLINE - 15:
            break
        
        # Try multiple times with different random seeds
        for attempt in range(10):
            if time.time() > _DEADLINE - 10:
                break
            
            grown = _try_grow(best['edges'] and _bits_from_edges(best['n'], best['edges']) or bits35,
                              target, rng, time_budget=5.0)
            if grown is not None:
                if not _has_K4_bitset(target, grown) and not _has_indep_6_bitset(target, grown):
                    best = {'n': target, 'edges': _edges_from_bits(target, grown)}
                    break
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    
    # Start with a known valid baseline
    best = {'n': 1, 'edges': []}
    
    # Try Exoo's 35-vertex graph
    try:
        bits35 = _exoo_35()
        if not _has_K4_bitset(35, bits35) and not _has_indep_6_bitset(35, bits35):
            best = {'n': 35, 'edges': _edges_from_bits(35, bits35)}
    except Exception:
        pass
    
    # Try structured growth
    try:
        improved = _search_structured_growth()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        # Fallback to a simple valid graph
        best = {'n': 1, 'edges': []}
    
    return best
