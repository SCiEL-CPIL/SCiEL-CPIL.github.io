"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_k222(adj, n):
    """Check if graph with adjacency bitmasks adj on n vertices contains K2,2,2 as a subgraph."""
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            cnt = cn_mask.bit_count()
            if cnt < 4:
                continue
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            L = len(cn_list)
            for i in range(L):
                u = cn_list[i]
                for j in range(i + 1, L):
                    v = cn_list[j]
                    common_uv = adj[u] & adj[v] & cn_mask
                    if common_uv.bit_count() >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222(comp, n):
        return False
    return True

def _build_paley(q):
    """Build Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_k222_violations(adj, n):
    """Count the number of K2,2,2 subgraphs in the graph (for local search objective)."""
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            cnt = cn_mask.bit_count()
            if cnt < 4:
                continue
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            L = len(cn_list)
            for i in range(L):
                u = cn_list[i]
                for j in range(i + 1, L):
                    v = cn_list[j]
                    common_uv = adj[u] & adj[v] & cn_mask
                    c = common_uv.bit_count()
                    if c >= 2:
                        count += 1
    return count

def _total_violations(adj, n):
    """Total K2,2,2 violations in both color classes."""
    v1 = _count_k222_violations(adj, n)
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    v2 = _count_k222_violations(comp, n)
    return v1 + v2

def _flip_edge(adj, n, u, v):
    """Flip the edge between u and v."""
    adj[u] ^= (1 << v)
    adj[v] ^= (1 << u)

def _local_search(adj, n, deadline, rng):
    """Local search: flip edges to reduce total K2,2,2 violations."""
    current_viol = _total_violations(adj, n)
    if current_viol == 0:
        return adj, 0
    
    best_adj = adj[:]
    best_viol = current_viol
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random edge to flip
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        _flip_edge(adj, n, u, v)
        new_viol = _total_violations(adj, n)
        
        if new_viol < best_viol:
            best_viol = new_viol
            best_adj = adj[:]
            no_improve = 0
        else:
            _flip_edge(adj, n, u, v)  # revert
            no_improve += 1
        
        if best_viol == 0:
            break
    
    return best_adj, best_viol

def _try_extend_with_search(adj, n, target_n, deadline, rng):
    """Try to extend coloring to target_n vertices using local search."""
    if n >= target_n:
        return adj, n
    
    # Initialize new vertices with random colors
    best_adj = adj[:]
    best_n = n
    best_viol = _total_violations(best_adj, n)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        candidate = best_adj[:]
        # Add new vertices with random edges
        for new_v in range(n, target_n):
            for old_v in range(new_v):
                if rng.random() < 0.5:
                    candidate[new_v] |= (1 << old_v)
                    candidate[old_v] |= (1 << new_v)
        
        # Run local search on the extended graph
        candidate, viol = _local_search(candidate, target_n, deadline, rng)
        
        if viol < best_viol:
            best_adj = candidate
            best_n = target_n
            best_viol = viol
        
        if best_viol == 0:
            break
    
    if best_viol == 0:
        return best_adj, best_n
    return adj, n

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    rng = random.Random(42)
    
    # Try Paley graphs for primes ≡ 1 mod 4
    paley_primes = [29, 17, 13, 5]
    best_adj = None
    best_n = 0
    
    for q in paley_primes:
        if time.time() > deadline:
            break
        adj = _build_paley(q)
        if _is_valid(adj, q):
            best_adj = adj
            best_n = q
            break
    
    if best_n == 0:
        best_adj = _build_paley(5)
        best_n = 5
    
    # Try to extend from best valid coloring to 30
    if best_n < 30:
        extended_adj, extended_n = _try_extend_with_search(best_adj, best_n, 30, deadline, rng)
        if extended_n > best_n:
            best_adj = extended_adj
            best_n = extended_n
    
    # Try to extend to 31
    if best_n < 31:
        extended_adj, extended_n = _try_extend_with_search(best_adj, best_n, 31, deadline, rng)
        if extended_n > best_n:
            best_adj = extended_adj
            best_n = extended_n
    
    # Final validation
    if best_n > 5:
        if not _is_valid(best_adj, best_n):
            # Fallback to known valid
            best_adj = _build_paley(29)
            best_n = 29
    
    # Build colors list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
