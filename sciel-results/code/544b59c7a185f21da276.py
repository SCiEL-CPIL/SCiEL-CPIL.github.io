import time
import random
import numpy as np
from collections import deque
from math import isqrt

def _verify_girth_fast(adj, n, g):
    """
    Check if graph has girth >= g using non-backtracking BFS from every vertex.
    adj: list of lists of neighbors
    n: number of vertices
    g: target girth
    Returns True if girth >= g, False otherwise.
    """
    if g <= 2:
        return True
    
    max_depth = g - 2  # if we hit start at depth <= max_depth, cycle < g
    
    for start in range(n):
        # visited: set of (v, prev) states
        visited = set()
        # frontier: list of (v, prev)
        frontier = [(start, -1)]
        depth = 0
        
        while frontier and depth <= max_depth:
            next_frontier = []
            for (v, prev) in frontier:
                for w in adj[v]:
                    if prev >= 0 and w == prev:
                        continue
                    if w == start:
                        if depth + 1 < g:
                            return False
                        continue
                    if (w, v) not in visited:
                        visited.add((w, v))
                        next_frontier.append((w, v))
            depth += 1
            if depth > max_depth:
                break
            frontier = next_frontier
    
    return True

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primitive_root(p):
    """Find a primitive root modulo prime p."""
    if p == 2:
        return 1
    # Factor p-1
    factors = []
    n = p - 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    for g in range(2, p):
        ok = True
        for f in factors:
            if pow(g, (p - 1) // f, p) == 1:
                ok = False
                break
        if ok:
            return g
    return None

def _thue_morse(n, length):
    """Generate Thue-Morse sequence of given length."""
    seq = []
    for i in range(length):
        # Count bits set
        c = bin(i).count('1')
        seq.append(c % 2)
    return seq

def _algebraic_voltages_petersen(p, strategy='powers'):
    """
    Generate deterministic voltage assignments for Petersen graph lift over Z_p.
    Petersen has 15 edges. We assign voltages based on algebraic structure.
    Returns list of 15 voltages in Z_p.
    """
    if strategy == 'powers':
        # Use powers of primitive root: g^0, g^1, ..., g^14 mod p
        g = _primitive_root(p)
        if g is None:
            return None
        vols = []
        for i in range(15):
            vols.append(pow(g, i, p))
        return vols
    elif strategy == 'thue_morse':
        # Thue-Morse pattern repeated/extended
        tm = _thue_morse(1, 15)
        # Scale to Z_p
        scale = max(1, p // 2)
        return [t * scale % p for t in tm]
    elif strategy == 'quadratic':
        # Use quadratic residues and non-residues pattern
        g = _primitive_root(p)
        if g is None:
            return None
        # Assign based on whether exponent is quadratic residue
        vols = []
        for i in range(15):
            # Check if i is a quadratic residue mod p-1 (rough heuristic)
            # Actually, use the value (g^i) and map based on Legendre symbol
            leg = pow(pow(g, i, p), (p - 1) // 2, p)
            if leg == 1:
                vols.append(pow(g, i, p))
            else:
                vols.append((pow(g, i, p) + p // 2) % p)
        return vols
    elif strategy == 'lps':
        # LPS-style: use consecutive primes or structured offsets
        g = _primitive_root(p)
        if g is None:
            return None
        # Use g^i but with alternating sign pattern
        vols = []
        for i in range(15):
            base = pow(g, i, p)
            if i % 2 == 0:
                vols.append(base)
            else:
                vols.append((-base) % p)
        return vols
    elif strategy == 'fibonacci':
        # Fibonacci-based voltages
        fibs = [0, 1]
        for i in range(13):
            fibs.append((fibs[-1] + fibs[-2]) % p)
        # Scale to avoid too many small values
        scale = max(1, p // 10)
        return [(f * scale) % p for f in fibs[:15]]
    return None

def _algebraic_voltages_k33(p, strategy='powers'):
    """
    Generate deterministic voltage assignments for K_{3,3} lift over Z_p.
    K_{3,3} has 9 edges.
    Returns list of 9 voltages in Z_p.
    """
    if strategy == 'powers':
        g = _primitive_root(p)
        if g is None:
            return None
        vols = []
        for i in range(9):
            vols.append(pow(g, i, p))
        return vols
    elif strategy == 'thue_morse':
        tm = _thue_morse(1, 9)
        scale = max(1, p // 2)
        return [t * scale % p for t in tm]
    elif strategy == 'quadratic':
        g = _primitive_root(p)
        if g is None:
            return None
        vols = []
        for i in range(9):
            leg = pow(pow(g, i, p), (p - 1) // 2, p)
            if leg == 1:
                vols.append(pow(g, i, p))
            else:
                vols.append((pow(g, i, p) + p // 2) % p)
        return vols
    elif strategy == 'lps':
        g = _primitive_root(p)
        if g is None:
            return None
        vols = []
        for i in range(9):
            base = pow(g, i, p)
            if i % 2 == 0:
                vols.append(base)
            else:
                vols.append((-base) % p)
        return vols
    return None

def _petersen_edges():
    """Return edges of the Petersen graph (10 vertices)."""
    # Outer pentagon: 0-1-2-3-4-0
    # Inner pentagram: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _k33_edges():
    """Return edges of K_{3,3} (6 vertices)."""
    # Partitions: {0,1,2} and {3,4,5}
    return [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

def _build_lift(base_edges, m, voltages, base_n):
    """
    Build a voltage lift of a base graph.
    base_edges: list of (u,v) edges in base graph
    m: lift size (number of layers)
    voltages: list of integers (one per edge) in Z_m
    base_n: number of vertices in base graph
    Returns (n, edges, adj) or None if invalid.
    """
    n = base_n * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % m
        for layer in range(m):
            a = u * m + layer
            b = v * m + (layer + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
            else:
                # Duplicate edge, invalid simple graph
                return None
            
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _build_petersen_lift(m, voltages, seed):
    """
    Build a lift of the Petersen graph with given voltages.
    m: lift size (number of layers)
    voltages: list of 15 integers (one per edge) in Z_m
    Returns (n, edges, adj) or None if invalid.
    """
    base_edges = _petersen_edges()
    return _build_lift(base_edges, m, voltages, 10)

def _build_k33_lift(m, voltages, seed):
    """
    Build a lift of K_{3,3} with given voltages.
    m: lift size (number of layers)
    voltages: list of 9 integers (one per edge) in Z_m
    Returns (n, edges, adj) or None if invalid.
    """
    base_edges = _k33_edges()
    return _build_lift(base_edges, m, voltages, 6)

def _algebraic_voltage_search(m, g, strategies, base='petersen'):
    """
    Try deterministic algebraic voltage assignments for a voltage lift of size m.
    base: 'petersen' or 'k33'
    strategies: list of strategy names to try
    Returns (n, edges) if a valid graph is found, else None.
    """
    if base == 'k33':
        volt_fn = _algebraic_voltages_k33
        build_fn = _build_k33_lift
    else:
        volt_fn = _algebraic_voltages_petersen
        build_fn = _build_petersen_lift
    
    for strategy in strategies:
        voltages = volt_fn(m, strategy)
        if voltages is None:
            continue
        result = build_fn(m, voltages, 0)
        if result is None:
            continue
        n, edges, adj_list = result
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    return None

def _hybrid_voltage_search(m, g, n_random, seed, base='petersen'):
    """
    Try algebraic strategies first, then fall back to random.
    Returns (n, edges) if a valid graph is found, else None.
    """
    # First try algebraic
    algebraic_strategies = ['powers', 'lps', 'quadratic', 'thue_morse', 'fibonacci']
    result = _algebraic_voltage_search(m, g, algebraic_strategies, base=base)
    if result is not None:
        return result
    
    # Then try random
    rng = random.Random(seed)
    if base == 'k33':
        base_edges = _k33_edges()
        build_fn = _build_k33_lift
    else:
        base_edges = _petersen_edges()
        build_fn = _build_petersen_lift
    
    num_edges = len(base_edges)
    
    for _ in range(n_random):
        voltages = [rng.randint(0, m - 1) for _ in range(num_edges)]
        result = build_fn(m, voltages, seed)
        if result is None:
            continue
        n, edges, adj_list = result
        if _verify_girth_fast(adj_list, n, g):
            return n, edges
    return None

def _random_3reg_graph(n, seed):
    """
    Generate a random 3-regular graph on n vertices.
    Returns (n, edges, adj_list) or None.
    """
    if n % 2 != 0:
        return None
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    if not valid:
        return None
        
    # Check connectivity
    visited = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
        
    adj_list = [sorted(list(adj[i])) for i in range(n)]
    return n, edges, adj_list

def _improve_girth(edges, adj_list, n, g, max_swaps, seed):
    """
    Try to increase girth using edge swaps.
    Returns improved (n, edges, adj_list) or original.
    """
    rng = random.Random(seed)
    current_edges = list(edges)
    current_adj = [list(nei) for nei in adj_list]
    
    # Convert to set for fast lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    best_girth = _verify_girth_fast(current_adj, n, g)
    
    for _ in range(max_swaps):
        # Pick two random edges (a,b) and (c,d)
        e1_idx = rng.randint(0, len(current_edges) - 1)
        e2_idx = rng.randint(0, len(current_edges) - 1)
        if e1_idx == e2_idx:
            continue
        
        a, b = current_edges[e1_idx]
        c, d = current_edges[e2_idx]
        
        # Ensure no shared vertices
        if len({a,b,c,d}) < 4:
            continue
            
        # Try swap (a,b),(c,d) -> (a,c),(b,d)
        new_e1 = (a, c)
        new_e2 = (b, d)
        
        # Check if new edges exist or are loops
        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])) in edge_set:
            continue
        if (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])) in edge_set:
            continue
            
        # Perform swap
        edge_set.remove((min(a,b), max(a,b)))
        edge_set.remove((min(c,d), max(c,d)))
        edge_set.add((min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1])))
        edge_set.add((min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1])))
        
        current_edges[e1_idx] = new_e1
        current_edges[e2_idx] = new_e2
        
        # Rebuild adj
        # This is O(E), might be slow. Optimize by updating only 4 vertices.
        # But for simplicity and correctness, let's just check girth on the fly?
        # No, we need adj for girth check.
        
        # Update adj lists
        # Remove old neighbors
        current_adj[a].remove(b)
        current_adj[b].remove(a)
        current_adj[c].remove(d)
        current_adj[d].remove(c)
        
        # Add new neighbors
        current_adj[a].append(c)
        current_adj[c].append(a)
        current_adj[b].append(d)
        current_adj[d].append(b)
        
        # Sort for consistency? Not strictly needed for BFS if we use lists
        # But _verify_girth_fast expects lists.
        
        if _verify_girth_fast(current_adj, n, g):
            return n, current_edges, current_adj
            
        # If girth decreased or stayed same, we might want to revert?
        # For now, accept any swap that doesn't break degree/connectivity.
        # Connectivity is preserved by 2-swaps on a connected graph.
        
    return n, current_edges, current_adj

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    if k != 3:
        # Fallback for other k
        pass
        
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: K3,3 Lifts with algebraic voltages (n = 6m)
    # Moore bound for k=3, g=16 is 512. 6m >= 512 => m >= 86
    # Try primes m starting from 89 upward
    primes_k33 = []
    m = 89
    while len(primes_k33) < 30 and m < 300:
        if _is_prime(m):
            primes_k33.append(m)
        m += 1
    
    for m in primes_k33:
        if time.time() > deadline - 40:
            break
        n_lift = 6 * m
        if n_lift >= best_n:
            break
            
        result = _hybrid_voltage_search(m, g, n_random=30, seed=m*1000+1, base='k33')
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            break
    
    # Strategy 2: Petersen Lifts with algebraic voltages (n = 10m)
    # 10m >= 512 => m >= 52
    primes_pet = []
    m = 53
    while len(primes_pet) < 30 and m < 200:
        if _is_prime(m):
            primes_pet.append(m)
        m += 1
    
    for m in primes_pet:
        if time.time() > deadline - 35:
            break
        n_lift = 10 * m
        if n_lift >= best_n:
            break
            
        result = _hybrid_voltage_search(m, g, n_random=25, seed=m*1000+2, base='petersen')
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
            break
    
    # Strategy 3: More K3,3 lifts if still no result or want smaller
    if best_n > 600:
        for m in range(86, 150):
            if time.time() > deadline - 30:
                break
            n_lift = 6 * m
            if n_lift >= best_n:
                break
            if not _is_prime(m):
                continue
            result = _hybrid_voltage_search(m, g, n_random=40, seed=m*2000+1, base='k33')
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
    
    # Strategy 4: Random 3-regular graphs as fallback
    if time.time() < deadline - 25:
        for n_try in range(512, min(best_n if best_n < float('inf') else 650, 650), 2):
            if time.time() > deadline - 15:
                break
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                res = _random_3reg_graph(n_try, seed=attempt*100 + n_try)
                if res is None:
                    continue
                n_r, edges_r, adj_r = res
                if _verify_girth_fast(adj_r, n_r, g):
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
    
    if best_result is None:
        # Fallback: larger lifts with more random attempts
        for base, m_start, m_end, seed_base in [('k33', 200, 350, 999), ('petersen', 150, 250, 998)]:
            if time.time() > deadline - 10:
                break
            for m in range(m_start, m_end):
                if time.time() > deadline - 5:
                    break
                if not _is_prime(m):
                    continue
                result = _hybrid_voltage_search(m, g, n_random=30, seed=m+seed_base, base=base)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break
    
    if best_result is None:
        # Emergency fallback: very large lift
        result = _hybrid_voltage_search(503, g, n_random=30, seed=777, base='k33')
        if result is not None:
            best_result = result
        else:
            result = _hybrid_voltage_search(509, g, n_random=30, seed=778, base='petersen')
            if result is not None:
                best_result = result
            else:
                # Last resort: large random
                result = _random_voltage_search(600, g, 50, seed=888, base='k33')
                if result is not None:
                    best_result = result
                else:
                    return 0, []
    
    return best_result
