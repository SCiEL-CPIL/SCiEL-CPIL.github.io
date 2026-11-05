import time
import random
import sys
from collections import deque
import numpy as np
from itertools import product

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3 with composite groups.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3:
        return _construct_petersen_lift_v2(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift_v2(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in composite groups.
    Uses multi-prime factorization and independent voltage assignments per prime factor.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    # Build numpy matrix for vectorized evaluation
    if len(coeff_list) > 0:
        coeff_matrix = np.array(coeff_list, dtype=np.int64)
        n_constraints = len(coeff_matrix)
    else:
        coeff_matrix = np.zeros((0, 15), dtype=np.int64)
        n_constraints = 0
    
    def is_prime(n):
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
    
    def factorize(n):
        """Return prime factorization as list of (prime, exponent)."""
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                exp = 0
                while n % d == 0:
                    n //= d
                    exp += 1
                factors.append((d, exp))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors
    
    def get_divisors(n):
        """Return all divisors of n."""
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    # Generate candidate composite group sizes: products of small primes
    # We want m such that 10*m is minimized, so try m = p1, p1*p2, p1*p2*p3, etc.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate candidate m values: primes and products of 2-3 small primes
    candidate_ms = set()
    for p in small_primes:
        candidate_ms.add(p)
    for i, p1 in enumerate(small_primes[:15]):
        for p2 in small_primes[i+1:i+10]:
            m = p1 * p2
            if m <= 200:
                candidate_ms.add(m)
    for i, p1 in enumerate(small_primes[:10]):
        for j, p2 in enumerate(small_primes[i+1:i+8]):
            for p3 in small_primes[j+1:j+5]:
                m = p1 * p2 * p3
                if m <= 100:
                    candidate_ms.add(m)
    
    # Sort by n = 10*m to try smallest first
    candidate_ms = sorted(candidate_ms, key=lambda m: 10 * m)
    
    best_n = float('inf')
    best_edges = None
    
    for m in candidate_ms:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Factorize m to get prime factors
        factors = factorize(m)
        n_factors = len(factors)
        
        # For composite m, we assign voltages independently for each prime factor
        # Then combine via CRT
        max_attempts = 100 if n_factors > 1 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # For each prime factor, assign random voltages
            factor_voltages = []
            for p, exp in factors:
                # For prime power p^exp, use Z_p^exp (treat as p for simplicity, or use p^exp)
                p_eff = p ** exp
                vols = np.array([random.randint(1, p_eff - 1) for _ in range(15)], dtype=np.int64)
                factor_voltages.append((p_eff, vols))
            
            # Local repair for each factor independently
            valid = True
            for fi, (p_eff, vols) in enumerate(factor_voltages):
                for repair_step in range(50):
                    if n_constraints > 0:
                        sums = (coeff_matrix @ vols) % p_eff
                        violated = np.where(sums == 0)[0]
                        
                        if len(violated) == 0:
                            break
                        
                        viol_idx = random.choice(violated.tolist())
                        c = coeff_list[viol_idx]
                        candidates = [i for i in range(15) if c[i] != 0]
                        if not candidates:
                            break
                        j = random.choice(candidates)
                        delta = random.randint(1, p_eff - 1)
                        vols[j] = (vols[j] + delta) % p_eff
                        if vols[j] == 0:
                            vols[j] = 1
                else:
                    # Check if still violated after max repairs
                    if n_constraints > 0:
                        sums = (coeff_matrix @ vols) % p_eff
                        if len(np.where(sums == 0)[0]) > 0:
                            valid = False
                            break
            
            if not valid:
                continue
            
            # Combine voltages via CRT
            # voltages[e] = sum of (vols_f[e] * M_f * inv(M_f, p_f)) mod m
            # where M_f = m / p_f
            combined_voltages = np.zeros(15, dtype=np.int64)
            for fi, (p_eff, vols) in enumerate(factor_voltages):
                M_f = m // p_eff
                # Compute inverse of M_f mod p_eff
                inv = pow(M_f, -1, p_eff)
                combined_voltages = (combined_voltages + vols * M_f * inv) % m
            
            # Ensure no zero voltages
            for i in range(15):
                if combined_voltages[i] == 0:
                    combined_voltages[i] = 1
            
            # Verify constraints
            if n_constraints > 0:
                sums = (coeff_matrix @ combined_voltages) % m
                if len(np.where(sums == 0)[0]) > 0:
                    continue
            
            edges = _build_lifted_graph_composite(pet_edges, combined_voltages.tolist(), m)
            return n_candidate, edges
    
    # Fallback to prime-only search
    return _construct_petersen_lift_prime_only(g, deadline, start_time, coeff_list, coeff_matrix, n_constraints)

def _construct_petersen_lift_prime_only(g, deadline, start_time, coeff_list, coeff_matrix, n_constraints):
    """Fallback: search over prime moduli only."""
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    def is_prime(n):
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
    
    best_n = float('inf')
    
    for m in range(53, 200):
        if time.time() > deadline:
            break
        if not is_prime(m):
            continue
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
            
            for repair_step in range(30):
                if n_constraints > 0:
                    sums = (coeff_matrix @ voltages) % m
                    violated = np.where(sums == 0)[0]
                    
                    if len(violated) == 0:
                        edges = _build_lifted_graph_composite(pet_edges, voltages.tolist(), m)
                        return n_candidate, edges
                    
                    viol_idx = random.choice(violated.tolist())
                    c = coeff_list[viol_idx]
                    candidates = [i for i in range(15) if c[i] != 0]
                    if not candidates:
                        break
                    j = random.choice(candidates)
                    delta = random.randint(1, m - 1)
                    voltages[j] = (voltages[j] + delta) % m
                    if voltages[j] == 0:
                        voltages[j] = 1
                else:
                    edges = _build_lifted_graph_composite(pet_edges, voltages.tolist(), m)
                    return n_candidate, edges
    
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph_composite(pet_edges, voltages, m):
    """Build lifted graph with voltages in Z_m (works for both prime and composite m)."""
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    
    return list(edges)

# _build_lifted_graph is replaced by _build_lifted_graph_composite above

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True
