"""Deterministic voltage lift of Petersen graph with algebraic voltage assignments."""
import time
import sys
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Petersen graph: 10 vertices, 15 edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    # base_edges has 15 entries, indexed 0..14

    # Moore bound for k=3, g=16 is 512. n = 10*m, so m >= 52.
    primes = _sieve_primes(3000)
    primes = [p for p in primes if p >= 53]

    best_result = None
    best_n = float('inf')

    for p in primes:
        if time.time() > deadline:
            break
        if p * 10 > best_n:
            continue

        # Try multiple voltage assignment strategies
        strategies = _get_voltage_strategies(p)
        
        for volts in strategies:
            if time.time() > deadline:
                break
            n, edges = _build_lift(base_edges, p, volts)
            if _verify_girth_nonbacktracking(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_result = (n, edges)
                break  # Found valid for this p, move to next prime

        # Also try a few random assignments for this prime
        if best_result is None or best_n > p * 10:
            for _ in range(3):
                if time.time() > deadline:
                    break
                volts = [random.randint(1, p-1) for _ in range(15)]
                n, edges = _build_lift(base_edges, p, volts)
                if _verify_girth_nonbacktracking(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break

    if best_result is not None:
        return best_result

    # Fallback
    p = 3001
    volts = [random.randint(1, p-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, p, volts)
    return n, edges


def _sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i in range(2, limit+1) if is_prime[i]]


def _get_voltage_strategies(p):
    """Generate several deterministic voltage assignment strategies."""
    strategies = []
    
    # Strategy 1: powers of 2
    volts1 = [pow(2, e, p) for e in range(15)]
    if all(v != 0 for v in volts1):
        strategies.append(volts1)
    
    # Strategy 2: powers of 3
    volts2 = [pow(3, e, p) for e in range(15)]
    if all(v != 0 for v in volts2):
        strategies.append(volts2)
    
    # Strategy 3: linear: v_e = e+1
    volts3 = [(e + 1) % p for e in range(15)]
    strategies.append(volts3)
    
    # Strategy 4: quadratic: v_e = (e+1)^2 mod p
    volts4 = [((e+1)*(e+1)) % p for e in range(15)]
    if all(v != 0 for v in volts4):
        strategies.append(volts4)
    
    # Strategy 5: find a primitive root a, use a^e
    a = _find_primitive_root(p)
    if a is not None:
        volts5 = [pow(a, e, p) for e in range(15)]
        if all(v != 0 for v in volts5):
            strategies.append(volts5)
        # Strategy 6: a^(e+1)
        volts6 = [pow(a, e+1, p) for e in range(15)]
        if all(v != 0 for v in volts6):
            strategies.append(volts6)
    
    return strategies


def _find_primitive_root(p):
    """Find a primitive root modulo prime p."""
    if p == 2:
        return 1
    n = p - 1
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    for a in range(2, p):
        is_gen = True
        for q in factors:
            if pow(a, (p-1)//q, p) == 1:
                is_gen = False
                break
        if is_gen:
            return a
    return None


def _build_lift(base_edges, m, voltages):
    n = 10 * m
    edges = []
    for idx, ((u, v), shift) in enumerate(zip(base_edges, voltages)):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_nonbacktracking(n, edges, g):
    """
    Verify girth >= g using non-backtracking walks.
    For each vertex s, do BFS on non-backtracking walks up to depth g-1.
    If we return to s at distance L < g, there's a cycle of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    k_actual = 3
    # Precompute reverse indices: rev_idx[u][v] = index of v in adj[u]
    # For speed, use a dictionary per vertex or a small array
    # Since k=3, we can just search
    rev_idx = [{} for _ in range(n)]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            rev_idx[u][v] = i
    
    # State: (u, in_idx) where in_idx is index in adj[u] of previous vertex, or k_actual for start
    total_states = n * (k_actual + 1)
    dist = np.full(total_states, -1, dtype=np.int32)
    
    for s in range(n):
        dist.fill(-1)
        start_id = s * (k_actual + 1) + k_actual
        dist[start_id] = 0
        queue = deque([start_id])
        
        while queue:
            state_id = queue.popleft()
            d = dist[state_id]
            if d >= g - 1:
                continue
            
            u = state_id // (k_actual + 1)
            in_idx = state_id % (k_actual + 1)
            
            if in_idx == k_actual:
                allowed = range(k_actual)
            else:
                allowed = [j for j in range(k_actual) if j != in_idx]
            
            for j in allowed:
                v = adj[u][j]
                u_in_v = rev_idx[v].get(u, -1)
                if u_in_v == -1:
                    continue
                new_id = v * (k_actual + 1) + u_in_v
                if dist[new_id] == -1:
                    dist[new_id] = d + 1
                    if v == s and d + 1 < g:
                        return False
                    queue.append(new_id)
    
    return True

# EVOLVE-BLOCK-END

