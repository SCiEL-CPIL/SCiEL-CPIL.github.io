import time
import random
import numpy as np
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 100.0

def _check_c5_fast(n, adj):
    """
    Fast C5 check using bitmasks for neighbor sets.
    Returns True if C5 found.
    """
    # Precompute neighbor bitmasks
    nmask = [0] * n
    for i in range(n):
        m = 0
        row = adj[i]
        for j in range(n):
            if row[j] and i != j:
                m |= (1 << j)
        nmask[i] = m
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append((u, v))
    
    for u, v in edges:
        # Find a, b, c such that u-a-b-c-v forms a C5
        # a in N(u) \ {v}, b in N(a) \ {u, v}, c in N(b) \ {a, u, v}, c in N(v)
        nu = nmask[u] & ~(1 << v)
        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            na = nmask[a] & ~(1 << u) & ~(1 << v)
            b_mask = na
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                nb = nmask[b] & ~(1 << a) & ~(1 << u) & ~(1 << v)
                # Check if any c in nb is also in N(v)
                if nb & nmask[v]:
                    return True
    
    return False

def _check_alpha_fast(n, adj, target=9):
    """
    Check if graph has an independent set of size 'target' using branch and bound.
    Uses complement graph clique finding.
    """
    # Build complement adjacency
    comp = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                comp[i, j] = True
                comp[j, i] = True
    
    # Find if there's a clique of size target in comp
    # Use Bron-Kerbosch variant with pivot
    best_clique = [0]
    
    def bronk(R, P, X):
        if time.time() > _DEADLINE - 5:
            return True  # Timeout, assume invalid
        
        if len(R) >= target:
            best_clique[0] = len(R)
            return True
        
        if not P and not X:
            return False
        
        # Pivot selection: choose vertex in P∪X with max degree in P
        U = P | X if hasattr(P, '__or__') else (set(P) | set(X))
        if not U:
            return False
        
        pivot = max(U, key=lambda v: sum(1 for u in P if comp[v, u]))
        candidates = [v for v in P if not comp[v, pivot]]
        
        for v in candidates:
            new_R = R | {v}
            new_P = P & {u for u in P if comp[v, u]}
            new_X = X & {u for u in X if comp[v, u]}
            
            if bronk(new_R, new_P, new_X):
                return True
            
            P = P - {v}
            X = X | {v}
        
        return False
    
    if bronk(frozenset(), frozenset(range(n)), frozenset()):
        return True
    return False

def _validate(n, edges):
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    if _check_c5_fast(n, adj):
        return False
    if _check_alpha_fast(n, adj, 9):
        return False
    return True

def construct():
    start_time = time.time()
    best_n = 32
    best_edges = []
    
    # Initialize best_edges with 8 K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
    
    # Try to find n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 15:
            break
        
        n = target_n
        adj = np.zeros((n, n), dtype=bool)
        
        # Start with 8 K4s
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    adj[base+i, base+j] = True
                    adj[base+j, base+i] = True
        
        # Extra vertices 32..n-1 are isolated initially
        # We need to add edges to reduce alpha from 9+ to <=8
        
        # Strategy: try to connect extra vertices to K4s in ways that avoid C5
        # and reduce independence number
        
        found = False
        max_iters = 50000
        for it in range(max_iters):
            if time.time() > _DEADLINE - 15:
                break
            
            # Pick a random non-edge involving an extra vertex (>=32) or between components
            if random.random() < 0.7 and n > 32:
                u = random.randint(32, n-1)
                v = random.randint(0, 31)
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
            
            if u == v or adj[u, v]:
                continue
            
            # Try adding edge
            adj[u, v] = True
            adj[v, u] = True
            
            # Quick C5 check
            if _check_c5_fast(n, adj):
                adj[u, v] = False
                adj[v, u] = False
                continue
            
            # Alpha check - only do full check if we're close to success
            # Quick heuristic: count isolated vertices
            iso_count = sum(1 for i in range(n) if not adj[i].any())
            if iso_count + 8 < 9:  # Even if all K4s contribute 1 each, we might be ok
                # More careful check
                if not _check_alpha_fast(n, adj, 9):
                    found = True
                    best_n = n
                    new_edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                new_edges.append([i, j])
                    best_edges = new_edges
                    break
            else:
                # Still too many isolated, keep adding edges
                pass
            
            # If alpha check says still bad, remove edge
            if _check_alpha_fast(n, adj, 9):
                adj[u, v] = False
                adj[v, u] = False
                continue
        
        if found:
            continue  # Try larger n
        else:
            break  # Can't find valid graph for this n, stop
    
    return {'n': best_n, 'edges': best_edges}
