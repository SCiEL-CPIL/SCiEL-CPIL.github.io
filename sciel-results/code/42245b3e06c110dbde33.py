import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_c6_full(adj_bits, n):
    """Check if graph with bitmask adjacency has a 6-cycle.
    adj_bits[i] is an int where bit j is set iff edge (i,j) exists.
    A C6: u-v-w-x-y-z-u with all distinct.
    We look for pairs (u,v) adjacent, then paths of length 4 from v to u avoiding the edge.
    Actually: C6 through edge (u,v): u-v-a-b-c-d-u
    So: d in N(u), a in N(v), b in N(a), c in N(b), d in N(c), d in N(u)
    Equiv: exists d in N(u)\{v}, a in N(v)\{u,d}, b in N(a)\{u,v,d}, c in N(b)\{u,v,a,d}, d in N(c)
    """
    for u in range(n):
        nu = adj_bits[u]
        for v in range(u + 1, n):
            if not (nu >> v) & 1:
                continue
            nv = adj_bits[v]
            # d in N(u) \ {v}
            ds = nu & ~(1 << v)
            while ds:
                d_bit = ds & (-ds)
                d = d_bit.bit_length() - 1
                ds ^= d_bit
                # a in N(v) \ {u, d}
                mask_av = nv & ~(1 << u) & ~(1 << d)
                while mask_av:
                    a_bit = mask_av & (-mask_av)
                    a = a_bit.bit_length() - 1
                    mask_av ^= a_bit
                    # b in N(a) \ {u, v, d}
                    na = adj_bits[a]
                    mask_b = na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                    while mask_b:
                        b_bit = mask_b & (-mask_b)
                        b = b_bit.bit_length() - 1
                        mask_b ^= b_bit
                        # c in N(b) \ {u, v, a, d} and d in N(c)
                        nb = adj_bits[b]
                        mask_c = nb & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        # c must also be adjacent to d
                        mask_c &= adj_bits[d]
                        if mask_c:
                            return True
    return False


def _check_c6_local(adj_bits, n, u, v):
    """Check if adding edge (u,v) creates a C6.
    A new C6 must use the new edge (u,v).
    Form: u-v-a-b-c-d-u where a,b,c,d are distinct and not in {u,v}.
    So: a in N(v), b in N(a), c in N(b), d in N(c) ∩ N(u), all distinct from u,v,a,b,c.
    Equiv: exists d in N(u)\{v}, a in N(v)\{u,d}, b in N(a)\{u,v,d}, c in N(b)∩N(d)\{u,v,a}
    """
    nu = adj_bits[u]
    nv = adj_bits[v]
    # d in N(u) \ {v}
    ds = nu & ~(1 << v)
    while ds:
        d_bit = ds & (-ds)
        d = d_bit.bit_length() - 1
        ds ^= d_bit
        # a in N(v) \ {u, d}
        mask_a = nv & ~(1 << u) & ~(1 << d)
        while mask_a:
            a_bit = mask_a & (-mask_a)
            a = a_bit.bit_length() - 1
            mask_a ^= a_bit
            # b in N(a) \ {u, v, d}
            na = adj_bits[a]
            mask_b = na & ~(1 << u) & ~(1 << v) & ~(1 << d)
            while mask_b:
                b_bit = mask_b & (-mask_b)
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                # c in N(b) ∩ N(d) \ {u, v, a}
                nb = adj_bits[b]
                nd = adj_bits[d]
                mask_c = nb & nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if mask_c:
                    return True
    return False


def _edge_count(adj_bits, n):
    return sum(bin(b).count('1') for b in adj_bits) // 2


def _greedy_build(n=40, seed=42, deadline=0.0):
    """Greedy edge insertion with fast C6 check."""
    random.seed(seed)
    adj_bits = [0] * n
    
    # Start with 8 copies of K5
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    best = _edge_count(adj_bits, n)
    
    # Greedy phase: try adding random edges
    while time.time() < deadline:
        progress = False
        for _ in range(500):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj_bits[u] >> v) & 1:
                continue
            if not _check_c6_local(adj_bits, n, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                progress = True
                ec = _edge_count(adj_bits, n)
                if ec > best:
                    best = ec
        if not progress:
            break
    
    return adj_bits, best


def _local_search(adj_bits, n, deadline, seed=0):
    """Local search: edge swaps and additions."""
    random.seed(seed)
    best = _edge_count(adj_bits, n)
    
    while time.time() < deadline:
        improved = False
        
        # Try random edge additions
        for _ in range(300):
            if time.time() > deadline:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj_bits[u] >> v) & 1:
                continue
            if not _check_c6_local(adj_bits, n, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                ec = _edge_count(adj_bits, n)
                if ec > best:
                    best = ec
                    improved = True
        
        # Try edge swaps: remove one edge, add another
        for _ in range(300):
            if time.time() > deadline:
                break
            # Pick random existing edge
            u = random.randint(0, n - 1)
            nu = adj_bits[u]
            if nu == 0:
                continue
            v_bit = (nu >> random.randint(0, n - 1))
            if not v_bit:
                # pick a random set bit
                v = random.choice([i for i in range(n) if (nu >> i) & 1])
            else:
                v = random.randint(0, n - 1)
            
            # Try to add a new edge
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b or (adj_bits[a] >> b) & 1:
                continue
            if (a, b) == (u, v) or (a, b) == (v, u):
                continue
            
            # Remove (u,v), check if (a,b) is C6-free
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
            if not _check_c6_local(adj_bits, n, a, b):
                adj_bits[a] |= (1 << b)
                adj_bits[b] |= (1 << a)
                ec = _edge_count(adj_bits, n)
                if ec > best:
                    best = ec
                    improved = True
                else:
                    # Revert
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
            else:
                # Revert the addition attempt, keep (u,v) removed? No, restore
                pass
            
            # Restore (u,v)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        if not improved:
            # Try a different random seed for diversity
            seed += 1
            if seed > 10:
                break
    
    return adj_bits, best


def _to_edges(adj_bits, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_bits[u] >> v) & 1:
                edges.append([u, v])
    return edges


def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    n = 40
    best_edges = None
    best_count = 0
    
    # Multiple restarts with different seeds
    for seed in [42, 123, 7, 99, 2024]:
        if time.time() > deadline - 20.0:
            break
        
        try:
            adj_bits, count = _greedy_build(n, seed, deadline - 15.0)
            
            # Local search
            adj_bits, count = _local_search(adj_bits, n, deadline - 10.0, seed)
            
            edges = _to_edges(adj_bits, n)
            
            # Validate
            if not _has_c6_full(adj_bits, n):
                if count > best_count:
                    best_count = count
                    best_edges = edges
        except Exception:
            continue
    
    if best_edges is None or best_count < 80:
        # Fallback: 8 copies of K5
        return _fallback()
    
    return best_edges


def _fallback():
    """8 disjoint copies of K5 = 80 edges."""
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])
    return edges

# EVOLVE-BLOCK-END
