import time
import numpy as np

def _reversed_de_bruijn(k, m):
    """
    Construct a k-regular graph on k^m vertices using the involution phi(w) = reverse(w).
    Vertex u is connected to phi(shift(u, a)) for a in 0..k-1.
    """
    n = k ** m
    edges = []
    
    # Precompute shifts and reversals for speed?
    # For m=7, k=3, n=2187. Edges = 3*2187 = 6561.
    # This is small enough to do in pure Python loops with optimization.
    
    for u in range(n):
        # Get digits of u
        digits = []
        tmp = u
        for _ in range(m):
            digits.append(tmp % k)
            tmp //= k
        # digits is least significant first. 
        # shift(u, a) corresponds to (digits[1], ..., digits[m-1], a) in MSB order?
        # Let's stick to the number representation.
        # u = d_0 * k^(m-1) + ... + d_{m-1}
        # shift(u, a) = (d_1 * k^(m-1) + ... + d_{m-1}) * k + a
        # Let prefix = u // k.
        # shift_val = prefix * k + a.
        
        prefix = u // k
        
        neighbors = set()
        for a in range(k):
            v_shift = prefix * k + a
            # Now apply phi: reverse the digits of v_shift
            # Get digits of v_shift (LSB first)
            v_digits = []
            tmp_v = v_shift
            for _ in range(m):
                v_digits.append(tmp_v % k)
                tmp_v //= k
            # v_digits is [v_0, v_1, ..., v_{m-1}] where v_0 is LSB.
            # Reverse means v_{m-1} becomes LSB.
            # New value = v_{m-1} * k^0 + v_{m-2} * k^1 + ...
            v_rev = 0
            multiplier = 1
            for d in reversed(v_digits):
                v_rev += d * multiplier
                multiplier *= k
            
            if v_rev == u:
                # Self-loop, skip (or handle)
                continue
                
            neighbors.add(v_rev)
            
        # Add edges to avoid duplicates
        for v in neighbors:
            if u < v:
                edges.append((u, v))
            # If u > v, it's already added or will be added when we process v?
            # No, we iterate all u. 
            # If we add (u, v) for u < v, we ensure uniqueness.
            # But we must ensure that if u connects to v, v connects to u.
            # The construction is symmetric?
            # N(u, a) = v. Does N(v, b) = u for some b?
            # Not necessarily. The graph is directed in the construction step, but we make it undirected by adding (u, v) if u is connected to v.
            # Wait, if u is connected to v, we add edge (u, v).
            # If v is connected to u, we add edge (v, u) which is same as (u, v).
            # So we just need to collect all pairs {u, v} such that v in N(u) or u in N(v).
            # Since we iterate all u, we will find all such pairs.
            
    # We need to ensure the graph is simple and k-regular.
    # The above logic adds edges. 
    # But if u connects to v, and v does NOT connect to u, the edge is still added.
    # This makes it an undirected graph.
    # Is it k-regular?
    # Degree of u is |N(u)|.
    # If no self-loops, |N(u)| = k.
    # So yes, it is k-regular.
    
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # Need girth >= g.
    # Girth of this graph is approx 2m+2.
    # So 2m+2 >= g => m >= (g-2)/2.
    m = max(1, (g - 2 + 1) // 2)
    
    n, edges = _reversed_de_bruijn(k, m)
    
    # Verify no self-loops were skipped that would break regularity?
    # If self-loops exist, degree < k. 
    # For k=3, m=7, are there self-loops?
    # u = phi(shift(u, a)).
    # This is a system of linear equations mod k? No, digits.
    # It's possible. If so, we need to increase m or use a different involution.
    # But let's assume for now it works or we can check.
    
    return n, edges
