"""Strategy: Structured values (powers of 2 and 3) for product collision, fixed dense graph for sum collision."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 14
    m = 35
    
    # Values: Powers of 2 and 3.
    # a*2^i * b*3^j = (ab) * 2^i * 3^j.
    # If we pick distinct primes a, b, then products are unique per pair (i,j).
    # To get collisions, we need same (i,j) sums? No, products are a_i * b_j.
    # Let's use: V = {2^i * 3^j} for i,j in small ranges.
    # 2^0*3^0=1, 2^1*3^0=2, 2^2*3^0=4, 2^3*3^0=8, 2^0*3^1=3, 2^1*3^1=6, 2^2*3^1=12, 2^3*3^1=24
    # 2^4*3^0=16, 2^0*3^2=9, 2^1*3^2=18, 2^2*3^2=36, 2^3*3^2=72, 2^4*3^1=48
    # These are 14 distinct integers.
    # Products: (2^i 3^j) * (2^k 3^l) = 2^(i+k) 3^(j+l).
    # Collision if i+k == i'+k' and j+l == j'+l'.
    # With 14 values, we can have many collisions.
    
    base_vals = [1, 2, 4, 8, 3, 6, 12, 24, 16, 9, 18, 36, 72, 48]
    # Check distinctness and <= 10000
    assert len(set(base_vals)) == 14
    assert all(v <= 10000 and v > 0 for v in base_vals)
    
    values = base_vals
    
    # Graph: We want to minimize distinct sums.
    # Sums of 2^i 3^j + 2^k 3^l.
    # If we connect nodes with similar magnitudes, sums are distinct.
    # If we connect small to large, sums are distinct.
    # Actually, sums of distinct powers of 2 (with 3 factors) are likely distinct unless there's carry-over structure, but 3 is not power of 2.
    # However, we have 35 edges. Max distinct sums is 35.
    # Can we force collisions?
    # 1+8=9, 3+6=9. (indices 0,3 and 4,5)
    # 2+4=6 (index 1,2) - wait, 6 is a value, but sum is 6.
    # 1+2=3, 1+3=4, 1+4=5...
    # Let's just pick the 35 edges that yield the minimum number of distinct sums.
    # Since n=14, total pairs = 91.
    # We need to select 35 edges.
    # We can compute all pair sums and pick the 35 edges with the most frequent sums?
    # No, we need to pick a set of 35 edges.
    # Let's compute all 91 sums.
    # Then we want to pick 35 edges such that the number of distinct sums in that subset is minimized.
    # This is a combinatorial optimization.
    # Heuristic: Pick edges that share sums.
    
    all_edges = list(itertools.combinations(range(n), 2))
    sums = {}
    for u, v in all_edges:
        s = values[u] + values[v]
        if s not in sums:
            sums[s] = []
        sums[s].append((u, v))
    
    # Sort sums by frequency descending
    sorted_sums = sorted(sums.items(), key=lambda x: -len(x[1]))
    
    selected_edges = []
    used = set()
    
    # Greedily add edges from most frequent sums
    # But we must ensure we can pick 35 distinct edges.
    # If we just take all edges from the top sum, then next, etc.
    for s, edges in sorted_sums:
        for e in edges:
            if e not in used:
                selected_edges.append(e)
                used.add(e)
                if len(selected_edges) == m:
                    break
        if len(selected_edges) == m:
            break
    
    # Check if we have 35 edges
    if len(selected_edges) < m:
        # Fill with remaining
        for e in all_edges:
            if e not in used:
                selected_edges.append(e)
                used.add(e)
                if len(selected_edges) == m:
                    break
    
    edges = selected_edges[:m]
    
    # Verify connectivity? The problem doesn't require connectivity, only "simple undirected edges".
    # "Minimize max(number of distinct edge sums, number of distinct edge products)."
    # It does NOT say the graph must be connected.
    # So we can ignore connectivity.
    
    # Calculate raw value
    distinct_sums = set()
    distinct_prods = set()
    for u, v in edges:
        distinct_sums.add(values[u] + values[v])
        distinct_prods.add(values[u] * values[v])
        
    raw = max(len(distinct_sums), len(distinct_prods))
    
    # Return types must be int and list
    vals = [int(x) for x in values]
    ed = [[int(u), int(v)] for u, v in edges]
    
    return {'values': vals, 'edges': ed}
# EVOLVE-BLOCK-END
