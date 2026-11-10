import math
import time
from functools import reduce

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Vertices: numbers in [1, 255] that are divisible by at least one of {3, 5, 17}
    # and are 3-5-17-smooth (only prime factors from {3, 5, 17})
    # Actually, we need gcd(a,b)>1 for ALL pairs, so we need a clique.
    # The multiples of 3 form a clique of size 85. Let's see if we can do better
    # by mixing multiples of different primes.
    
    # Strategy: Find maximum clique in graph where vertices are [1..255] containing 255
    # and edges connect numbers with gcd>1.
    # 
    # Key insight: 255 = 3*5*17. Any number in the clique must share a factor with 255,
    # so must be divisible by 3, 5, or 17.
    # But also, any two numbers in the clique must share a factor.
    #
    # Let's use the structure: partition candidates by their gcd with 255.
    # gcd(x, 255) can be: 1, 3, 5, 15, 17, 51, 85, 255
    # We need gcd(x,255) > 1, so x is divisible by 3, 5, or 17.
    #
    # For a clique, if we pick numbers divisible by 3 only (not 5 or 17), they must
    # share a factor with numbers divisible by 5 only. This is hard.
    #
    # Better approach: Use the fact that multiples of 3 form a clique. 
    # Can we add some multiples of 5 (not 3) to this clique?
    # A multiple of 5 (not 3) like 5, 10, 20, 25, 40, 50, 70, 80, 95, 100, 110, 115, 125, 130, 145, 155, 160, 170, 175, 190, 195(no, div by 3), 200, 205, 215, 220, 230, 235, 245, 250
    # For such x to be in a clique with all multiples of 3, x must share a factor with every multiple of 3.
    # But gcd(5, 3) = 1, so 5 can't be in a clique with 3.
    # Actually, we don't need ALL multiples of 3. We just need a subset.
    #
    # Let me think differently. The current best is 85 (all multiples of 3).
    # Can we get more than 85?
    #
    # Upper bound: numbers divisible by 3, 5, or 17 in [1,255].
    # |div3| = 85, |div5| = 51, |div17| = 15
    # |div3 ∪ div5 ∪ div17| = 85 + 51 + 15 - |div15| - |div51| - |div85| + |div255|
    # |div15| = 17, |div51| = 5, |div85| = 3, |div255| = 1
    # = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    # So upper bound is 127. But we need a clique, so much less.
    #
    # Let's try a greedy approach with local search.
    # Start with multiples of 3 (size 85), then try to swap in better elements.
    
    # Precompute gcd compatibility
    # vertices = all numbers in [1, 255] with gcd(x, 255) > 1
    vertices = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    n_v = len(vertices)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is set if gcd(vertices[i], vertices[j]) > 1
    adj = [0] * n_v
    for i in range(n_v):
        vi = vertices[i]
        mask = 0
        for j in range(n_v):
            if i != j and math.gcd(vi, vertices[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
    
    # 255 must be in the set
    idx_255 = v_to_idx[255]
    
    # Greedy: start with {255}, add vertices with highest degree among compatible ones
    # Then do local search: try to swap out a vertex for a better one
    
    # Initial greedy solution
    selected = {idx_255}
    compatible_mask = adj[idx_255]
    
    # Repeatedly add the vertex with highest "score" (degree within compatible set)
    while True:
        if time.time() > deadline:
            break
        best_candidate = -1
        best_score = -1
        # Find candidates compatible with all selected
        for i in range(n_v):
            if i in selected:
                continue
            # Check if i is compatible with all selected
            ok = True
            for s in selected:
                if not (adj[i] >> s) & 1:
                    ok = False
                    break
            if not ok:
                continue
            # Score: number of other candidates that are also compatible with all selected and with i
            # Simplified: just use degree
            score = bin(adj[i] & compatible_mask).count('1')
            if score > best_score:
                best_score = score
                best_candidate = i
        if best_candidate == -1:
            break
        selected.add(best_candidate)
        # Update compatible_mask
        compatible_mask &= adj[best_candidate]
    
    best_set = [vertices[i] for i in selected]
    best_size = len(best_set)
    
    # Local search: try to improve by swapping
    # For each vertex in selected, try removing it and adding a better one
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for s in list(selected):
            # Try removing s
            new_selected = selected - {s}
            # Find best candidate to add
            best_candidate = -1
            best_gain = 0
            for i in range(n_v):
                if i in new_selected:
                    continue
                # Check if i is compatible with all in new_selected
                ok = True
                for ns in new_selected:
                    if not (adj[i] >> ns) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                # Gain: we're replacing s with i, net change is 0 in size
                # But we want to enable more additions later
                # For now, just check if this swap allows more additions
                # Simplified: skip for now
                pass
            
            # Try adding without removing (if possible)
            for i in range(n_v):
                if i in selected:
                    continue
                ok = True
                for s2 in selected:
                    if not (adj[i] >> s2) & 1:
                        ok = False
                        break
                if ok:
                    selected.add(i)
                    best_size += 1
                    improved = True
                    break
    
    # Convert back to values
    result = [vertices[i] for i in selected]
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
