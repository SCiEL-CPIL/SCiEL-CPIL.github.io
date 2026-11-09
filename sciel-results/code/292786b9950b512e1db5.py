import itertools
import math

def has_monochromatic_c6(n, colors):
    """
    Check if the coloring of K_n has a monochromatic C6.
    colors: list of colors for edges (i, j) with i < j, in lexicographic order.
    Returns True if a monochromatic C6 is found.
    """
    # Build adjacency sets for each color
    adj = [set() for _ in range(4)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            adj[c].add((j, i))
            idx += 1
            
    # Precompute all 60 cycles on 6 vertices (0..5)
    # A cycle is a tuple of 6 distinct vertices.
    # To avoid duplicates, we can fix vertex 0 at the start and consider only one direction?
    # Or just generate all permutations and check if it's a cycle in the graph.
    # Since we are checking a specific set of 6 vertices, we can map them to 0..5.
    
    # Generate all unique cycles on vertices 0..5
    # There are 60 such cycles.
    cycles = set()
    for perm in itertools.permutations(range(6)):
        # Normalize the cycle to avoid duplicates (rotation and reversal)
        # Fix 0 at the first position
        # Find index of 0 in perm
        idx0 = perm.index(0)
        rotated = perm[idx0:] + perm[:idx0]
        # Check both directions
        c1 = tuple(rotated)
        c2 = tuple(reversed(rotated))
        cycles.add(c1)
        cycles.add(c2)
        
    # The set `cycles` now contains all unique cycles on 6 vertices represented as tuples of length 6 starting with 0.
    # Actually, just adding both is fine, the set will deduplicate.
    
    for c in range(4):
        if len(adj[c]) < 6:
            continue
            
        # Iterate over all subsets of 6 vertices
        for subset in itertools.combinations(range(n), 6):
            # Map subset vertices to 0..5
            mapping = {v: i for i, v in enumerate(subset)}
            
            # Check if any of the 60 cycles exists in adj[c]
            # We only need to check cycles that start with the smallest vertex in the subset?
            # No, the `cycles` set contains cycles on abstract vertices 0..5.
            # We need to check if the induced subgraph on `subset` contains a C6.
            # This is equivalent to checking if there is a cycle in `cycles` such that
            # for all k, (mapping[subset[cycle[k]]], mapping[subset[cycle[(k+1)%6]]]) is in adj[c]?
            # Wait, `adj[c]` contains pairs of original vertex IDs.
            
            found = False
            for cyc in cycles:
                # Check if edges (cyc[i], cyc[(i+1)%6]) exist in adj[c] for the mapped vertices
                valid = True
                for i in range(6):
                    u = subset[cyc[i]]
                    v = subset[cyc[(i+1) % 6]]
                    if (u, v) not in adj[c]:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if found:
                return True
    return False

def construct():
    # Try n from 17 down to 1
    for n in range(17, 0, -1):
        half_n = n // 2
        # Iterate over all mappings of distances 1..half_n to colors 0..3
        # Number of mappings is 4^half_n
        for dist_colors in itertools.product(range(4), repeat=half_n):
            # Construct coloring
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = j - i
                    if dist > n - dist:
                        dist = n - dist
                    # dist is in 1..half_n
                    c = dist_colors[dist - 1]
                    colors.append(c)
            
            # Check validity
            if not has_monochromatic_c6(n, colors):
                return {'n': n, 'colors': colors}
                
    # Fallback: return a small valid coloring if none found (should not happen for n=9)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
