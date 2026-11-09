import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Precompute all K4s
    # Vertices 0..15
    # K4 is a set of 4 vertices
    # Edges in K4 are the 6 pairs
    
    # To check validity, we need to count distinct colors in each K4.
    # A K4 is valid if it has >= 5 distinct colors.
    # This is equivalent to: max frequency of any color in the 6 edges is 2 (and if it is 2, no other color has freq > 1? No.
    # If freqs are 2,1,1,1,1,0 -> 5 distinct. OK.
    # If freqs are 2,2,1,1,0,0 -> 4 distinct. BAD.
    # If freqs are 3,1,1,1,0,0 -> 4 distinct. BAD.
    # So condition: No color appears > 2 times. AND if a color appears 2 times, no other color appears > 1 time?
    # Actually, sum of (freq - 1) for all colors <= 1.
    # Because distinct colors = 6 - sum(freq - 1).
    # We want 6 - sum(freq - 1) >= 5  => sum(freq - 1) <= 1.
    
    # So the constraint is: In any K4, the sum of (count(c) - 1) over all colors c present is <= 1.
    
    num_vertices = 16
    num_edges = num_vertices * (num_vertices - 1) // 2
    
    # Generate K4s
    k4s = []
    for a in range(num_vertices):
        for b in range(a+1, num_vertices):
            for c in range(b+1, num_vertices):
                for d in range(c+1, num_vertices):
                    # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                    # Indices in the edge list:
                    # We need a mapping from (u,v) to index
                    pass
    
    # Mapping (u,v) to index in lex order
    def edge_index(u, v):
        if u > v: u, v = v, u
        # i is u, j is v
        # Number of edges before i: i * (n-1) - i*(i-1)/2 ?
        # Lex order: (0,1), (0,2)...(0,15), (1,2)...
        # Index = sum_{k=0}^{u-1} (n - 1 - k) + (v - u - 1)
        # sum_{k=0}^{u-1} (15 - k) = 16u - u(u-1)/2
        # Wait, n=16.
        # Count of edges with first vertex < u:
        # For k=0, 15 edges.
        # For k=u-1, 16-u edges.
        # Sum = u * (15 + 16 - u) / 2 = u * (31 - u) / 2
        # Then add (v - u - 1)
        return u * (31 - u) // 2 + (v - u - 1)
        
    k4_edges = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    idxs = [
                        edge_index(a,b), edge_index(a,c), edge_index(a,d),
                        edge_index(b,c), edge_index(b,d), edge_index(c,d)
                    ]
                    k4_edges.append(idxs)
                    
    def count_violations(colors):
        # colors is a list of 120 ints
        violations = 0
        for idxs in k4_edges:
            c0 = colors[idxs[0]]
            c1 = colors[idxs[1]]
            c2 = colors[idxs[2]]
            c3 = colors[idxs[3]]
            c4 = colors[idxs[4]]
            c5 = colors[idxs[5]]
            
            # Count frequencies
            # Since only 6 edges, we can just check pairs
            # Violation if sum(freq-1) > 1
            # This happens if:
            # 1. Any color appears >= 3 times
            # 2. Two different colors appear >= 2 times
            
            # Check 1: any triple
            # (c0,c1,c2), (c0,c1,c3)...
            # Faster:
            # Create a small dict or sorted list?
            # For speed, manual check:
            # Pairs:
            # (0,1), (0,2), (0,3), (0,4), (0,5)
            # (1,2), (1,3), (1,4), (1,5)
            # (2,3), (2,4), (2,5)
            # (3,4), (3,5)
            # (4,5)
            
            # If we find 2 disjoint pairs with same color?
            # Or 3 edges with same color?
            
            # Let's just use a set and count
            # But 1820 K4s * 120s is too slow if done naively in Python loops?
            # 1820 * 120 = 218,400 operations per evaluation.
            # Local search might do 1000 evaluations. 218M ops. Might be too slow.
            
            # Optimization:
            # We can maintain a "badness" score incrementally?
            # Or just make the check very fast.
            
            # Check for >= 3 same:
            # c0==c1==c2 or c0==c1==c3 ...
            # This is 20 comparisons for triples? No, just check if any color count >= 3.
            
            # Let's use a property:
            # If (c0==c1 and c2==c3 and c0!=c2) -> 2 pairs. Bad.
            # If (c0==c1 and c0==c2) -> 3 same. Bad.
            
            # Simple check:
            # Sort the 6 colors? No.
            
            # Let's count distinct colors.
            # If distinct < 5, violation.
            
            # c = [c0..c5]
            # distinct = len(set(c))
            # if distinct < 5: violations += 1
            
            # set() creation is slow.
            
            # Manual distinct count:
            # d = 6
            # if c0==c1: d-=1
            # if c0==c2: d-=1
            # ...
            # This is 15 comparisons.
            
            d = 6
            if c0 == c1: d -= 1
            if c0 == c2: d -= 1
            if c0 == c3: d -= 1
            if c0 == c4: d -= 1
            if c0 == c5: d -= 1
            if c1 == c2: d -= 1
            if c1 == c3: d -= 1
            if c1 == c4: d -= 1
            if c1 == c5: d -= 1
            if c2 == c3: d -= 1
            if c2 == c4: d -= 1
            if c2 == c5: d -= 1
            if c3 == c4: d -= 1
            if c3 == c5: d -= 1
            if c4 == c5: d -= 1
            
            if d < 5:
                violations += 1
                
        return violations

    # Local Search for 14 colors
    def local_search(num_colors, max_iters=5000):
        random.seed(42)
        colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
        current_bad = count_violations(colors)
        
        if current_bad == 0:
            return colors
            
        for i in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            
            # Try all colors
            best_new_c = old_c
            best_bad = current_bad
            
            # To speed up, only try colors that might help?
            # Just try all num_colors
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                colors[e_idx] = new_c
                new_bad = count_violations(colors)
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_new_c = new_c
                if new_bad == 0:
                    colors[e_idx] = new_c
                    return colors
            colors[e_idx] = best_new_c
            current_bad = best_bad
            
            # If stuck, try a random move (Simulated Annealing style)
            if current_bad > 0:
                # Occasionally accept worse moves to escape local minima
                if random.random() < 0.1:
                    e_idx2 = random.randint(0, num_edges - 1)
                    colors[e_idx2] = random.randint(0, num_colors - 1)
                    current_bad = count_violations(colors)
                    
        return None

    # Try 14 colors
    sol14 = local_search(14)
    if sol14 is not None:
        return sol14

    # Try 15 colors
    sol15 = local_search(15)
    if sol15 is not None:
        return sol15

    # Fallback: Seed (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
