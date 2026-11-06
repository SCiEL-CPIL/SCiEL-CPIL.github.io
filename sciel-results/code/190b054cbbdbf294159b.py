import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Targets to try, in descending order of n
    # We try to find the largest valid n
    targets = [34, 30, 25, 20, 15, 12, 10, 9]
    
    # Pre-compute triple indices for a given n? 
    # Since n varies, we do it inside the solver function.
    
    def solve_for_n(n, time_limit):
        """
        Attempts to find a valid coloring for K_n^3.
        Returns (colors_list, n) if found, else (None, 0).
        """
        if time.time() > time_limit:
            return None, 0
            
        # Generate all triples
        triples = list(combinations(range(n), 3))
        M = len(triples)
        
        # Map triple to index
        idx_map = {}
        for i, t in enumerate(triples):
            idx_map[t] = i
            
        # Helper to get index of triple (a,b,c)
        # Ensure a<b<c
        def get_idx(a, b, c):
            if a > b: a, b = b, a
            if b > c: b, c = c, b
            if a > b: a, b = b, a
            return idx_map[(a, b, c)]

        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(M)]
        
        # Initial violation count
        # This is expensive: O(n^4) + O(n^5)
        # For n=30, n^5 is ~24M. Might be too slow in Python if done naively.
        # We can use the incremental logic to build it up? Or just count.
        # Let's count efficiently.
        
        # Count Red Violations (K4 all red)
        red_violations = 0
        # Iterate over all 4-sets
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check 4 edges
            e1 = get_idx(a, b, c)
            e2 = get_idx(a, b, d)
            e3 = get_idx(a, c, d)
            e4 = get_idx(b, c, d)
            if colors[e1] == 0 and colors[e2] == 0 and colors[e3] == 0 and colors[e4] == 0:
                red_violations += 1
                
        # Count Blue Violations (K5 all blue)
        blue_violations = 0
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            # Check 10 edges
            # Edges: abc, abd, abe, acd, ace, ade, bcd, bce, bde, cde
            edges = [
                get_idx(a, b, c), get_idx(a, b, d), get_idx(a, b, e),
                get_idx(a, c, d), get_idx(a, c, e), get_idx(a, d, e),
                get_idx(b, c, d), get_idx(b, c, e), get_idx(b, d, e),
                get_idx(c, d, e)
            ]
            if all(colors[ei] == 1 for ei in edges):
                blue_violations += 1
                
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors, n

        # Local Search
        # Precompute for each edge index, the list of (4th vertex) for red checks
        # and list of (5th, 6th vertices) for blue checks?
        # No, just iterate on the fly.
        
        # To speed up, we can precompute the indices of the "other" edges for each edge.
        # For edge (a,b,c):
        # Red neighbors: for each d not in {a,b,c}, the set of 3 other edges in the quad.
        # Blue neighbors: for each pair (d,e) not in {a,b,c}, the set of 9 other edges in the quint.
        
        # This precomputation is O(n * (n^2)) = O(n^3) per edge? No.
        # Total edges M ~ n^3/6.
        # For each edge, red neighbors: n-3 quads.
        # For each edge, blue neighbors: (n-3)(n-4)/2 quints.
        # Total storage: M * (n + n^2) ~ n^3 * n^2 = n^5. Too big for n=30?
        # 30^5 = 24,000,000. List of lists of ints. Might be ~100-200MB. Risky.
        
        # Alternative: Calculate on the fly.
        # We already established on-the-fly is ~3000 ops/step.
        
        steps = 0
        last_improvement_time = time.time()
        
        while time.time() < time_limit:
            # Pick random edge
            idx = random.randint(0, M - 1)
            a, b, c = triples[idx]
            old_color = colors[idx]
            new_color = 1 - old_color
            
            # Calculate Delta Red
            delta_red = 0
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # Check edges (a,b,d), (a,c,d), (b,c,d)
                i1 = get_idx(a, b, d)
                i2 = get_idx(a, c, d)
                i3 = get_idx(b, c, d)
                
                # If old_color was 0 (Red):
                # If other 3 are 0, it WAS a violation. Now e is 1, so NOT a violation.
                # delta_red -= 1
                # If old_color was 1 (Blue):
                # If other 3 are 0, it BECOMES a violation.
                # delta_red += 1
                
                if colors[i1] == 0 and colors[i2] == 0 and colors[i3] == 0:
                    if old_color == 0:
                        delta_red -= 1
                    else:
                        delta_red += 1
            
            # Calculate Delta Blue
            delta_blue = 0
            # Iterate d < e, d,e not in {a,b,c}
            # Optimization: generate combinations of the other vertices
            others = [v for v in range(n) if v != a and v != b and v != c]
            for i in range(len(others)):
                d = others[i]
                for j in range(i + 1, len(others)):
                    e = others[j]
                    # Check 9 other edges in {a,b,c,d,e}
                    # Edges not involving (a,b,c) as a triple?
                    # The 10 edges are:
                    # abc (the one we flip)
                    # abd, abe, acd, ace, ade, bcd, bce, bde, cde
                    
                    i_abd = get_idx(a, b, d)
                    i_abe = get_idx(a, b, e)
                    i_acd = get_idx(a, c, d)
                    i_ace = get_idx(a, c, e)
                    i_ade = get_idx(a, d, e)
                    i_bcd = get_idx(b, c, d)
                    i_bce = get_idx(b, c, e)
                    i_bde = get_idx(b, d, e)
                    i_cde = get_idx(c, d, e)
                    
                    if (colors[i_abd] == 1 and colors[i_abe] == 1 and 
                        colors[i_acd] == 1 and colors[i_ace] == 1 and 
                        colors[i_ade] == 1 and colors[i_bcd] == 1 and 
                        colors[i_bce] == 1 and colors[i_bde] == 1 and 
                        colors[i_cde] == 1):
                        
                        if old_color == 1: # Was Blue, becomes Red -> destroys violation
                            delta_blue -= 1
                        else: # Was Red, becomes Blue -> creates violation
                            delta_blue += 1
            
            total_delta = delta_red + delta_blue
            
            # Accept if improvement or random (simulated annealing style, but simple greedy first)
            # Given time constraints, simple greedy (accept if < 0) is safer/faster to converge if possible.
            # But might get stuck. Let's add a small probability of accepting worse moves if stuck?
            # For now, accept if total_delta < 0.
            if total_delta < 0:
                colors[idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
                total_violations += total_delta
                last_improvement_time = time.time()
                
                if total_violations == 0:
                    return colors, n
            # If stuck for a while, we could restart or randomize, but let's keep it simple.
            
        return None, 0

    # Try targets
    # Allocate time based on difficulty? 
    # Larger n is harder. 
    # Let's give 20s to 34, 20s to 30, 20s to 25, 20s to 20, 20s to 15, 10s to rest.
    
    remaining_targets = [
        (34, 20),
        (30, 20),
        (25, 20),
        (20, 20),
        (15, 10),
        (12, 5),
        (10, 5),
        (9, 5)
    ]
    
    for n_target, t_alloc in remaining_targets:
        if time.time() > deadline:
            break
            
        # If we already found a valid n >= n_target, we can skip?
        # No, we want the MAX n. If best_n is 30, and we try 34, and fail, we keep 30.
        # If we try 25, and best_n is 30, we don't need to try 25.
        if n_target <= best_n:
            continue
            
        # Adjust time limit to not exceed deadline
        current_time = time.time()
        allowed_time = min(t_alloc, deadline - current_time)
        
        if allowed_time < 1.0:
            break
            
        colors, n_found = solve_for_n(n_target, current_time + allowed_time)
        
        if colors is not None and n_found > best_n:
            best_n = n_found
            best_colors = colors
            
    # Fallback: If no better than seed found, return seed
    if best_n < 9 or best_colors is None:
        # Use the seed construction for n=9
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}
