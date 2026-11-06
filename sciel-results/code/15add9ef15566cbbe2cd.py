"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair.
    Maximize cardinality.
    
    Strategy:
    1. Build compatibility graph using bitsets.
    2. Initialize best clique with divisors of a highly composite number (e.g., 360).
    3. Run Tomita's algorithm (Bron-Kerbosch with greedy coloring) to find the maximum clique.
    4. Return the maximum clique found.
    """
    import time
    from math import gcd
    
    start_time = time.time()
    # The evaluator kills at 90s. We use 85s as a safety margin to ensure we return a valid result.
    deadline = start_time + 85.0
    
    N = 512
    
    # 1. Build compatibility graph using bitsets
    # adj_mask[v] is a bitmask where bit u is set if lcm(v, u) <= N and u != v
    adj_mask = [0] * (N + 1)
    
    # Optimization: Precompute LCMs or just check on the fly. 
    # N=512 is small enough that O(N^2) gcd checks are fast (~262k ops).
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
    
    # 2. Initialize best clique with divisors of 360
    # 360 is highly composite. Its divisors form a valid clique because if a|360 and b|360, 
    # then lcm(a,b) divides 360, so lcm(a,b) <= 360 <= 512.
    # Let's check 360, 420, 480, 504.
    # 360: 24 divisors.
    # 420: 24 divisors.
    # 480: 20 divisors.
    # 504: 24 divisors.
    # Actually, let's just pick the set of divisors of 360 as a safe strong seed.
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    seed_values = get_divisors(360)
    # Ensure all are <= 512 (they are, max is 360)
    best_clique = seed_values
    best_size = len(best_clique)
    
    # 3. Tomita's Algorithm for Maximum Clique
    # We represent the set of available vertices P and excluded vertices X as bitmasks.
    # Vertices are 1..N. Bit i in mask corresponds to vertex i.
    
    # To make coloring and ordering efficient, we might want to work with a list of active vertices.
    # However, bitmasks are very fast for intersection.
    # Tomita's algorithm typically orders vertices and colors them.
    
    # Let's implement a recursive expand function.
    # P: bitmask of candidates
    # X: bitmask of excluded candidates (for completeness, though for max size we mainly care about P)
    # path: list of vertices in current clique
    
    # Pre-compute degrees for initial ordering?
    # Tomita usually orders by degree in the current subgraph P.
    
    def expand(P, X, path):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(path) > best_size:
                best_size = len(path)
                best_clique = list(path)
            return
        
        # Greedy Coloring on P
        # We need to assign colors to vertices in P such that adjacent vertices have different colors.
        # The number of colors used is an upper bound on the clique size in P.
        # We process vertices in a specific order (usually descending degree in P) to get a tighter bound.
        
        # Extract vertices from P
        # To speed up, we can maintain a list of vertices in P sorted by degree, but degrees change as P changes.
        # For N=512, extracting and sorting 512 ints is fast.
        
        verts = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts.append(v)
            temp ^= lsb
            
        if not verts:
            return
            
        # Calculate degrees in P for sorting
        # deg(v) = popcount(adj_mask[v] & P)
        degs = []
        for v in verts:
            deg = (adj_mask[v] & P).bit_count()
            degs.append((deg, v))
            
        # Sort by degree descending
        degs.sort(key=lambda x: -x[0])
        verts = [v for _, v in degs]
        
        # Greedy Coloring
        # color[v] = color index
        # We want to minimize the number of colors.
        # Standard greedy: for v in verts, assign smallest color not used by neighbors in verts already colored.
        
        color = {}
        # To check if a color is used by a neighbor, we can keep a list of vertices per color.
        # colors[c] = list of vertices with color c
        # If any vertex in colors[c] is a neighbor of v, v cannot take color c.
        
        colors = [] # list of lists
        max_color = 0
        
        for v in verts:
            assigned = False
            for c in range(len(colors)):
                # Check if v is connected to any vertex in colors[c]
                conflict = False
                for u in colors[c]:
                    if (adj_mask[v] >> u) & 1:
                        conflict = True
                        break
                if not conflict:
                    colors[c].append(v)
                    color[v] = c
                    assigned = True
                    break
            if not assigned:
                colors.append([v])
                color[v] = len(colors) - 1
                max_color = len(colors) - 1
                
        # Upper bound check
        # The max clique size we can add is at most max_color + 1
        if len(path) + max_color + 1 <= best_size:
            return
        
        # Branching
        # We should branch on vertices in reverse color order (highest color first)
        # This is because vertices with higher colors are "later" in the greedy ordering,
        # and branching on them first often finds larger cliques sooner, improving pruning.
        
        # Group vertices by color
        # We already have 'colors' list.
        # Process colors from max_color down to 0
        
        # We need to iterate through vertices in P in reverse color order.
        # Let's create a list of (color, vertex) and sort by color desc.
        # Or just iterate c from max_color down to 0, and for each v in colors[c].
        
        # However, as we branch and remove vertices from P, the coloring becomes invalid for the new P.
        # Tomita's algorithm handles this by re-coloring in the recursive call.
        # The pruning condition `len(path) + c + 1 <= best_size` is applied dynamically.
        
        # To be precise with Tomita's pruning:
        # When we are at color c, the remaining vertices (in this color and lower) can form a clique of size at most c+1?
        # Actually, the bound is that the max clique in the subgraph induced by vertices with color <= c is <= c+1.
        # So if len(path) + c + 1 <= best_size, we can stop processing colors <= c.
        
        for c in range(max_color, -1, -1):
            if len(path) + c + 1 <= best_size:
                break
                
            # Iterate through vertices with color c
            # We must ensure we only process vertices still in P
            # Note: 'colors' was computed for the initial P of this call.
            # As we remove vertices from P, some vertices in colors[c] might no longer be in P.
            
            # To maintain correctness and efficiency, we iterate through the vertices assigned color c.
            # If a vertex is no longer in P, we skip it.
            
            # We need to be careful: if we remove a vertex from P, it doesn't affect the colors of other vertices in the current loop iteration's logic for pruning, 
            # but it does affect the recursive call's P.
            
            # Standard Tomita implementation detail:
            # The loop is over vertices in P ordered by color.
            # Since P changes, we can't just iterate over the static 'colors' list blindly if we remove vertices that are in higher colors first?
            # No, we process high colors first.
            
            # Let's stick to the standard structure:
            # For each vertex v in P (ordered by color desc):
            #   Recurse on P & adj[v]
            #   Remove v from P
            #   Add v to X
            #   Prune if len(path) + color_bound_of_remaining_P <= best_size
            
            # To get the "color bound of remaining P", we would need to re-color.
            # However, a simpler and effective variant is:
            # The max color assigned to any vertex in the *current* P is an upper bound.
            # As we remove vertices, the max color of the remaining P might decrease.
            # But calculating the exact max color of the remaining P requires re-coloring.
            
            # Alternative simpler pruning used in many implementations:
            # If we process vertices in reverse color order, and we have processed all vertices with color > c,
            # then the remaining vertices have color <= c.
            # Thus, the max clique size in the remaining P is <= c + 1.
            # This holds if the coloring was valid for the initial P and we only remove vertices.
            # Removing vertices does not invalidate the coloring of the remaining vertices (they still have distinct colors from their neighbors).
            # So, yes, if we process colors in descending order, the bound `c+1` is valid for the remaining subgraph.
            
            for v in colors[c]:
                if time.time() > deadline:
                    return
                
                # Check if v is still in P
                if (P >> v) & 1 == 0:
                    continue
                    
                # Recurse
                new_P = P & adj_mask[v]
                new_X = X | (P & ~adj_mask[v]) # X is not strictly needed for size, but good for completeness if we wanted to output all max cliques
                # Actually, for max clique size, X is not needed in the recursion if we only care about size and one clique.
                # Bron-Kerbosch usually passes X to ensure we find *all* maximal cliques. 
                # For maximum clique, we can ignore X in the pruning logic, but passing it is standard.
                # However, the coloring bound relies only on P.
                
                expand(new_P, new_X, path + [v])
                
                # Remove v from P
                P &= ~(1 << v)
                
                # Update X? 
                # In Tomita, X is often used for reconstruction or finding all cliques. 
                # For just finding the max size, we can skip updating X or just pass it along.
                # The pruning `len(path) + c + 1 <= best_size` depends on the structure of P.
                
    # Initial call
    # P should contain all vertices 1..N
    # Bitmask for 1..N:
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
        
    # X is initially 0
    initial_X = 0
    
    expand(initial_P, initial_X, [])
    
    # Verify result
    # The result should be valid. 
    # Just in case, do a quick check.
    res = best_clique
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if res[i] * res[j] // gcd(res[i], res[j]) > N:
                # Fallback to seed if something went wrong
                return seed_values
    return res

# EVOLVE-BLOCK-END
