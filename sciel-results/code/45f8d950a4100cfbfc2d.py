import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 75  # Safety margin (90s kill, leave buffer)
    
    # 1. Precompute compatibility graph
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= limit
    compat = [0] * (limit + 1)
    for i in range(1, limit + 1):
        mask = 0
        for j in range(1, limit + 1):
            g = gcd(i, j)
            if (i // g) * j <= limit:
                mask |= (1 << j)
        compat[i] = mask

    # 2. Initial Solution: Max divisors of any number <= limit
    best_size = 0
    best_clique = []
    
    # Check all n, find max divisors
    max_divs = 0
    best_n = 1
    for n in range(1, limit + 1):
        d = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                d += 1
                if i != n // i:
                    d += 1
        if d > max_divs:
            max_divs = d
            best_n = n
            
    # Get divisors of best_n
    divs = [i for i in range(1, limit + 1) if best_n % i == 0]
    best_size = len(divs)
    best_clique = divs
    
    # 3. Tomita-style coloring bound for upper bound estimation
    def color_bound(candidates):
        """Compute greedy coloring upper bound for max clique in subgraph induced by candidates."""
        if candidates == 0:
            return 0
        # Greedy sequential coloring: assign colors to vertices in candidates
        # such that adjacent vertices get different colors.
        # The number of colors used is an upper bound on clique size.
        uncolored = candidates
        color_count = 0
        while uncolored:
            color_count += 1
            # Pick a maximal independent set from uncolored (greedy)
            # A vertex can take this color if it's not adjacent to any already-colored-with-this-color vertex
            colorable = uncolored
            colored_this_round = 0
            while colorable:
                # Pick first available vertex
                lsb = colorable & -colorable
                idx = lsb.bit_length() - 1
                # Remove this vertex and its neighbors from colorable
                colorable &= ~compat[idx]
                colored_this_round |= lsb
            uncolored &= ~colored_this_round
        return color_count

    # 4. Branch and Bound for Max Clique with Tomita-style bounds
    def expand(candidates, current_clique):
        nonlocal best_size, best_clique
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning 1: Simple cardinality bound
        if len(current_clique) + candidates.bit_count() <= best_size:
            return
        
        # Pruning 2: Coloring bound (Tomita-style)
        cb = color_bound(candidates)
        if len(current_clique) + cb <= best_size:
            return
        
        # If no candidates, update best
        if candidates == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique.copy()
            return

        # Branching: Pick vertex with highest degree in candidate subgraph
        max_deg = -1
        v = -1
        c = candidates
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            deg = (compat[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            c ^= lsb
            
        if v == -1:
            return

        bit_v = 1 << v
        candidates_without_v = candidates & ~bit_v
        
        # Include v first (more likely to find good solutions early)
        new_candidates = candidates_without_v & compat[v]
        expand(new_candidates, current_clique + [v])
        
        # Exclude v
        expand(candidates_without_v, current_clique)

    # Start B&B
    all_candidates = (1 << (limit + 1)) - 2  # bits 1..512
    
    expand(all_candidates, [])
    
    return sorted(best_clique)
