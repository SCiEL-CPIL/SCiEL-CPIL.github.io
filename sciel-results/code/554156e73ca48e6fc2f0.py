"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import gcd

def construct():
    """
    Returns a subset of [1, 512] with lcm(a, b) <= 512 for all pairs,
    maximizing cardinality.
    
    Strategy:
    1. Build compatibility graph (edge if lcm(a,b) <= 512).
    2. Use branch-and-bound with greedy coloring for upper bounds.
    3. Initialize with divisors of highly composite numbers.
    4. Polish with local search.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute adjacency as bitmasks
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and i * j // gcd(i, j) <= N:
                mask |= (1 << j)
        adj[i] = mask
    
    # Greedy coloring for upper bound
    def greedy_color_ub(candidates_mask):
        """Upper bound on clique size via greedy coloring."""
        if candidates_mask == 0:
            return 0
        
        # Extract vertices sorted by degree descending
        verts = []
        m = candidates_mask
        while m:
            low = m & (-m)
            v = low.bit_length()
            verts.append(v)
            m ^= low
        
        # Sort by degree in candidates
        verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count(), reverse=True)
        
        # Greedy coloring
        color_of = {}
        max_color = -1
        for v in verts:
            used_colors = set()
            av = adj[v] & candidates_mask
            # Check neighbors
            m2 = av
            while m2:
                low = m2 & (-m2)
                u = low.bit_length()
                if u in color_of:
                    used_colors.add(color_of[u])
                m2 ^= low
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
        
        return max_color + 1
    
    # Branch and bound
    best_clique = set()
    best_size = 0
    
    def bb(candidates_mask, current_clique):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = set(current_clique)
            return
        
        ub = greedy_color_ub(candidates_mask)
        if len(current_clique) + ub <= best_size:
            return
        
        # Get vertices in candidates, sorted by degree desc
        verts = []
        m = candidates_mask
        while m:
            low = m & (-m)
            v = low.bit_length()
            verts.append(v)
            m ^= low
        verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count(), reverse=True)
        
        for v in verts:
            if time.time() > deadline:
                return
            
            new_clique = current_clique | {v}
            new_cand = candidates_mask & adj[v] & ~(1 << v)
            
            bb(new_cand, new_clique)
            
            candidates_mask &= ~(1 << v)
            
            if candidates_mask == 0:
                break
            ub = greedy_color_ub(candidates_mask)
            if len(current_clique) + ub <= best_size:
                break
    
    # Initialize with divisors of highly composite numbers
    for L in [360, 420, 480, 504, 336, 240, 432, 448, 384, 400]:
        if L > N:
            continue
        divs = set()
        for i in range(1, int(L**0.5) + 1):
            if L % i == 0:
                divs.add(i)
                if i != L // i:
                    divs.add(L // i)
        # Verify valid clique
        valid = True
        dl = list(divs)
        for i in range(len(dl)):
            for j in range(i+1, len(dl)):
                if dl[i] * dl[j] // gcd(dl[i], dl[j]) > N:
                    valid = False
                    break
            if not valid:
                break
        if valid and len(divs) > best_size:
            best_size = len(divs)
            best_clique = set(divs)
    
    # Run B&B
    full_mask = 0
    for i in range(1, N + 1):
        full_mask |= (1 << i)
    
    bb(full_mask, set())
    
    # Local search: add elements
    if time.time() < deadline:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            mask_clique = 0
            for y in best_clique:
                mask_clique |= (1 << y)
            for x in range(1, N + 1):
                if x in best_clique:
                    continue
                if (adj[x] & mask_clique) == mask_clique:
                    best_clique.add(x)
                    best_size += 1
                    changed = True
                    break
    
    return list(best_clique)
# EVOLVE-BLOCK-END
