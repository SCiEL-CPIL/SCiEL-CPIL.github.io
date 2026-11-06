import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    N = 512
    
    # Precompute compatibility bitmasks
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                mask |= (1 << j)
        compat[i] = mask
        
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_mask(mask):
        # Check if the set represented by mask is a clique
        # For each i in mask, mask must be a subset of compat[i]
        # i.e., mask & ~compat[i] == 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            if mask & ~compat[i]:
                return False
            m ^= lsb
        return True

    def complete(mask):
        # Greedily add all compatible elements
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 1.0:
                    return mask
                bit = 1 << x
                if mask & bit:
                    continue
                # Check if x is compatible with all elements in mask
                if mask & ~compat[x]:
                    continue
                mask |= bit
                changed = True
        return mask

    def branch_and_bound(max_clique_size, current_clique, candidates_mask):
        """
        Branch and bound maximum clique search.
        candidates_mask: bitmask of vertices that can be added to current_clique
        Returns (best_size, best_clique_mask)
        """
        nonlocal deadline
        best_size = max_clique_size
        best_clique = current_clique
        
        # If no candidates, return current best
        if candidates_mask == 0:
            return best_size, best_clique
        
        # Get list of candidate vertices
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
        
        # Order candidates by degree (number of neighbors in candidates) descending
        # This gives better bounds
        def degree_in_candidates(v):
            return popcount(compat[v] & candidates_mask)
        
        candidates.sort(key=degree_in_candidates, reverse=True)
        
        # Process candidates in order
        for i, v in enumerate(candidates):
            if time.time() > deadline - 2.0:
                break
            
            # Upper bound: current clique size + remaining candidates
            if popcount(current_clique) + len(candidates) - i <= best_size:
                break
            
            # Add v to clique
            new_clique = current_clique | (1 << v)
            
            # New candidates: neighbors of v that are still in candidates
            # and come after v in the ordering (to avoid duplicates)
            new_candidates = candidates_mask & compat[v]
            # Remove v and all vertices before v in the ordering
            # We only consider vertices that come after v
            remaining = 0
            for j in range(i + 1, len(candidates)):
                u = candidates[j]
                if new_candidates & (1 << u):
                    remaining |= (1 << u)
            
            # Recurse
            sub_size, sub_clique = branch_and_bound(best_size, new_clique, remaining)
            if sub_size > best_size:
                best_size = sub_size
                best_clique = sub_clique
            
            # Remove v from candidates for next iteration
            candidates_mask &= ~(1 << v)
        
        return best_size, best_clique

    def greedy_coloring_bound(candidates_mask):
        """
        Compute a greedy coloring upper bound on clique size in candidates_mask.
        Returns the number of colors used, which is an upper bound on max clique size.
        """
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
        
        if not candidates:
            return 0
        
        # Greedy coloring: assign colors to vertices
        color_of = {}
        color_classes = []
        uncolored = set(candidates)
        
        while uncolored:
            # Create a new color class
            color_class = []
            available = set()
            for v in uncolored:
                # v can be in this color class if it's not adjacent to any vertex in color_class
                can_add = True
                for u in color_class:
                    if compat[v] & (1 << u):
                        can_add = False
                        break
                if can_add:
                    color_class.append(v)
            
            if not color_class:
                # Shouldn't happen, but break to avoid infinite loop
                break
            
            for v in color_class:
                uncolored.remove(v)
            color_classes.append(color_class)
        
        return len(color_classes)

    def branch_and_bound_with_coloring(current_clique, candidates_mask, best_size):
        """
        Branch and bound with greedy coloring for tighter bounds.
        Returns (best_size, best_clique_mask)
        """
        nonlocal deadline
        
        if candidates_mask == 0:
            return best_size, current_clique
        
        # Get candidate vertices
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
        
        # Compute greedy coloring to get upper bound
        # We'll use a simple approach: order by degree, then color
        def degree_in_candidates(v):
            return popcount(compat[v] & candidates_mask)
        
        candidates.sort(key=degree_in_candidates, reverse=True)
        
        # Greedy coloring
        color_of = {}
        color_count = 0
        for v in candidates:
            # Find the smallest color that doesn't conflict
            used_colors = set()
            for u in candidates:
                if u == v:
                    break
                if u in color_of and (compat[v] & (1 << u)):
                    used_colors.add(color_of[u])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            color_count = max(color_count, c + 1)
        
        # Upper bound: current clique size + number of colors
        if popcount(current_clique) + color_count <= best_size:
            return best_size, current_clique
        
        # Branch on vertices in order of color (process higher colors first)
        # Group by color
        by_color = {}
        for v, c in color_of.items():
            if c not in by_color:
                by_color[c] = []
            by_color[c].append(v)
        
        # Process colors in descending order
        for c in sorted(by_color.keys(), reverse=True):
            if time.time() > deadline - 2.0:
                break
            
            for v in by_color[c]:
                if time.time() > deadline - 2.0:
                    break
                
                # Upper bound check
                if popcount(current_clique) + c + 1 <= best_size:
                    break
                
                # Add v to clique
                new_clique = current_clique | (1 << v)
                
                # New candidates: neighbors of v with color < c
                new_candidates = 0
                for u in candidates:
                    if color_of[u] < c and (compat[v] & (1 << u)):
                        new_candidates |= (1 << u)
                
                sub_size, sub_clique = branch_and_bound_with_coloring(
                    new_clique, new_candidates, best_size
                )
                if sub_size > best_size:
                    best_size = sub_size
                    current_clique = sub_clique
        
        return best_size, current_clique

    # Generate initial candidates for seeding
    def get_initial_clique():
        """Get a good initial clique using greedy heuristics."""
        best_mask = 0
        best_size = 0
        
        # Strategy 1: Greedy from small to large
        mask = 0
        for i in range(1, N + 1):
            if mask & ~compat[i] == 0:
                mask |= (1 << i)
        if popcount(mask) > best_size:
            best_size = popcount(mask)
            best_mask = mask
        
        # Strategy 2: Greedy from large to small
        mask = 0
        for i in range(N, 0, -1):
            if mask & ~compat[i] == 0:
                mask |= (1 << i)
        if popcount(mask) > best_size:
            best_size = popcount(mask)
            best_mask = mask
        
        # Strategy 3: Start with 1, add by degree
        mask = 1 << 1
        remaining = set(range(2, N + 1))
        while remaining:
            # Find vertex with max degree in remaining that's compatible with mask
            best_v = None
            best_deg = -1
            for v in remaining:
                if mask & ~compat[v]:
                    continue
                deg = popcount(compat[v] & (1 << 0 | 0))  # Just count compatible in remaining
                # Actually count neighbors in remaining
                deg = 0
                for u in remaining:
                    if u != v and (compat[v] & (1 << u)):
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None:
                break
            mask |= (1 << best_v)
            remaining.remove(best_v)
        if popcount(mask) > best_size:
            best_size = popcount(mask)
            best_mask = mask
        
        # Complete the best mask
        best_mask = complete(best_mask)
        return best_mask, popcount(best_mask)

    # Run branch and bound
    initial_clique, initial_size = get_initial_clique()
    
    # All vertices as candidates
    all_candidates = 0
    for i in range(1, N + 1):
        all_candidates |= (1 << i)
    
    # Start branch and bound from empty clique
    best_size, best_clique = branch_and_bound_with_coloring(0, all_candidates, initial_size)
    
    # Final polish: try to add any remaining compatible elements
    m = best_clique
    for x in range(1, N + 1):
        if m & (1 << x):
            continue
        if m & ~compat[x] == 0:
            m |= (1 << x)
    
    # Convert best mask to list
    result = []
    mm = m
    while mm:
        lsb = mm & -mm
        result.append(lsb.bit_length() - 1)
        mm ^= lsb
        
    if not result:
        return list(range(1, 23))
        
    return sorted(result)

# EVOLVE-BLOCK-END
