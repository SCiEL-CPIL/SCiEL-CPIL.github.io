"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples
    # bad_triples[x] will store a list of (y, z) such that {x, y, z} is a bad triple
    # and y < z, y != x, z != x.
    bad_triples = [[] for _ in range(N + 1)]
    
    # To avoid duplicates, we can just iterate all triples once? 
    # Or for each x, find all y, z.
    # Total triples is ~ 128^3 / 6 = 350k. This is fast enough to precompute fully.
    # But we only need the adjacency for the local search.
    
    # Let's build a set of bad triples as frozensets for easy lookup? 
    # No, the list of pairs per x is better for the "can_add" check.
    
    # Efficient way to find bad triples:
    # For each x in 1..N:
    #   For each y in 1..N, y != x:
    #     g_xy = gcd(x, y)
    #     For each z in 1..N, z != x, z != y:
    #       if gcd(x, z) == g_xy and gcd(y, z) == g_xy:
    #         add (y, z) to bad_triples[x]
    # This is O(N^3) = 2M operations. Very fast.
    
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x == y:
                continue
            g_xy = gcd_table[x][y]
            for z in range(1, N + 1):
                if z == x or z == y:
                    continue
                if gcd_table[x][z] == g_xy and gcd_table[y][z] == g_xy:
                    # We store the pair (min(y,z), max(y,z)) to be consistent?
                    # Actually, for the check "is there a pair in S", order doesn't matter.
                    # We just need to know if BOTH are in S.
                    # To avoid duplicates in the list, we can just add it.
                    # Duplicates are handled by checking set membership.
                    # But we can optimize by only adding if y < z to keep list smaller?
                    # No, (y,z) and (z,y) are the same pair. 
                    # If we iterate all y, z, we will add (y,z) and (z,y) potentially?
                    # No, the loop is fixed x. y and z are distinct variables.
                    # If we find (y, z), we should store the sorted pair.
                    a, b = (y, z) if y < z else (z, y)
                    bad_triples[x].append((a, b))
                    
        # Remove duplicates in bad_triples[x]
        # Use a set to unique them
        unique_pairs = set(bad_triples[x])
        bad_triples[x] = list(unique_pairs)
        
        # Sort by frequency? Or just leave it.
        
    # 3. Greedy Construction
    def greedy_build(order):
        selected = []
        selected_set = set()
        for x in order:
            if x in selected_set:
                continue
            # Check if x can be added
            valid = True
            for (y, z) in bad_triples[x]:
                if y in selected_set and z in selected_set:
                    valid = False
                    break
            if valid:
                selected.append(x)
                selected_set.add(x)
        return selected, selected_set

    def local_search_fill(selected, selected_set):
        """Try to add any remaining element."""
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x in selected_set:
                    continue
                valid = True
                for (y, z) in bad_triples[x]:
                    if y in selected_set and z in selected_set:
                        valid = False
                        break
                if valid:
                    selected.append(x)
                    selected_set.add(x)
                    changed = True
        return selected, selected_set

    def local_search_swap(selected, selected_set):
        """
        Try to swap elements to potentially unlock more additions.
        """
        improved = True
        while improved:
            improved = False
            # Try all swaps
            # To speed up, we can limit the number of swap attempts per round
            # or just do one pass.
            for u in selected[:]: # Iterate over a copy
                if time.time() > deadline - 5:
                    return selected, selected_set
                for v in range(1, N + 1):
                    if v in selected_set:
                        continue
                    # Try removing u and adding v
                    # Check if v is valid in (selected - {u})
                    valid = True
                    for (y, z) in bad_triples[v]:
                        # y and z must be in (selected - {u})
                        # If y==u or z==u, they are removed, so they don't conflict.
                        # If y!=u and z!=u, they must be in selected_set.
                        if y == u or z == u:
                            continue
                        if y in selected_set and z in selected_set:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        selected.remove(u)
                        selected_set.remove(u)
                        selected.append(v)
                        selected_set.add(v)
                        
                        # Try to fill
                        selected, selected_set = local_search_fill(selected, selected_set)
                        
                        # If we filled more, we improved (size increased)
                        # We don't strictly know if size increased just from the swap itself,
                        # but the fill might add more.
                        # We track size outside?
                        # Actually, if we do the swap, we continue the outer loop.
                        improved = True
                        break
                if improved:
                    break
        return selected, selected_set

    # 4. Main Search Loop
    best_selected = []
    best_set = set()
    best_size = 0
    
    rng = random.Random(42)
    
    # Heuristic 1: Sort by number of bad triples (ascending)
    # Elements with fewer conflicts are "safer"
    badness = [len(bad_triples[i]) for i in range(N + 1)]
    order_by_badness = sorted(range(1, N + 1), key=lambda x: badness[x])
    
    # Heuristic 2: Random
    # Heuristic 3: Descending badness (risky, but might find dense clusters to avoid?)
    
    orders = [
        order_by_badness,
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -badness[x])
    ]
    
    # Add some random orders
    for _ in range(20):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline - 10:
            break
            
        # Greedy
        sel, sel_set = greedy_build(order)
        
        # Fill
        sel, sel_set = local_search_fill(sel, sel_set)
        
        # Swap
        if time.time() < deadline - 5:
            sel, sel_set = local_search_swap(sel, sel_set)
            
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_set = sel_set
            
            # If we get very high, we can stop?
            # Max is 128.
            if best_size > 100: # Just a guess, probably won't reach
                break

    # Final polish: Try to add any remaining
    best_selected, best_set = local_search_fill(best_selected, best_set)
    
    if not best_selected:
        best_selected = [1]
        
    return best_selected
# EVOLVE-BLOCK-END
