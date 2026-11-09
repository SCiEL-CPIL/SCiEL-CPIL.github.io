"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 128
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Identify all bad triples and build partner structures
    # partners[i][z] = set of y such that {i, y, z} is a bad triple
    # Note: i, y, z are distinct.
    # We can store partners as a list of dicts: partners[i] is {z: set_of_y}
    partners = [dict() for _ in range(N + 1)]
    
    # bad_triples_set for quick lookup if needed, but partners is sufficient for local search
    # We also need to know the "degree" or "badness" for greedy init
    
    badness = [0] * (N + 1)
    
    # Iterate through all triples (i, j, k) with i < j < k
    # If gcd(i,j) == gcd(j,k) == gcd(i,k), it's a bad triple.
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            for k in range(j + 1, N + 1):
                if gcd_table[i][k] == g_ij and gcd_table[j][k] == g_ij:
                    # Bad triple (i, j, k)
                    # Update partners
                    # For i: if j and k are in S, i cannot be added.
                    # So partners[i][j].add(k), partners[i][k].add(j)
                    # For j: partners[j][i].add(k), partners[j][k].add(i)
                    # For k: partners[k][i].add(j), partners[k][j].add(i)
                    
                    # i, j, k
                    if j not in partners[i]: partners[i][j] = set()
                    partners[i][j].add(k)
                    if k not in partners[i]: partners[i][k] = set()
                    partners[i][k].add(j)
                    
                    if i not in partners[j]: partners[j][i] = set()
                    partners[j][i].add(k)
                    if k not in partners[j]: partners[j][k] = set()
                    partners[j][k].add(i)
                    
                    if i not in partners[k]: partners[k][i] = set()
                    partners[k][i].add(j)
                    if j not in partners[k]: partners[k][j] = set()
                    partners[k][j].add(i)
                    
                    # Increment badness
                    badness[i] += 1
                    badness[j] += 1
                    badness[k] += 1

    # 3. Helper functions for Local Search
    
    def is_valid_add(v, S):
        """Check if v can be added to S."""
        if v in S:
            return False
        # v can be added if there is no pair (a,b) in S such that {v,a,b} is bad.
        # This is equivalent to: For all z in S, v is not in partners[z] ?
        # No, partners[z] maps w -> set of x.
        # If {v, a, b} is bad, then a in partners[v] and b in partners[v][a].
        # So we need to check if there exists a in S such that b in S and b in partners[v][a].
        # Iterating over S is O(|S|).
        # Since |S| <= 128, this is fast.
        for a in S:
            if a in partners[v]:
                # Check if any b in partners[v][a] is in S
                # partners[v][a] is a set. S is a set.
                # Intersection check:
                if partners[v][a] & S:
                    return False
        return True

    def get_forbidden_set(v, S):
        """Get the set of nodes that cannot be added alongside v given S."""
        # If we add v, we cannot add any y such that there exists z in S with {v,y,z} bad.
        # Forbidden(y) = True if y in union_{z in S} partners[v][z]
        forbidden = set()
        for z in S:
            if z in partners[v]:
                forbidden.update(partners[v][z])
        return forbidden

    def is_valid_add_pair(v, w, S):
        """Check if v and w can be added to S simultaneously."""
        if v in S or w in S or v == w:
            return False
        # Check if {v, w, z} is bad for any z in S.
        # This is equivalent to: w is in forbidden_set(v, S) OR v is in forbidden_set(w, S).
        # Due to symmetry, one check is enough if we are careful, but let's do both to be safe or just one.
        # If {v,w,z} is bad, then w is in partners[v][z] for some z in S.
        # So w in get_forbidden_set(v, S).
        forbidden_v = get_forbidden_set(v, S)
        if w in forbidden_v:
            return False
        # Also need to ensure v and w don't conflict with each other? 
        # The condition is "no three distinct elements". 
        # If we add v and w, we form triples with existing elements.
        # Do v and w form a bad triple with each other? No, a triple needs 3 distinct elements.
        # So we only care about triples involving at least one existing element.
        # Wait, what if S is empty? Then adding v,w is always valid.
        # What if S has 1 element z? Then {v,w,z} must not be bad.
        # The check `w in get_forbidden_set(v, S)` covers all z in S.
        return True

    def local_search(S):
        """Perform local search to improve S."""
        S = set(S)
        improved = True
        iterations = 0
        max_iterations = 5000  # Hard cap on iterations to avoid infinite loops
        
        while improved and time.time() < deadline - 2:
            improved = False
            iterations += 1
            if iterations > max_iterations:
                break
                
            # Phase 1: Try to add any single element
            for v in range(1, N + 1):
                if v not in S and is_valid_add(v, S):
                    S.add(v)
                    improved = True
                    # Restart loop to catch more additions
                    break 
            
            if improved:
                continue
                
            # Phase 2: Try 1-Swap (remove u, add v)
            # This is useful to change the composition of S to allow more additions later.
            # We prioritize swaps that don't decrease size, but might unlock additions.
            # Actually, 1-swap keeps size same. It's a "move".
            # We should only do this if we are stuck.
            # To avoid cycling, we can try random swaps or specific ones.
            # Let's try to swap u with v if v has "lower badness" or similar heuristic?
            # Or just try all? O(N^2) per iteration.
            
            # Phase 3: Try 1-Remove-2-Add (remove u, add v, w) -> Net +1
            # This is the most powerful move for growth.
            # For each u in S:
            #   Remove u
            #   Find pair (v, w) not in S such that is_valid_add_pair(v, w, S - {u})
            #   If found, add v, w.
            
            found_growth = False
            for u in list(S):
                if time.time() > deadline - 2:
                    return S
                S_temp = S - {u}
                
                # To find a valid pair (v, w), we can iterate v, then w.
                # Optimization: Precompute forbidden sets for S_temp?
                # S_temp is large.
                # Let's iterate v from 1..N not in S_temp.
                # For each v, compute forbidden_v = get_forbidden_set(v, S_temp).
                # Then find w not in S_temp, w != v, w not in forbidden_v.
                # If such w exists, we have a valid pair.
                
                candidates_v = [v for v in range(1, N + 1) if v not in S_temp]
                # Sort candidates by badness? Lower badness is safer.
                candidates_v.sort(key=lambda x: badness[x])
                
                for v in candidates_v:
                    forbidden_v = get_forbidden_set(v, S_temp)
                    # Find any w in candidates_v (excluding v) not in forbidden_v
                    # Also w must not be in S_temp (already ensured by candidates_v)
                    # And w != v
                    for w in candidates_v:
                        if w == v:
                            continue
                        if w not in forbidden_v:
                            # Check if v and w are compatible?
                            # is_valid_add_pair(v, w, S_temp) is True by construction.
                            # Perform the move
                            S.remove(u)
                            S.add(v)
                            S.add(w)
                            found_growth = True
                            improved = True
                            break
                    if found_growth:
                        break
                if found_growth:
                    break
            
            if found_growth:
                continue
                
            # Phase 4: Try 2-Remove-3-Add? Too expensive.
            # Phase 5: Randomized Restart / Perturbation
            # If stuck, randomly remove 1-2 elements and try to fill.
            if time.time() < deadline - 5:
                # Randomly remove 1 element
                if S:
                    u = random.choice(list(S))
                    S.remove(u)
                    # Try to fill
                    for _ in range(2): # Try a few times to re-add
                        # Re-run Phase 1 logic briefly
                        for v in range(1, N + 1):
                            if v not in S and is_valid_add(v, S):
                                S.add(v)
                                break
                    # If size increased or stayed same, it's a move.
                    # If size decreased, it's a perturbation.
                    # We accept it to escape local optimum.
                    improved = True
                    break # Restart loop

        return S

    # 4. Main Search Loop
    
    best_S = set()
    best_size = 0
    
    rng = random.Random(42)
    
    # Heuristic 1: Sort by badness (ascending) - "Safest" first
    order_by_badness = sorted(range(1, N + 1), key=lambda x: badness[x])
    
    # Heuristic 2: Reverse badness - "Riskiest" first?
    order_by_badness_rev = sorted(range(1, N + 1), key=lambda x: -badness[x])
    
    # Heuristic 3: Natural order
    order_nat = list(range(1, N + 1))
    
    # Heuristic 4: Random
    # We will generate random orders on the fly.
    
    def greedy_build(order):
        S = set()
        for x in order:
            if is_valid_add(x, S):
                S.add(x)
        return S

    # Try deterministic orders first
    orders = [
        order_by_badness,
        order_nat,
        order_by_badness_rev
    ]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        S = greedy_build(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Try random orders
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = greedy_build(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Early stop if we found a very large set?
        # Max possible is 128.
        # If we get > 110, it's very good.
        if best_size > 110:
            # Do a few more iterations to polish
            pass

    # Final polish on the best set found
    if best_S:
        best_S = local_search(best_S)
        
    if not best_S:
        best_S = {1}
        
    return list(best_S)
# EVOLVE-BLOCK-END
