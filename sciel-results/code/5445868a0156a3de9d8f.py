"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations
from math import lcm

def construct():
    N = 128
    # Precompute LCMs for all pairs to speed up triple checking
    # lcm[i][j] for 1 <= i < j <= N
    # Actually, just compute on the fly or precompute a list of bad triples.
    
    # 1. Identify all "bad" triples (a, b, c) with a < b < c
    # such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # For efficiency, we can group by the common LCM value?
    # Or just iterate. 128 choose 3 is ~350k. LCM is fast.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) and lab == lcm(b, c):
                    bad_triples.append((a, b, c))
    
    # 2. Build adjacency structures for fast lookup
    # For each vertex v, list of pairs (u, w) such that (v, u, w) is a bad triple (unordered u, w)
    # And for each pair (u, v), list of w such that (u, v, w) is a bad triple.
    
    # v_to_pairs[v] = set of frozensets {u, w}
    v_to_pairs = {i: set() for i in range(1, N + 1)}
    # pair_to_vertices[(u, v)] = set of w
    pair_to_vertices = {}
    
    for a, b, c in bad_triples:
        # Update v_to_pairs
        v_to_pairs[a].add(frozenset((b, c)))
        v_to_pairs[b].add(frozenset((a, c)))
        v_to_pairs[c].add(frozenset((a, b)))
        
        # Update pair_to_vertices
        pair_to_vertices.setdefault(frozenset((a, b)), set()).add(c)
        pair_to_vertices.setdefault(frozenset((a, c)), set()).add(b)
        pair_to_vertices.setdefault(frozenset((b, c)), set()).add(a)
    
    # 3. Initial Greedy Solution for Lower Bound
    # Greedy: pick vertices in order of increasing degree in the hypergraph?
    # Or just 1 to N.
    greedy_selected = []
    for x in range(1, N + 1):
        # Check if x can be added
        valid = True
        for a, b in combinations(greedy_selected, 2):
            if (a, b, x) in bad_triples or (a, x, b) in bad_triples or (x, a, b) in bad_triples:
                 # Need efficient check
                 pass
        # This greedy check is slow if done naively.
        # Let's just use the seed's greedy for the initial bound.
        selected = []
        for x in range(1, N + 1):
            if all(not (lcm(a,b)==lcm(a,x)==lcm(b,x)) for a,b in combinations(selected,2)):
                selected.append(x)
        best_size = len(selected)
        best_set = selected
    
    # 4. Branch and Bound
    # Order vertices by "constraint strength" (number of bad triples they are in)
    degrees = {i: len(v_to_pairs[i]) for i in range(1, N + 1)}
    ordered_vertices = sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True)
    
    # Map original index to position in ordered list for bitmask?
    # Or just iterate through ordered_vertices.
    
    # State:
    # idx: index in ordered_vertices
    # selected_mask: bitmask of selected vertices (1..128)
    # banned_mask: bitmask of vertices that cannot be selected
    # forbidden_pairs: set of frozensets {u, v} that cannot both be selected
    
    # To speed up, we use bitmasks for selected and banned.
    # forbidden_pairs is a set of (min(u,v), max(u,v))
    
    def bit(n):
        return 1 << (n - 1)
    
    def has_bit(mask, n):
        return (mask & bit(n)) != 0
    
    def set_bit(mask, n):
        return mask | bit(n)
    
    def clear_bit(mask, n):
        return mask & ~bit(n)
    
    # Precompute bitmasks for v_to_pairs?
    # For each v, we need to know which vertices become banned if v is selected.
    # If v is selected, for every pair {u, w} in v_to_pairs[v]:
    #    If u is selected, w is banned.
    #    If w is selected, u is banned.
    #    If neither is selected, {u, w} is added to forbidden_pairs.
    
    # This dynamic update of forbidden_pairs is the tricky part.
    # We can pass forbidden_pairs as a set in the recursion.
    
    sys.setrecursionlimit(10000)
    
    best_size = 0
    best_set = []
    
    def update_state(selected_mask, banned_mask, forbidden_pairs, v):
        new_banned = banned_mask
        new_forbidden = set(forbidden_pairs)
        
        # For each pair {u, w} involving v in bad triples
        for pair in v_to_pairs[v]:
            u, w = tuple(pair)
            u_sel = has_bit(selected_mask, u)
            w_sel = has_bit(selected_mask, w)
            
            if u_sel and w_sel:
                # This shouldn't happen if state is valid, but if it does, it's a conflict
                # Actually, if u and w were selected, v would have been banned.
                pass
            elif u_sel:
                # w becomes banned
                if not has_bit(new_banned, w):
                    new_banned = set_bit(new_banned, w)
            elif w_sel:
                # u becomes banned
                if not has_bit(new_banned, u):
                    new_banned = set_bit(new_banned, u)
            else:
                # Neither selected, so {u, w} becomes a forbidden pair
                # Only add if u, w are not already banned?
                # If u or w is banned, the pair is irrelevant.
                if not has_bit(banned_mask, u) and not has_bit(banned_mask, w):
                    new_forbidden.add(frozenset((u, w)))
        
        return new_banned, new_forbidden

    def branch_and_bound(idx, selected_mask, banned_mask, forbidden_pairs, current_size):
        nonlocal best_size, best_set
        
        # Upper Bound
        # Max possible additional vertices = N - current_size (very weak)
        # Better: Count available vertices
        # available = [v for v in ordered_vertices[idx:] if not has_bit(banned_mask, v)]
        # Even better: Check forbidden_pairs to see how many we can realistically add.
        # For now, use: current_size + (len(ordered_vertices) - idx)
        if current_size + (len(ordered_vertices) - idx) <= best_size:
            return
        
        if idx == len(ordered_vertices):
            if current_size > best_size:
                best_size = current_size
                # Reconstruct set
                s = []
                for i in range(1, N + 1):
                    if has_bit(selected_mask, i):
                        s.append(i)
                best_set = s
            return
        
        v = ordered_vertices[idx]
        
        # Option 1: Exclude v
        branch_and_bound(idx + 1, selected_mask, banned_mask, forbidden_pairs, current_size)
        
        # Option 2: Include v
        if not has_bit(banned_mask, v):
            # Check if v conflicts with any existing forbidden pair?
            # Forbidden pairs are between unselected vertices.
            # If v is selected, it is no longer "unselected".
            # Does selecting v violate any constraint?
            # The constraint is: no bad triple in the final set.
            # If we select v, we must ensure no bad triple {v, u, w} is formed where u, w are in final set.
            # Our state update handles this by banning or forbidding u, w.
            # So we just need to check if v is banned.
            
            new_banned, new_forbidden = update_state(selected_mask, banned_mask, forbidden_pairs, v)
            
            # Check if any newly banned vertex was already selected?
            # If has_bit(selected_mask, w) and w in new_banned, then conflict.
            # But update_state only bans u or w if the other is selected.
            # If u is selected, w is banned. If w was already selected, it would have been caught earlier?
            # No, if w was selected, and u is selected, then {u, w, v} is a bad triple.
            # If u, w were selected, v should have been banned in a previous step.
            # So if v is not banned, it's safe to select?
            
            # However, we must ensure that selecting v doesn't create a conflict with
            # existing selected vertices via a third vertex that is ALSO selected.
            # This is covered by the "banned" logic.
            
            # One more check: Did we ban a vertex that is currently selected?
            # If yes, this branch is invalid.
            valid = True
            # Check if any vertex in new_banned is in selected_mask
            # We can do this by checking intersection
            if new_banned & selected_mask:
                valid = False
            
            if valid:
                branch_and_bound(idx + 1, set_bit(selected_mask, v), new_banned, new_forbidden, current_size + 1)
    
    # Start B&B
    # Initial state: no selected, no banned, no forbidden pairs
    branch_and_bound(0, 0, 0, set(), 0)
    
    if not best_set:
        # Fallback to greedy if B&B failed or found nothing
        best_set = []
        for x in range(1, N + 1):
            if all(not (lcm(a,b)==lcm(a,x)==lcm(b,x)) for a,b in combinations(best_set,2)):
                best_set.append(x)
                
    return best_set

# EVOLVE-BLOCK-END
