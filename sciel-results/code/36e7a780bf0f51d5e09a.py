import itertools
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _is_dissociated(denominators):
    """
    Check if the set of reciprocals of 'denominators' has distinct subset sums.
    Returns True if dissociated.
    """
    k = len(denominators)
    if k == 0:
        return True
    L = _lcm_list(denominators)
    ints = [L // d for d in denominators]
    
    sums = set()
    # Optimization: We only need to check if any sum repeats.
    # Since all ints are positive, sums are unique if the set is dissociated.
    # We can just generate all subset sums and check for duplicates.
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += ints[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _find_minimal_bad_sets(universe):
    """
    Find all minimal subsets of 'universe' that are NOT dissociated.
    We focus on sizes 3 and 4, as larger minimal bad sets are rare for small integers 
    and are usually supersets of smaller ones or require more complex dependencies 
    that are likely captured by the graph structure if we include size 3 and 4 conflicts.
    """
    bad_sets = []
    
    # Check size 3
    for comb in itertools.combinations(universe, 3):
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    bad_size_3 = [s for s in bad_sets if len(s) == 3]
    
    # Check size 4
    # Optimization: Only check if the 4-set doesn't contain a known bad 3-set
    for comb in itertools.combinations(universe, 4):
        comb_set = set(comb)
        is_superset_of_bad = False
        for b in bad_size_3:
            if b.issubset(comb_set):
                is_superset_of_bad = True
                break
        if is_superset_of_bad:
            continue
        if not _is_dissociated(list(comb)):
            bad_sets.append(set(comb))
            
    # We generally don't need to check size 5 for this specific universe range [1,22] 
    # to find the MAX dissociated set, because the density of conflicts is high.
    # However, to be safe, we could check size 5, but it increases pre-computation time.
    # Given 60s limit, checking 3 and 4 is very fast.
    # If a set of 5 is bad but has no bad subset of 3 or 4, it would be missed.
    # But typically, if a set of 5 is bad, it contains a smaller bad set.
    # Let's stick to 3 and 4 for speed, as the MIS will likely avoid large dependencies anyway.
    
    return bad_sets

def _build_conflict_graph(universe, bad_sets):
    """
    Build adjacency bitmasks for the conflict graph.
    Two numbers are connected if they appear in the same minimal bad set.
    """
    n = len(universe)
    num_to_idx = {num: i for i, num in enumerate(universe)}
    
    adj_masks = [0] * n
    
    for bset in bad_sets:
        indices = [num_to_idx[x] for x in bset]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u = indices[i]
                v = indices[j]
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
    return adj_masks

def _max_independent_set(adj_masks, n):
    """
    Solve Maximum Independent Set using Branch and Bound with bitsets.
    Uses greedy coloring of the candidate subgraph for upper bounds.
    """
    best_set = 0
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_color_bound(candidates):
        """
        Returns an upper bound on the size of the MIS of the subgraph induced by 'candidates'.
        This is the chromatic number of the subgraph, which is >= size of any independent set.
        We compute a greedy coloring.
        """
        if not candidates:
            return 0
            
        bound = 0
        rem = candidates
        while rem:
            bound += 1
            # Pick a vertex to color with the current color (bound)
            # Standard greedy: pick the first available vertex, then remove it and its neighbors from 'rem'
            # This forms a maximal independent set in the remaining graph? No.
            # Greedy coloring:
            # 1. Pick a vertex v from rem.
            # 2. Assign color 'bound' to v.
            # 3. Remove v and all its neighbors in the subgraph from rem.
            #    Wait, if we remove neighbors, we are finding an independent set.
            #    The number of colors needed is the size of the partition into independent sets.
            #    So, each iteration of the while loop finds one independent set (color class).
            #    The number of iterations is the number of colors.
            
            # To make the bound tight, we should pick vertices strategically.
            # Usually, picking the vertex with the highest degree in the remaining graph helps.
            # But for simplicity and speed, picking the first one is often sufficient for small n.
            # Let's try picking the vertex with the highest degree in 'rem' to minimize the number of colors.
            
            best_v = -1
            max_deg = -1
            r = rem
            while r:
                bit = r & -r
                idx = bit.bit_length() - 1
                deg = popcount(adj_masks[idx] & rem)
                if deg > max_deg:
                    max_deg = deg
                    best_v = idx
                r ^= bit
            
            if best_v == -1:
                break
                
            v_bit = 1 << best_v
            # Remove best_v and its neighbors from rem
            # Neighbors in the subgraph are adj_masks[best_v] & rem
            rem &= ~(v_bit | (adj_masks[best_v] & rem))
            
        return bound

    def backtrack(candidates, current_set, current_size):
        nonlocal best_set, best_size
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound check
        ub = greedy_color_bound(candidates)
        if current_size + ub <= best_size:
            return
            
        # Branching heuristic: Pick a vertex.
        # Heuristic: Vertex with highest degree in the induced subgraph.
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            bit = c & -c
            idx = bit.bit_length() - 1
            deg = popcount(adj_masks[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = idx
            c ^= bit
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates are those NOT adjacent to best_v, excluding best_v.
        new_candidates_include = candidates & ~adj_masks[best_v] & ~v_bit
        backtrack(new_candidates_include, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates_exclude = candidates & ~v_bit
        backtrack(new_candidates_exclude, current_set, current_size)

    backtrack((1 << n) - 1, 0, 0)
    
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(i)
    return result

def construct():
    universe = list(range(1, 23))
    bad_sets = _find_minimal_bad_sets(universe)
    
    if not bad_sets:
        return universe
        
    adj_masks = _build_conflict_graph(universe, bad_sets)
    
    indices = _max_independent_set(adj_masks, len(universe))
    
    result = [universe[i] for i in indices]
    
    if not result:
        return [1]
        
    # Verification
    if not _is_dissociated(result):
        # Fallback to a known good set or greedy selection
        # A simple greedy approach: sort by value, add if dissociated
        greedy_set = []
        for num in sorted(universe, reverse=True):
            test_set = greedy_set + [num]
            if _is_dissociated(test_set):
                greedy_set.append(num)
        return greedy_set
        
    return result
# EVOLVE-BLOCK-END
