"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    """
    Multi-start randomized greedy search with clique pruning and local improvement.
    
    Key Insight: If gcd(m1, m2) == 1, the classes always intersect. 
    Thus, the set of moduli must be pairwise non-coprime (a clique in the non-coprime graph).
    """
    if deadline is None:
        deadline = time.time() + 85  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Pre-compute gcds and non-coprime adjacency
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    non_coprime_neighbors = defaultdict(set)
    
    all_moduli = list(range(2, max_mod+1))
    
    for m1 in all_moduli:
        for m2 in all_moduli:
            g = gcd(m1, m2)
            gcd_table[m1][m2] = g
            if g > 1:
                non_coprime_neighbors[m1].add(m2)
                
    # Calculate a "centrality" score for each modulus to guide heuristics
    # Higher degree in non-coprime graph means it can connect to more others
    degree = {m: len(non_coprime_neighbors[m]) for m in all_moduli}
    
    def _try_add(modulus, current_set, used_mods):
        """
        Try to add modulus to current_set.
        Returns (new_set, new_used_mods) if successful, else (None, None).
        """
        # 1. Hard Constraint: Must share a factor with ALL existing moduli
        for r_old, m_old in current_set:
            if gcd_table[modulus][m_old] == 1:
                return None, None
        
        # 2. Soft Constraint: Find a residue r that is disjoint from all existing
        # We need r such that for all (r_old, m_old): (r - r_old) % gcd(modulus, m_old) != 0
        
        # Optimization: Group constraints by gcd value
        # For a specific gcd g, r cannot be congruent to r_old mod g.
        # Since m <= 128, simple iteration is fast enough.
        
        for r in range(modulus):
            ok = True
            for r_old, m_old in current_set:
                g = gcd_table[modulus][m_old]
                if (r - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                new_set = current_set + [[r, modulus]]
                new_used = used_mods | {modulus}
                return new_set, new_used
        return None, None

    def _local_search(current_set, used_mods, deadline):
        """
        Try to improve current_set by adding more moduli.
        Prioritize moduli with high degree and low "resistance" (many shared factors).
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Candidate moduli not yet used
            candidates = [m for m in all_moduli if m not in used_mods]
            
            # Sort candidates by heuristic: 
            # 1. Must be non-coprime with all in current set (pre-filter)
            # 2. Prefer higher degree (more connections)
            # 3. Prefer smaller modulus? (Often easier to find residue? Not necessarily, but let's try)
            
            valid_candidates = []
            for m in candidates:
                # Quick check: is m non-coprime with all in current_set?
                is_clique_member = True
                for _, m_old in current_set:
                    if gcd_table[m][m_old] == 1:
                        is_clique_member = False
                        break
                if is_clique_member:
                    valid_candidates.append(m)
            
            if not valid_candidates:
                break
            
            # Sort by degree descending, then random tie-breaker
            random.shuffle(valid_candidates)
            valid_candidates.sort(key=lambda x: -degree[x])
            
            # Try to add the best candidates
            for m in valid_candidates:
                new_set, new_used = _try_add(m, current_set, used_mods)
                if new_set is not None:
                    current_set = new_set
                    used_mods = new_used
                    improved = True
                    break
                    
        return current_set, used_mods

    def _order_by_degree(moduli):
        """Order by non-coprime degree descending."""
        return sorted(moduli, key=lambda x: -degree[x])
    
    def _order_by_composite_score(moduli):
        """Order by number of divisors (proxy for composite-ness)."""
        def score(m):
            divs = 0
            for i in range(2, int(m**0.5)+1):
                if m % i == 0:
                    divs += 2
                    if i*i == m:
                        divs -= 1
            return divs
        return sorted(moduli, key=score, reverse=True)

    start_time = time.time()
    max_iters = 10000
    
    # Strategies for initial ordering
    order_strategies = [
        _order_by_degree,
        _order_by_composite_score,
        lambda mods: (random.shuffle(mods), mods)[1], # Random
        lambda mods: sorted(mods, reverse=True), # Descending
        lambda mods: sorted(mods), # Ascending
    ]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Cycle through strategies, but weight random higher as it explores more
        if it % 5 == 0:
             strategy_idx = 0 # Degree
        elif it % 5 == 1:
             strategy_idx = 1 # Composite
        elif it % 5 == 2:
             strategy_idx = 2 # Random
        elif it % 5 == 3:
             strategy_idx = 3 # Desc
        else:
             strategy_idx = 4 # Asc

        moduli = all_moduli[:]
        order_strategies[strategy_idx](moduli)
        
        current_set = []
        used_mods = set()
        
        # Greedy construction
        for m in moduli:
            # Prune: if m is coprime with any in current_set, skip immediately
            # This is O(N) per candidate, but N is small and prunes heavily
            can_add = True
            for _, m_old in current_set:
                if gcd_table[m][m_old] == 1:
                    can_add = False
                    break
            
            if not can_add:
                continue
                
            new_set, new_used = _try_add(m, current_set, used_mods)
            if new_set is not None:
                current_set = new_set
                used_mods = new_used
        
        # Local search improvement
        current_set, used_mods = _local_search(current_set, used_mods, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
            
            # If we found a very large set, we might stop early? 
            # Theoretical max is likely around 60+ (all even numbers).
            # If we hit 60+, keep searching for 61, 62...
    
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
