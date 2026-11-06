"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # All candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    n = len(candidates)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i, m1 in enumerate(candidates):
        for j, m2 in enumerate(candidates):
            gcd_table[i][j] = gcd(m1, m2)
    
    # Precompute for each pair (i, j) with gcd g:
    # conflict[i][j][r_i][r_j] = True if r_i % g == r_j % g
    # Instead, for each pair and each residue of i, precompute which residues of j are compatible
    # compatible[i][j] = list of length m_i, where compatible[i][j][r_i] is a bitmask of valid r_j values
    # Since m can be up to 200, use Python ints as bitmasks
    
    # For efficiency, precompute: for each pair (i,j) and each r_i in [0, m_i), 
    # the set of r_j in [0, m_j) such that r_i % g != r_j % g
    # This is: all r_j EXCEPT those where r_j % g == r_i % g
    
    # Number of r_j with r_j % g == target is either m_j//g or m_j//g + 1
    # Specifically: count = (m_j - target + g - 1) // g if target < m_j else 0... 
    # Actually: r_j in [0, m_j) with r_j % g == t:
    # These are t, t+g, t+2g, ... < m_j
    # Count = floor((m_j - 1 - t) / g) + 1 if t < m_j, else 0
    # More precisely: count = ((m_j - 1 - t) // g + 1) if t < m_j else 0
    # Or: count = (m_j - 1 - t) // g + 1 if t < m_j, but need t >= 0
    # Since t = r_i % g and 0 <= t < g <= m_j (since g = gcd(m_i, m_j) <= min(m_i, m_j) <= m_j),
    # so t < m_j always.
    # count = (m_j - 1 - t) // g + 1
    
    # For the bitmask approach: for given m_j and g and target t, the valid r_j are those 
    # NOT in {t, t+g, t+2g, ...}
    # We can precompute a "forbidden mask" for each (m_j, g, t)
    
    # Let's precompute forbidden_masks[m_j][g][t] = bitmask of r_j in [0, m_j) where r_j % g == t
    # Then compatible mask = full_mask[m_j] XOR forbidden_masks[m_j][g][t]
    
    # full_mask[m] = (1 << m) - 1
    full_mask = {}
    for m in candidates:
        full_mask[m] = (1 << m) - 1
    
    # forbidden_masks[m][g][t] for m in candidates, g dividing m, t in [0, g)
    # But g depends on the pair. Let's precompute for all m in [20,200], all g that could be gcd, all t
    # Actually, let's just compute on the fly with memoization
    
    forbidden_cache = {}
    def get_forbidden_mask(m, g, t):
        """Bitmask of r in [0, m) where r % g == t"""
        key = (m, g, t)
        if key in forbidden_cache:
            return forbidden_cache[key]
        mask = 0
        # r = t, t+g, t+2g, ... < m
        r = t
        while r < m:
            mask |= (1 << r)
            r += g
        forbidden_cache[key] = mask
        return mask
    
    # Now the main approach: AC-3 based constraint propagation + branch and bound
    # Variables: which moduli to include (subset of candidates)
    # For each included modulus m_i, we assign a residue r_i
    # Constraint: for any pair (i, j) both included, r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)
    
    # Strategy: 
    # 1. Start with all moduli, domains = all residues [0, m_i)
    # 2. Use AC-3 to prune domains based on pairwise constraints
    # 3. Use branch-and-bound: pick variable, try values, propagate, bound with sum of 1/m for remaining
    
    # Actually, the key insight: we want to MAXIMIZE sum of 1/m over included moduli.
    # So we should prefer including small moduli. The question is which subset can be assigned residues.
    
    # Let me think differently. The constraint is:
    # For moduli m1, m2 with gcd g: we need r1 % g != r2 % g
    # This means: for each "g-class" (residue mod g), at most one modulus from the set can use that g-class.
    
    # Key structural insight: If we group moduli by their gcd relationships...
    
    # Alternative approach: Think of it as a graph coloring / independent set problem.
    # But the structure is more complex.
    
    # Let me try a different structural approach:
    # Process moduli in order of increasing m (highest weight first).
    # For each modulus, decide include or exclude.
    # If include, assign a residue that's compatible with all previously included moduli.
    # Use memoization / DP where the state is the set of "used residue classes" for each gcd value.
    
    # The state space: for each possible gcd value g (2..200), which residues mod g are "used"
    # But this is too large.
    
    # Better: Note that if m1 | m2, then gcd(m1, m2) = m1, so r1 % m1 != r2 % m1, meaning r1 != r2 % m1.
    # If gcd(m1, m2) = 1, then no constraint (always disjoint).
    
    # Key observation: The constraint only depends on gcd(m1, m2).
    # For a fixed modulus m, its constraints with other moduli m' depend on gcd(m, m').
    
    # Let me try: greedy with lookahead + local search, but with a better representation.
    # Represent the "used" constraints as: for each divisor d of the selected moduli, which residue classes mod d are taken.
    
    # Actually, let me try a cleaner branch-and-bound:
    # Order moduli by 1/m descending (i.e., m ascending).
    # At each step, decide to include or exclude the next modulus.
    # If include, find a valid residue.
    # Upper bound: current_score + sum of 1/m for all remaining moduli (optimistic).
    # Prune if upper bound <= best found.
    
    # To make this work, I need to efficiently check if a residue is valid and find valid residues.
    
    # State for checking validity: set of (m, r) pairs already selected.
    # For new (m_new, r_new): check for each (m_old, r_old): r_new % gcd(m_new, m_old) != r_old % gcd(m_new, m_old)
    
    # Let me implement a systematic branch-and-bound with good pruning.
    
    best_score = Fraction(0, 1)
    best_solution = []
    
    # Sort moduli by 1/m descending = m ascending
    ordered_mods = sorted(candidates)
    
    # Precompute: for each modulus m and each potential gcd g with another modulus,
    # what residue classes mod g are "blocked" by existing selections
    
    # I'll use a different representation: maintain for each possible gcd value g (2 to 200),
    # a set of residue classes mod g that are "claimed" by some selected modulus.
    # When adding modulus m with residue r, for each other selected modulus m', 
    # g = gcd(m, m'), and we claim residue r % g in the g-class.
    # But this isn't quite right because the same g-class residue could be claimed by multiple moduli
    # with different m values, and that's fine as long as they don't conflict.
    
    # Actually, the constraint is pairwise. Let me just maintain the list of selected (m, r) pairs
    # and check compatibility directly. With at most ~180 moduli, this is O(n) per check.
    
    def is_compatible(m_new, r_new, selected):
        for (m_old, r_old) in selected:
            g = gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def find_valid_residues(m_new, selected):
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected):
                valid.append(r)
        return valid
    
    # Branch and bound
    # To make this tractable, I'll use a greedy initialization + local improvement approach
    # but with a different structure: AC-3 style propagation
    
    # Let me try: iterative deepening / systematic search with strong pruning
    # Process moduli in order. At each step, branch on include/exclude.
    # Use upper bound: current + sum(1/m for remaining)
    # This might be too slow for 180 moduli.
    
    # Better approach: Greedy with multiple passes + targeted local search
    
    # Phase 1: Greedy construction
    def greedy_construct(order):
        selected = []
        for m in order:
            if time.time() > deadline - 30:
                break
            valid_residues = find_valid_residues(m, selected)
            if valid_residues:
                # Choose residue that maximizes future flexibility
                # Heuristic: choose residue that blocks the fewest future options
                best_r = valid_residues[0]
                best_block_count = float('inf')
                for r in valid_residues:
                    block_count = 0
                    for m2 in order:
                        if m2 <= m:
                            continue
                        g = gcd(m, m2)
                        target = r % g
                        # Count how many residues of m2 would be blocked
                        # A residue r2 of m2 is blocked if r2 % g == target
                        # Number of such r2: approximately m2/g
                        # But we only care about whether it conflicts
                        # Actually, this residue choice blocks exactly the g-class 'target' for m2
                        # The "cost" is roughly 1/g of m2's domain
                        block_count += 1.0 / g
                    if block_count < best_block_count:
                        best_block_count = block_count
                        best_r = r
                selected.append((m, best_r))
        return selected
    
    # Phase 1: Multiple greedy strategies
    states = []
    
    # Strategy 1: Ascending order
    s1 = greedy_construct(ordered_mods)
    sc1 = sum(Fraction(1, m) for m, r in s1)
    states.append((s1, sc1))
    
    # Strategy 2: Descending order  
    s2 = greedy_construct(list(reversed(ordered_mods)))
    sc2 = sum(Fraction(1, m) for m, r in s2)
    states.append((s2, sc2))
    
    # Strategy 3: Randomized orders
    import random
    rng = random.Random(12345)
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        order = list(ordered_mods)
        rng.shuffle(order)
        s = greedy_construct(order)
        sc = sum(Fraction(1, m) for m, r in s)
        states.append((s, sc))
    
    # Pick best
    best_state = max(states, key=lambda x: x[1])
    best_selected = best_state[0]
    best_score = best_state[1]
    
    # Phase 2: Local search - try to add missing moduli
    def local_search_improve(selected, max_time=30):
        selected = list(selected)
        score = sum(Fraction(1, m) for m, r in selected)
        end_time = time.time() + max_time
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            used_mods = set(m for m, r in selected)
            
            # Try adding each missing modulus
            for m in ordered_mods:
                if m in used_mods:
                    continue
                valid_residues = find_valid_residues(m, selected)
                if valid_residues:
                    # Try each valid residue, pick the one that enables most future additions
                    best_r = valid_residues[0]
                    best_future = -1
                    for r in valid_residues:
                        # Count how many additional moduli could be added after this choice
                        new_selected = selected + [(m, r)]
                        future_count = 0
                        for m2 in ordered_mods:
                            if m2 in used_mods or m2 == m:
                                continue
                            if find_valid_residues(m2, new_selected):
                                future_count += 1
                        if future_count > best_future:
                            best_future = future_count
                            best_r = r
                    
                    new_selected = selected + [(m, best_r)]
                    new_score = score + Fraction(1, m)
                    if new_score > score:
                        selected = new_selected
                        score = new_score
                        improved = True
                        break
        
        return selected, score
    
    best_selected, best_score = local_search_improve(best_selected, max_time=25)
    
    # Phase 3: Try swapping - remove one modulus, try to add two others
    def try_swaps(selected, max_time=15):
        selected = list(selected)
        score = sum(Fraction(1, m) for m, r in selected)
        end_time = time.time() + max_time
        
        for i in range(len(selected)):
            if time.time() > end_time:
                break
            m_remove, r_remove = selected[i]
            without = selected[:i] + selected[i+1:]
            score_without = score - Fraction(1, m_remove)
            
            # Try to add one or two moduli that were previously blocked
            used_mods = set(m for m, r in without)
            additions = []
            for m in ordered_mods:
                if m in used_mods:
                    continue
                valid_residues = find_valid_residues(m, without)
                if valid_residues:
                    additions.append((m, valid_residues[0]))
            
            # Try adding the best additions
            for m_add, r_add in additions:
                new_selected = without + [(m_add, r_add)]
                new_score = score_without + Fraction(1, m_add)
                if new_score > score:
                    selected = new_selected
                    score = new_score
                    break
            
            # Try adding two
            if len(additions) >= 2:
                for i_add in range(len(additions)):
                    if time.time() > end_time:
                        break
                    m1_add, r1_add = additions[i_add]
                    new_selected = without + [(m1_add, r1_add)]
                    for j_add in range(len(additions)):
                        if i_add == j_add:
                            continue
                        m2_add, r2_add = additions[j_add]
                        if m2_add in set(m for m, r in new_selected):
                            continue
                        if find_valid_residues(m2_add, new_selected):
                            # Find a valid residue
                            valid_r2 = find_valid_residues(m2_add, new_selected)
                            if valid_r2:
                                final_selected = new_selected + [(m2_add, valid_r2[0])]
                                final_score = score_without + Fraction(1, m1_add) + Fraction(1, m2_add)
                                if final_score > score:
                                    selected = final_selected
                                    score = final_score
                                    break
                    if time.time() > end_time:
                        break
        
        return selected, score
    
    best_selected, best_score = try_swaps(best_selected, max_time=15)
    
    # Phase 4: Final greedy fill
    used_mods = set(m for m, r in best_selected)
    for m in ordered_mods:
        if m in used_mods:
            continue
        if time.time() > deadline - 5:
            break
        valid_residues = find_valid_residues(m, best_selected)
        if valid_residues:
            best_selected.append((m, valid_residues[0]))
            best_score += Fraction(1, m)
    
    # Convert to required format
    result = [[m, r] for m, r in best_selected]
    return result
# EVOLVE-BLOCK-END
