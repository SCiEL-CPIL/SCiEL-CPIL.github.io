import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Moduli range
    MOD_MIN = 20
    MOD_MAX = 200
    all_moduli = list(range(MOD_MIN, MOD_MAX + 1))
    
    best_score = 0.0
    best_solution = []
    
    # Helper to check if a candidate (m, r) is compatible with a list of selected pairs
    def is_compatible(m, r, selected):
        for m_sel, r_sel in selected:
            g = gcd(m, m_sel)
            if r % g == r_sel % g:
                return False
        return True

    # Helper to find a valid residue for m given selected pairs
    # Returns None if no valid residue exists
    def find_valid_residue(m, selected):
        # Try residues in a specific order. 
        # Often residue 0 or small residues are "tighter" or "looser" depending on gcds.
        # We can randomize or try all.
        # For speed, we can just iterate 0 to m-1.
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None

    # Calculate score
    def calc_score(selected):
        return sum(1.0 / m for m, r in selected)

    # Local Search Step: Try to improve a given solution
    def local_search(selected, deadline):
        selected = [pair[:] for pair in selected] # Deep copy
        improved = True
        while improved and time.time() < deadline:
            improved = False
            score = calc_score(selected)
            
            # 1. Try to add unselected moduli if they fit
            # Sort unselected by weight (1/m) descending
            selected_mods = set(m for m, r in selected)
            unselected = [m for m in all_moduli if m not in selected_mods]
            unselected.sort(key=lambda x: -1.0/x)
            
            for m in unselected:
                if time.time() > deadline: break
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append([m, r])
                    improved = True
                    # Restart the loop to see if adding this opens up others? 
                    # Or just continue. Continuing is faster.
            
            # 2. Try swaps: Remove a set of conflicting moduli to add a high-value one
            # Identify "blocked" high-value moduli
            selected_mods = set(m for m, r in selected)
            unselected = [m for m in all_moduli if m not in selected_mods]
            unselected.sort(key=lambda x: -1.0/x)
            
            for m_new in unselected[:50]: # Limit candidates for speed
                if time.time() > deadline: break
                # Find conflicts
                conflicts = []
                for m_sel, r_sel in selected:
                    g = gcd(m_new, m_sel)
                    # Check if ALL residues of m_new conflict with r_sel?
                    # No, we just need to know if m_new CAN be added if we remove m_sel.
                    # If we remove m_sel, does m_new have a valid residue against the rest?
                    # Actually, we need to check if m_new is blocked by m_sel.
                    # m_new is blocked by m_sel if for ALL r_new in [0, m_new), 
                    # r_new % g == r_sel % g is false? No.
                    # m_new is compatible with m_sel if there exists r_new such that r_new % g != r_sel % g.
                    # Since g < m_new (usually), there are m_new/g residues. 
                    # Only 1 residue class mod g conflicts. 
                    # So m_new is NEVER completely blocked by a single m_sel unless g == m_new (impossible as m_sel < m_new? no, m_sel can be anything).
                    # Wait, if gcd(m_new, m_sel) = g, then r_new must not be congruent to r_sel mod g.
                    # There are m_new/g choices for r_new. Only m_new/g * (1/g) ? No.
                    # Total residues m_new. Bad residues: those where r_new % g == r_sel % g.
                    # Count = m_new / g.
                    # So m_new is blocked by m_sel only if m_new / g == m_new => g=1? No.
                    # If g=1, no conflict.
                    # If g>1, there are always valid residues for m_new against m_sel alone.
                    # So a modulus is blocked only by the *combination* of constraints?
                    # No, the constraints are pairwise. 
                    # r_new must satisfy: r_new % g_i != r_i % g_i for all selected i.
                    # If we remove a subset S of selected, we only need to satisfy the rest.
                    
                    # Let's just try removing each selected modulus one by one and see if m_new fits.
                    # If removing one isn't enough, try removing pairs? Too slow.
                    # Let's stick to single removals or "greedy removal".
                    
                    pass # This logic is getting complex. Let's use a simpler heuristic.
            
            # Simpler Local Search: 
            # Try to change residue of an existing modulus to allow adding another.
            # Or, try to remove a low-value modulus and add a high-value one if it fits.
            
            selected_mods = set(m for m, r in selected)
            unselected = [m for m in all_moduli if m not in selected_mods]
            unselected.sort(key=lambda x: -1.0/x)
            
            # Try adding any unselected that fits
            for m in unselected:
                if time.time() > deadline: break
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append([m, r])
                    improved = True
            
            # Try swapping: 
            # For each unselected high-value m_new, find the minimal set of selected moduli to remove
            # such that m_new fits.
            # Approximation: Try removing each selected m_sel one by one.
            # If m_new fits without m_sel, and 1/m_new > 1/m_sel, do it.
            
            for m_new in unselected[:20]:
                if time.time() > deadline: break
                if m_new in selected_mods: continue # already added above
                
                # Check if m_new fits now
                if find_valid_residue(m_new, selected) is not None:
                    # It fits, should have been added in previous loop. 
                    # If not, maybe it was added but we are in a new iteration?
                    # The previous loop adds ALL that fit. So if it's still here, it doesn't fit.
                    pass
                
                # Try removing one selected modulus
                for i, (m_sel, r_sel) in enumerate(selected):
                    if time.time() > deadline: break
                    # Temporarily remove
                    temp_selected = selected[:i] + selected[i+1:]
                    r_new = find_valid_residue(m_new, temp_selected)
                    if r_new is not None:
                        # Check if swap is beneficial
                        if 1.0/m_new > 1.0/m_sel:
                            # Perform swap
                            selected = temp_selected + [[m_new, r_new]]
                            improved = True
                            break
                if improved:
                    break # Restart loop
            
            # Try residue changes:
            # Does changing r for a selected m allow adding another?
            # This is expensive. Skip for now if time is tight.
            
        return selected

    # Run multiple restarts with different random seeds/orderings
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(i)
        
        # Greedy Initialization
        selected = []
        # Shuffle moduli slightly or use random tie-breaking?
        # Standard: sort by 1/m descending.
        # To diversify, we can shuffle the order of checking residues or the order of moduli if weights are equal (none are).
        # Let's just use standard greedy but with random residue selection if multiple fit?
        # find_valid_residue returns the first one.
        # Let's modify find_valid_residue to pick randomly from valid ones?
        # That might be better for diversity.
        
        def find_valid_residue_random(m, selected):
            valid_res = []
            for r in range(m):
                if is_compatible(m, r, selected):
                    valid_res.append(r)
            if valid_res:
                return random.choice(valid_res)
            return None
        
        # Greedy Phase
        for m in all_moduli: # Increasing m = Decreasing weight
            if time.time() > deadline: break
            r = find_valid_residue_random(m, selected)
            if r is not None:
                selected.append([m, r])
        
        # Local Search Phase
        selected = local_search(selected, deadline)
        
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_solution = [pair[:] for pair in selected]
            
        # Optional: If we found a very good score, we can tighten deadline or break?
        # No, keep searching.
        
    return best_solution

# Quick sanity check / optimization
# The local search above tries single swaps. 
# A stronger move is to try to add a modulus by removing a *set* of moduli.
# But single swaps are often sufficient for this kind of packing problem.
# Also, the "add if fits" loop in local_search is crucial.

# Let's refine the local search to be more aggressive with swaps.
# The current local_search in the draft above has a flaw: 
# It adds all fitting moduli first. Then it tries swaps.
# But after a swap, new moduli might fit. The `while improved` loop handles this.

# One issue: `find_valid_residue` is O(N*M) where N is selected count, M is m.
# N is at most 181. M is at most 200. 181*200 = 36k ops. 
# 181 moduli * 36k = ~6M ops for one greedy pass. Very fast.
# Local search iterations might be more, but still fast.

# The main bottleneck is the number of restarts and local search iterations.
# 140s is plenty for this.

# Let's double check the disjoint condition.
# "pairwise disjoint (r_1 != r_2 modulo gcd(m_1, m_2))"
# My `is_compatible` checks `r % g != r_sel % g`. This is correct.

# Final check on return type: list of lists [[m, r], ...]
# The function returns `best_solution` which is a list of lists.

# One improvement: In `local_search`, when trying swaps, we iterate `unselected[:20]`.
# We should probably sort unselected by weight every time or maintain a sorted list.
# The list `unselected` is created inside the loop, so it's fresh.

# Let's ensure `best_solution` is valid.
# The construction guarantees validity.

# To improve score further, we can try a "simulated annealing" style approach
# or just more aggressive local search.
# Given the time limit, the current approach (Greedy + Local Search + Restarts) is robust.

# One specific improvement: 
# In the greedy phase, picking the *smallest* valid residue is often better than random, 
# as it might leave "larger" residues for moduli that have fewer options? 
# Or vice versa. 
# Actually, for moduli with large gcds, the constraint is tight. 
# Let's stick to random choice in greedy for diversity, but maybe try a "smallest residue" strategy in one of the restarts.

# Revised Plan for `construct`:
# 1. Run greedy with "smallest residue" strategy.
# 2. Run greedy with "random residue" strategy.
# 3. Apply local search to both.
# 4. Keep best.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    MOD_MIN = 20
    MOD_MAX = 200
    all_moduli = list(range(MOD_MIN, MOD_MAX + 1))
    
    best_score = 0.0
    best_solution = []
    
    def is_compatible(m, r, selected):
        for m_sel, r_sel in selected:
            g = gcd(m, m_sel)
            if r % g == r_sel % g:
                return False
        return True

    def find_valid_residue(m, selected, randomize=False, rng=None):
        valid_res = []
        for r in range(m):
            if is_compatible(m, r, selected):
                valid_res.append(r)
        if not valid_res:
            return None
        if randomize and rng:
            return rng.choice(valid_res)
        return valid_res[0] # Smallest

    def calc_score(selected):
        return sum(1.0 / m for m, r in selected)

    def local_search(selected, deadline, rng):
        selected = [pair[:] for pair in selected]
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            if iterations > 1000: # Safety break for infinite loops
                break
                
            selected_mods = set(m for m, r in selected)
            
            # 1. Add any fitting unselected
            unselected = [m for m in all_moduli if m not in selected_mods]
            unselected.sort(key=lambda x: -1.0/x)
            
            added_any = False
            for m in unselected:
                if time.time() > deadline: break
                r = find_valid_residue(m, selected, randomize=True, rng=rng)
                if r is not None:
                    selected.append([m, r])
                    selected_mods.add(m)
                    improved = True
                    added_any = True
            if added_any:
                continue # Restart loop to check for more additions
            
            # 2. Try swaps (Remove 1, Add 1)
            # We look for a high-value m_new that is blocked, and a low-value m_sel 
            # whose removal allows m_new.
            
            # Precompute conflicts for unselected high-value moduli?
            # Just iterate.
            unselected_high = [m for m in unselected if 1.0/m > 1.0/100] # Arbitrary threshold? 
            # Better: just take top 50 unselected by weight.
            unselected.sort(key=lambda x: -1.0/x)
            candidates_new = unselected[:50]
            
            for m_new in candidates_new:
                if time.time() > deadline: break
                
                # Check if m_new fits now (it shouldn't, from step 1)
                if find_valid_residue(m_new, selected) is not None:
                    r = find_valid_residue(m_new, selected)
                    selected.append([m_new, r])
                    improved = True
                    break
                
                # Try removing each selected m_sel
                for i, (m_sel, r_sel) in enumerate(selected):
                    if time.time() > deadline: break
                    # Only consider swapping if 1/m_new > 1/m_sel
                    if 1.0/m_new <= 1.0/m_sel:
                        continue
                    
                    temp_selected = selected[:i] + selected[i+1:]
                    r_new = find_valid_residue(m_new, temp_selected)
                    if r_new is not None:
                        # Perform swap
                        selected = temp_selected + [[m_new, r_new]]
                        improved = True
                        break
                if improved:
                    break
            
            # 3. Try 2-for-1 swaps? (Remove 2, Add 1)
            # This is O(N^2) per m_new. Might be too slow.
            # Let's stick to 1-for-1 for now.
            
        return selected

    # Strategy 1: Greedy with smallest residue
    selected1 = []
    for m in all_moduli:
        if time.time() > deadline: break
        r = find_valid_residue(m, selected1)
        if r is not None:
            selected1.append([m, r])
    
    sol1 = local_search(selected1, deadline, random)
    score1 = calc_score(sol1)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1

    # Strategy 2: Greedy with random residue
    rng = random.Random(42)
    selected2 = []
    for m in all_moduli:
        if time.time() > deadline: break
        r = find_valid_residue(m, selected2, randomize=True, rng=rng)
        if r is not None:
            selected2.append([m, r])
    
    sol2 = local_search(selected2, deadline, rng)
    score2 = calc_score(sol2)
    if score2 > best_score:
        best_score = score2
        best_solution = sol2

    # Strategy 3: Multiple random restarts
    while time.time() < deadline - 10: # Leave 10s buffer
        rng = random.Random()
        selected_r = []
        # Randomize the order of moduli? No, weights are fixed. 
        # Randomize residue choice.
        for m in all_moduli:
            if time.time() > deadline: break
            r = find_valid_residue(m, selected_r, randomize=True, rng=rng)
            if r is not None:
                selected_r.append([m, r])
        
        sol_r = local_search(selected_r, deadline, rng)
        score_r = calc_score(sol_r)
        if score_r > best_score:
            best_score = score_r
            best_solution = sol_r

    return best_solution

