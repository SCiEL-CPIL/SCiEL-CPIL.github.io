"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Moduli range
    moduli = list(range(20, 201))
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    def get_blocked_residues(m_sel, r_sel, m_target):
        """
        Returns the set of residues in [0, m_target) that are blocked
        by the selection (m_sel, r_sel).
        """
        g = gcd_table.get((min(m_sel, m_target), max(m_sel, m_target)), math.gcd(m_sel, m_target))
        r_mod = r_sel % g
        # Residues r_t in [0, m_target) such that r_t % g == r_mod
        # These are r_mod, r_mod + g, r_mod + 2g, ...
        blocked = set()
        start = r_mod
        if start >= m_target:
            return blocked
        for r_t in range(start, m_target, g):
            blocked.add(r_t)
        return blocked

    def is_compatible(selected, m, r):
        """
        Check if (m, r) is compatible with all pairs in selected.
        selected: list of [m, r]
        """
        for m_sel, r_sel in selected:
            g = gcd_table.get((min(m, m_sel), max(m, m_sel)), math.gcd(m, m_sel))
            if r % g == r_sel % g:
                return False
        return True

    def greedy_construct(order):
        """
        Construct a solution by processing moduli in the given order.
        For each modulus, if compatible, pick the residue that minimizes
        the number of newly blocked residues for all subsequent moduli.
        """
        selected = []
        # Track blocked residues for each modulus? 
        # It's easier to just check compatibility on the fly.
        
        # To optimize the residue choice, we need to know which residues are already blocked
        # by previously selected moduli.
        # Let's maintain a dict: blocked_by[m] = set of residues in m blocked by selected.
        blocked_by = {m: set() for m in moduli}
        
        for m in order:
            # Find valid residues
            valid_residues = [r for r in range(m) if r not in blocked_by[m]]
            if not valid_residues:
                continue
            
            # If only one valid residue, pick it
            if len(valid_residues) == 1:
                r = valid_residues[0]
            else:
                # Pick the residue that minimizes the "damage" to future moduli
                # Damage = sum over future moduli n of | (blocked_by[n] U blocked(m,r,n)) - blocked_by[n] |
                # Which is sum of | blocked(m,r,n) - blocked_by[n] |
                
                best_r = None
                min_damage = float('inf')
                
                # We only care about future moduli in the order
                # But 'order' is a permutation. We can just iterate through all other moduli
                # that come after m in the order.
                # To speed up, we can precompute the position of each modulus in the order.
                pos_in_order = {m: i for i, m in enumerate(order)}
                current_pos = pos_in_order[m]
                
                for r in valid_residues:
                    damage = 0
                    for n in moduli:
                        if n == m or pos_in_order[n] < current_pos:
                            continue
                        # Calculate blocked residues of n by (m, r)
                        g = gcd_table.get((min(m, n), max(m, n)), math.gcd(m, n))
                        r_mod = r % g
                        # Count how many residues in n are blocked by this (m,r)
                        # and are NOT already in blocked_by[n]
                        # blocked_by[n] contains residues blocked by PREVIOUSLY selected moduli.
                        # We need to check if the AP defined by (m,r) adds new elements to blocked_by[n].
                        
                        # Instead of iterating all residues, we can check:
                        # The set of residues blocked by (m,r) is {k, k+g, ...}.
                        # We want to count how many of these are NOT in blocked_by[n].
                        
                        # Optimization: If blocked_by[n] is empty, the count is m_n/g.
                        # If blocked_by[n] is full, count is 0.
                        
                        # For simplicity and speed, we approximate damage.
                        # Or we can just pick the first valid residue if we are tight on time.
                        # Let's try a simple heuristic: pick r that is 'central' or random?
                        # Actually, the exact damage calculation might be slow.
                        
                        # Let's use a simpler heuristic:
                        # Pick the residue that is least likely to conflict.
                        # Since we don't know the future selections, any residue is symmetric w.r.t weight.
                        # The only difference is which residues they block.
                        # Blocking residues with high "value" (from small moduli) is bad.
                        # But we process in order.
                        
                        # Let's just pick the first valid residue to ensure speed.
                        # We can improve this later if time permits.
                        pass
                    
                    # Fallback: Pick the first valid residue
                    # This is O(1) per modulus.
                    # Let's implement a slightly better heuristic:
                    # Pick r such that r % gcd(m, n) is "common" with other blocked residues?
                    # Too complex.
                    
                    # Let's try: Pick the residue that minimizes the number of conflicts with 
                    # the *already selected* moduli? No, they are all compatible by definition of valid_residues.
                    
                    # Let's just pick the first one.
                    best_r = r
                    break # Just pick the first valid one for now to be fast
                    
                r = best_r
            
            # Select (m, r)
            selected.append([m, r])
            
            # Update blocked_by for all other moduli
            for n in moduli:
                if n == m:
                    continue
                g = gcd_table.get((min(m, n), max(m, n)), math.gcd(m, n))
                r_mod = r % g
                # Add blocked residues to blocked_by[n]
                start = r_mod
                if start < n:
                    for r_n in range(start, n, g):
                        blocked_by[n].add(r_n)
        
        return selected

    # Strategy:
    # 1. Try greedy with moduli sorted by increasing m (standard).
    # 2. Try greedy with moduli sorted by decreasing m? (Unlikely to be better).
    # 3. Try random restarts of greedy.
    
    best_solution = []
    best_score = Fraction(0, 1)
    
    # Precompute scores for all possible solutions? No.
    
    def score_solution(sol):
        return sum(Fraction(1, m) for m, r in sol)
    
    # Initial greedy
    order1 = sorted(moduli)
    sol1 = greedy_construct(order1)
    score1 = score_solution(sol1)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
    
    # Try a few random orders
    # The number of moduli is 181.
    # We have ~100 seconds left.
    # Each greedy run is fast (~1ms).
    # We can do 10,000 runs.
    
    # But random orders might not be better than sorted.
    # Sorted by m is good because small m have high weight.
    # Maybe we should prioritize moduli that are "easy" to fit?
    
    # Let's try a local search on the best solution found.
    # Move: Change residue of a selected modulus.
    # If we change residue of m from r to r', it must be compatible with all others.
    # And it might unblock some residues for other moduli?
    # No, changing r for m only changes which residues are blocked in OTHER moduli.
    # It does not change the set of selected moduli.
    # So the score is the same!
    # Unless... changing r allows us to ADD another modulus?
    # Yes! If we change r for m, we might unblock a residue for some other modulus n that was previously fully blocked.
    
    def local_search(sol, max_iters=1000):
        # sol is a list of [m, r]
        # We want to see if we can improve the score.
        # Since score depends only on the set of moduli, we need to ADD moduli.
        # To add a modulus, we need to free up a residue.
        
        # 1. Identify moduli that are NOT in sol.
        selected_moduli = set(m for m, r in sol)
        unselected_moduli = [m for m in moduli if m not in selected_moduli]
        
        # 2. For each unselected modulus, check if it's blocked.
        # If it is blocked, can we change a residue of a selected modulus to unblock it?
        
        # This is a complex search.
        # Simpler: Randomly perturb a residue of a selected modulus and see if a new modulus becomes available.
        
        best = list(sol)
        best_score_val = score_solution(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random selected modulus
            if not best:
                break
            idx = random.randint(0, len(best) - 1)
            m, r = best[idx]
            
            # Pick a new random residue r'
            r_prime = random.randint(0, m - 1)
            if r_prime == r:
                continue
            
            # Check if (m, r_prime) is compatible with all other selected moduli
            compatible = True
            for i, (m2, r2) in enumerate(best):
                if i == idx:
                    continue
                g = gcd_table.get((min(m, m2), max(m, m2)), math.gcd(m, m2))
                if r_prime % g == r2 % g:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Try the new configuration
            new_sol = [x[:] for x in best]
            new_sol[idx] = [m, r_prime]
            
            # Check if this allows adding any new modulus
            # We can just run a greedy fill on the unselected moduli with this new base?
            # That might be slow.
            
            # Instead, let's just check if any unselected modulus is now compatible.
            # This is O(N_unselected * N_selected).
            # 181 * 181 ~ 30k ops. Fast.
            
            added = False
            for m_new in unselected_moduli:
                # Check if m_new is compatible with new_sol
                ok = True
                for m_sel, r_sel in new_sol:
                    g = gcd_table.get((min(m_new, m_sel), max(m_new, m_sel)), math.gcd(m_new, m_sel))
                    if m_new % g == r_sel % g: # Wait, residue of m_new is not fixed yet.
                         # We need to check if there EXISTS a residue for m_new.
                         pass
                
                # We need to check if there is ANY valid residue for m_new given new_sol.
                valid = False
                for r_new in range(m_new):
                    ok = True
                    for m_sel, r_sel in new_sol:
                        g = gcd_table.get((min(m_new, m_sel), max(m_new, m_sel)), math.gcd(m_new, m_sel))
                        if r_new % g == r_sel % g:
                            ok = False
                            break
                    if ok:
                        valid = True
                        break
                
                if valid:
                    # We found a new modulus we can add!
                    # Add the first valid residue
                    # (We should probably pick the one that minimizes future damage, but let's just pick one)
                    # Re-find the residue
                    for r_new in range(m_new):
                        ok = True
                        for m_sel, r_sel in new_sol:
                            g = gcd_table.get((min(m_new, m_sel), max(m_new, m_sel)), math.gcd(m_new, m_sel))
                            if r_new % g == r_sel % g:
                                ok = False
                                break
                        if ok:
                            new_sol.append([m_new, r_new])
                            added = True
                            break
                    if added:
                        break
            
            if added:
                # Score increased
                new_score = score_solution(new_sol)
                if new_score > best_score_val:
                    best = new_sol
                    best_score_val = new_score
                    # Update unselected_moduli
                    selected_moduli = set(m for m, r in best)
                    unselected_moduli = [m for m in moduli if m not in selected_moduli]
        
        return best

    # Run local search on the initial greedy solution
    improved_sol = local_search(best_solution, max_iters=500)
    improved_score = score_solution(improved_sol)
    if improved_score > best_score:
        best_score = improved_score
        best_solution = improved_sol

    # Try a few more greedy starts with different tie-breaking or slight randomization
    for i in range(50):
        if time.time() > deadline:
            break
        # Randomize the order of moduli slightly?
        # Or just run local search again with different random seed?
        # Let's try a greedy that picks residues randomly from valid ones.
        
        def greedy_random(order):
            selected = []
            blocked_by = {m: set() for m in moduli}
            for m in order:
                valid_residues = [r for r in range(m) if r not in blocked_by[m]]
                if not valid_residues:
                    continue
                r = random.choice(valid_residues)
                selected.append([m, r])
                for n in moduli:
                    if n == m:
                        continue
                    g = gcd_table.get((min(m, n), max(m, n)), math.gcd(m, n))
                    r_mod = r % g
                    start = r_mod
                    if start < n:
                        for r_n in range(start, n, g):
                            blocked_by[n].add(r_n)
            return selected

        sol_r = greedy_random(order1)
        score_r = score_solution(sol_r)
        if score_r > best_score:
            best_score = score_r
            best_solution = sol_r
        else:
            # Try local search on this random solution
            sol_ls = local_search(sol_r, max_iters=100)
            score_ls = score_solution(sol_ls)
            if score_ls > best_score:
                best_score = score_ls
                best_solution = sol_ls

    return best_solution
# EVOLVE-BLOCK-END
