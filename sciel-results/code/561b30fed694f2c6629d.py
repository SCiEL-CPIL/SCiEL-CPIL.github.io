import math
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Two classes (m1, r1) and (m2, r2) are disjoint iff r1 != r2 mod gcd(m1, m2).
    """
    deadline = time.time() + 115  # Safety margin before 150s limit
    rng = random.Random(42)
    
    min_m = 20
    max_m = 200
    all_moduli = list(range(min_m, max_m + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def find_residues(moduli_set, max_restarts=50):
        """
        Attempts to find valid residues for the given set of moduli.
        Returns a dict {m: r} if successful, None otherwise.
        Uses randomized greedy with backtracking-like restarts.
        """
        if not moduli_set:
            return {}
        
        # Sort moduli by size? Or by number of constraints?
        # Smaller moduli have fewer residues, so they are more constrained?
        # Actually, larger moduli have more residues, so they are easier to satisfy.
        # So we should assign residues to smaller moduli first? 
        # Or larger? 
        # If m is large, it has many choices. If m is small, it has few.
        # Usually, assigning the most constrained variable first is better.
        # So sort by m ascending.
        sorted_moduli = sorted(moduli_set)
        
        for _ in range(max_restarts):
            assignment = {}
            success = True
            # Randomize the order of moduli to break symmetry
            current_moduli = list(sorted_moduli)
            rng.shuffle(current_moduli)
            
            for m in current_moduli:
                valid_residues = []
                # Check all residues r in 0..m-1
                for r in range(m):
                    is_valid = True
                    for m2, r2 in assignment.items():
                        if m2 == m: continue
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            is_valid = False
                            break
                    if is_valid:
                        valid_residues.append(r)
                
                if not valid_residues:
                    success = False
                    break
                
                # Choose randomly from valid residues to diversify
                r_choice = rng.choice(valid_residues)
                assignment[m] = r_choice
            
            if success:
                return assignment
        
        return None

    def calc_score(moduli_set):
        score = 0.0
        for m in moduli_set:
            score += 1.0 / m
        return score

    # Phase 1: Greedy Construction
    # Start with empty set, add moduli in increasing order (highest value 1/m)
    # Check feasibility after each addition.
    
    best_moduli = set()
    best_score = 0.0
    best_assignment = {}
    
    # Try to build a large feasible set
    # We want to include as many small moduli as possible.
    
    # Strategy: Iterate moduli 20..200. 
    # If adding m to current set keeps it feasible, add it.
    # This is a simple greedy.
    
    current_moduli = set()
    for m in range(min_m, max_m + 1):
        trial_moduli = current_moduli | {m}
        # Feasibility check is expensive. 
        # Optimization: Only check if the new modulus conflicts with existing ones in a way that *might* be resolvable?
        # No, just run the solver. 
        # To speed up, we can cache feasibility of subsets? No, subsets are too many.
        
        # We can try to assign residues incrementally?
        # The current find_residues is O(N^2 * M * Restarts). 
        # N=181, M=200. 181^2 * 200 * 50 is too slow if called 181 times.
        
        # We need a faster feasibility check or a smarter construction.
        pass

    # Let's use the existing greedy logic but improved.
    # The existing code does:
    # for m in 20..200:
    #   find r for m such that it doesn't conflict with existing.
    # This is effectively the same as checking feasibility of the set built so far + m.
    # And it's fast because it builds incrementally.
    
    def greedy_construct():
        selected = [] # list of [m, r]
        for m in range(min_m, max_m + 1):
            # Find any valid r for m
            found = False
            # Try residues in random order? Or 0..m-1?
            # Random order might help avoid getting stuck in local minima of residue choices
            # that block future moduli.
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                valid = True
                for m2, r2 in selected:
                    g = gcd_table[m][m2]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    selected.append([m, r])
                    found = True
                    break
        return selected

    # Run greedy multiple times with different seeds/orders
    best_sol = None
    best_score = 0.0
    
    start_time = time.time()
    for i in range(50):
        if time.time() - start_time > 10: break
        rng_seed = i
        local_rng = random.Random(rng_seed)
        # Monkey patch rng? No, just pass it or use a closure.
        # The greedy_construct above uses the global rng. 
        # Let's make it accept a rng.
        
        def greedy_construct_local(rng_local):
            selected = []
            for m in range(min_m, max_m + 1):
                found = False
                residues = list(range(m))
                rng_local.shuffle(residues)
                for r in residues:
                    valid = True
                    for m2, r2 in selected:
                        g = gcd_table[m][m2]
                        if r % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        selected.append([m, r])
                        found = True
                        break
            return selected
        
        sol = greedy_construct_local(local_rng)
        score = calc_score([m for m, _ in sol])
        if score > best_score:
            best_score = score
            best_sol = sol
    
    # Phase 2: Local Search on the Set of Moduli
    # We have a feasible set of moduli.
    # We want to maximize sum 1/m.
    # Moves:
    # 1. Add a modulus m not in set. Check if new set is feasible.
    # 2. Remove a modulus m from set. (Always feasible, score decreases)
    # 3. Swap m for m'.
    
    # To check feasibility of a new set, we use find_residues.
    # We can speed this up by starting from the previous assignment and trying to extend/modify it.
    
    current_moduli = {m for m, _ in best_sol}
    current_assignment = {m: r for m, r in best_sol}
    current_score = best_score
    
    def try_add_modulus(m_add, current_moduli, current_assignment):
        """
        Tries to add m_add to the current solution.
        Returns (new_moduli, new_assignment, new_score) if successful, else None.
        """
        if m_add in current_moduli:
            return None
        
        # Optimization: Check if m_add conflicts with any existing modulus in a way that makes it impossible?
        # No, just run the solver on the new set.
        # To make it faster, we can try to reuse the old assignment.
        # 1. Keep old residues.
        # 2. Find a residue for m_add that is compatible with all old ones.
        # 3. If no such residue, we might need to change old residues.
        #    This requires a full CSP solve.
        
        # Step 1: Try to find r for m_add compatible with current_assignment
        valid_r = []
        for r in range(m_add):
            ok = True
            for m2, r2 in current_assignment.items():
                g = gcd_table[m_add][m2]
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_r.append(r)
        
        if valid_r:
            new_assignment = dict(current_assignment)
            new_assignment[m_add] = rng.choice(valid_r)
            new_moduli = set(current_moduli)
            new_moduli.add(m_add)
            new_score = current_score + 1.0/m_add
            return new_moduli, new_assignment, new_score
        
        # Step 2: If no direct fit, run full solver on new_moduli
        # This is expensive. Only do this if we have time.
        new_moduli = set(current_moduli)
        new_moduli.add(m_add)
        
        # Use a limited number of restarts for the solver
        new_assignment = find_residues(new_moduli, max_restarts=20)
        if new_assignment:
            new_score = current_score + 1.0/m_add
            return new_moduli, new_assignment, new_score
            
        return None

    # Local Search Loop
    sa_start = time.time()
    while time.time() - sa_start < 90:
        # Try to add a random modulus
        # Bias towards smaller moduli (higher value)
        available_moduli = [m for m in all_moduli if m not in current_moduli]
        if not available_moduli:
            break
            
        # Sort available by value descending, pick from top K
        available_moduli.sort(key=lambda m: 1.0/m, reverse=True)
        k = min(10, len(available_moduli))
        m_candidate = rng.choice(available_moduli[:k])
        
        result = try_add_modulus(m_candidate, current_moduli, current_assignment)
        if result:
            current_moduli, current_assignment, current_score = result
            if current_score > best_score:
                best_score = current_score
                best_sol = [[m, r] for m, r in current_assignment.items()]
        
        # Try to remove a modulus and add a better one?
        # Or just try to add more.
        # If we can't add, maybe we should remove a large modulus to allow two smaller ones?
        # This is a "swap" or "replace" move.
        
        # Let's try a swap: Remove m_old, try to add m_new (where 1/m_new > 1/m_old is not guaranteed, but maybe m_new allows other additions?)
        # Actually, since we are maximizing sum 1/m, we should always prefer smaller moduli.
        # If we have a large modulus m_large, and we can remove it and add two smaller moduli m1, m2 such that 1/m1 + 1/m2 > 1/m_large, that's a win.
        # But finding such a pair is hard.
        
        # Simple heuristic: If we can't add any modulus, try to remove the largest modulus and see if we can add a smaller one that was previously blocked.
        if not result:
            # Find the largest modulus in current set
            if current_moduli:
                m_largest = max(current_moduli)
                # Try removing it
                trial_moduli = set(current_moduli)
                trial_moduli.remove(m_largest)
                # Try to add a smaller modulus
                # Which one? One that was blocked by m_largest?
                # Just try a random smaller modulus
                candidates = [m for m in all_moduli if m < m_largest and m not in trial_moduli]
                if candidates:
                    # Sort by value
                    candidates.sort(key=lambda m: 1.0/m, reverse=True)
                    m_try = candidates[0]
                    # Check if trial_moduli U {m_try} is feasible
                    # Use find_residues
                    trial_moduli.add(m_try)
                    new_assignment = find_residues(trial_moduli, max_restarts=10)
                    if new_assignment:
                        new_score = sum(1.0/m for m in trial_moduli)
                        if new_score > current_score:
                            current_moduli = trial_moduli
                            current_assignment = new_assignment
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_sol = [[m, r] for m, r in current_assignment.items()]
    
    # Final result
    if best_sol:
        return [[m, r] for m, r in best_sol]
    else:
        # Fallback
        return []
# EVOLVE-BLOCK-END
