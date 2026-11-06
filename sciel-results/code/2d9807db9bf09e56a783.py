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
    moduli = list(range(min_m, max_m + 1))
    n_mod = len(moduli)
    
    # Precompute gcd table indexed by modulus value
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # For each modulus m, precompute the set of residue classes mod g
    # that are "compatible" with each other? No, we need to check pairs.
    
    def is_valid(m, r, selected):
        """Check if (m, r) is valid against a list of selected (m, r) pairs."""
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
    
    def calc_score(selected):
        score = Fraction(0)
        for m, r in selected:
            score += Fraction(1, m)
        return score
    
    def greedy_construct(strategy='first', rng_seed=None):
        """Greedy: for each m in increasing order, pick a valid residue."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = rng
        
        selected = []
        for m in range(min_m, max_m + 1):
            if strategy == 'first':
                for r in range(m):
                    if is_valid(m, r, selected):
                        selected.append([m, r])
                        break
            elif strategy == 'random':
                valid_residues = [r for r in range(m) if is_valid(m, r, selected)]
                if valid_residues:
                    r = local_rng.choice(valid_residues)
                    selected.append([m, r])
            elif strategy == 'min_conflict':
                # Choose residue that minimizes future conflicts
                best_r = None
                best_conflicts = float('inf')
                for r in range(m):
                    if not is_valid(m, r, selected):
                        continue
                    conflicts = 0
                    for future_m in range(m + 1, max_m + 1):
                        g = gcd_table[m][future_m]
                        # This residue r blocks r % g mod g for future_m
                        # The number of blocked residues in [0, future_m) is future_m // g
                        blocked = future_m // g
                        conflicts += blocked
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_r = r
                if best_r is not None:
                    selected.append([m, best_r])
        return selected
    
    def simulated_annealing(initial_solution, time_budget):
        """Simulated Annealing to improve the solution."""
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        
        start_time = time.time()
        
        # Precompute set of selected moduli for quick lookup
        selected_mods = set(m for m, _ in current_sol)
        
        while time.time() - start_time < time_budget and T > min_T:
            # Generate a neighbor
            move_type = rng.random()
            
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False
            
            if move_type < 0.5:
                # Add a modulus
                # Choose a random unselected modulus, biased towards small m
                # We can just pick a random m from 20..200 and check if it's already there
                m_add = rng.randint(min_m, max_m)
                if m_add in selected_mods:
                    continue
                
                # Try a random valid residue
                # To be faster, we can try a few random r's
                found_r = False
                for _ in range(10):
                    r_add = rng.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += Fraction(1, m_add)
                        valid_move = True
                        found_r = True
                        break
                
                if not found_r:
                    continue
                    
            elif move_type < 0.7:
                # Remove a modulus
                if not new_sol:
                    continue
                idx = rng.randint(0, len(new_sol) - 1)
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= Fraction(1, m_rem)
                valid_move = True
                
            else:
                # Change residue
                if not new_sol:
                    continue
                idx = rng.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                
                # Try a few random new residues
                found_r = False
                for _ in range(10):
                    new_r = rng.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    # Check if new_r is valid for m_chg against all OTHERS
                    # We must check against all (m2, r2) where m2 != m_chg
                    temp_sol = [x for i, x in enumerate(new_sol) if i != idx]
                    if is_valid(m_chg, new_r, temp_sol):
                        # Update score? Score doesn't change for residue change
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        found_r = True
                        break
                
                if not found_r:
                    continue
            
            if not valid_move:
                continue
                
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta.numerator / (delta.denominator * T)):
                current_sol = new_sol
                current_score = new_score
                
                # Update selected_mods
                selected_mods = set(m for m, _ in current_sol)
                
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]
            
            T *= cooling
            
        return best_sol, best_score
    
    # Phase 1: Multiple greedy starts with different strategies
    best_selected = None
    best_score = Fraction(0)
    
    strategies = ['first', 'random', 'min_conflict']
    for strategy in strategies:
        selected = greedy_construct(strategy)
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = selected
    
    # More random restarts
    start_time = time.time()
    for attempt in range(20):
        if time.time() - start_time > 10:
            break
        selected = greedy_construct('random', rng_seed=attempt)
        score = calc_score(selected)
        if score > best_score:
            best_score = score
            best_selected = selected
    
    # Phase 2: Simulated Annealing on best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_selected, best_score = simulated_annealing(best_selected, min(remaining_time - 2, 40))
    
    # Phase 3: More local search iterations with different starting points
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try local search from a few other good solutions
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            selected = greedy_construct('random', rng_seed=100 + attempt)
            score = calc_score(selected)
            if score > best_score - 0.5:  # Only if reasonably close
                budget = min(deadline - time.time() - 3, 20)
                if budget > 2:
                    new_selected, new_score = simulated_annealing(selected, budget)
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = new_selected
    
    # Convert to output format
    result = [[m, r] for m, r in best_selected]
    return result
# EVOLVE-BLOCK-END
