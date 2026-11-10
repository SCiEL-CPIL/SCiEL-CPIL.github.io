import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    # Precompute GCD table for O(1) access
    # gcd_table[m1][m2] = gcd(m1, m2)
    # Since m is up to 128, a 129x129 list is small.
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_valid_residue(r, m, selected):
        """
        Check if residue r for modulus m is disjoint from all (r_j, m_j) in selected.
        Condition: (r - r_j) % gcd(m, m_j) != 0 for all j.
        """
        for r_j, m_j in selected:
            g = gcd_table[m][m_j]
            if (r - r_j) % g == 0:
                return False
        return True

    def find_valid_residue(m, selected):
        """
        Find any valid residue for modulus m given current selected classes.
        Returns None if no valid residue exists.
        """
        # Try residues in random order to avoid bias
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_valid_residue(r, m, selected):
                return r
        return None

    def try_add_modulus(m_new, selected, unused):
        """
        Try to add m_new to selected. 
        If successful, updates selected and unused.
        Returns True if added.
        """
        r_new = find_valid_residue(m_new, selected)
        if r_new is not None:
            selected.append([r_new, m_new])
            unused.discard(m_new)
            return True
        return False

    best_solution = []
    
    # Phase 1: Greedy Initialization with multiple heuristics
    # Heuristic 1: Descending number of divisors (more constraints first?)
    def num_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
            i += 1
        return count + 1

    # Heuristic 2: Ascending number of prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    heuristics = [
        lambda m: num_divisors(m), # Desc
        lambda m: -num_prime_factors(m), # Asc (negative for descending sort if we use reverse, or just sort asc)
        lambda m: m, # Asc value
        lambda m: -m, # Desc value
        lambda m: random.random() # Random
    ]

    for h in heuristics:
        if time.time() > deadline:
            break
            
        # Sort moduli based on heuristic
        # For random, we just shuffle
        if h is heuristics[-1]:
            current_moduli = all_moduli[:]
            random.shuffle(current_moduli)
        else:
            current_moduli = sorted(all_moduli, key=h, reverse=(h is heuristics[0] or h is heuristics[3]))
            # Note: num_divisors desc, num_prime_factors asc (so -npf desc? No, we want simple first? 
            # Let's just try both asc and desc for each metric to be safe, or pick one.
            # Actually, sorting by key with reverse flag is tricky if key is negative.
            # Let's just generate a few specific orders.
            
    # Let's do a cleaner loop for initializations
    orders = []
    # 1. Descending divisor count
    orders.append(sorted(all_moduli, key=num_divisors, reverse=True))
    # 2. Ascending divisor count
    orders.append(sorted(all_moduli, key=num_divisors))
    # 3. Descending value
    orders.append(sorted(all_moduli, reverse=True))
    # 4. Ascending value
    orders.append(sorted(all_moduli))
    # 5. Random
    orders.append(all_moduli[:])
    random.shuffle(orders[-1])
    # 6. Another Random
    orders.append(all_moduli[:])
    random.shuffle(orders[-1])

    for order in orders:
        if time.time() > deadline:
            break
        
        selected = []
        unused = set(all_moduli)
        
        for m in order:
            if m not in unused:
                continue
            if try_add_modulus(m, selected, unused):
                pass
            # If it fails, we just skip it for now in this greedy pass
        
        if len(selected) > len(best_solution):
            best_solution = [pair[:] for pair in selected]
            
    # Phase 2: Local Search Improvement
    # Start with the best solution found
    current_selected = [pair[:] for pair in best_solution]
    current_unused = set(all_moduli) - set(m for _, m in current_selected)
    
    # If no solution found yet, start with one
    if not current_selected:
        current_selected = [[0, 2]]
        current_unused = set(all_moduli) - {2}

    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Try to add any unused modulus
        if current_unused:
            # Shuffle unused to pick random target
            unused_list = list(current_unused)
            random.shuffle(unused_list)
            
            for m_new in unused_list:
                if try_add_modulus(m_new, current_selected, current_unused):
                    improved = True
                    # If we added one, restart the loop to try adding more
                    break
        
        # If we couldn't add a modulus directly, try to perturb an existing one
        # to allow adding a specific new modulus
        if not improved and current_unused and current_selected:
            # Pick a random target to add
            m_target = random.choice(list(current_unused))
            
            # Pick a random existing modulus to perturb
            idx_old = random.randint(0, len(current_selected) - 1)
            r_old, m_old = current_selected[idx_old]
            
            # Try to find a new residue r_new for m_old such that m_target becomes addable
            # We iterate through possible r_new for m_old
            # Optimization: Only try r_new that are valid against all OTHER selected moduli
            others = [current_selected[i] for i in range(len(current_selected)) if i != idx_old]
            
            found_new_config = False
            residues_to_try = list(range(m_old))
            random.shuffle(residues_to_try)
            
            for r_new in residues_to_try:
                # Check if r_new is valid against others
                if not is_valid_residue(r_new, m_old, others):
                    continue
                
                # Temporarily set m_old to r_new
                current_selected[idx_old][0] = r_new
                
                # Check if m_target can now be added
                r_target = find_valid_residue(m_target, current_selected)
                if r_target is not None:
                    # Success!
                    current_selected.append([r_target, m_target])
                    current_unused.discard(m_target)
                    improved = True
                    found_new_config = True
                    break
                else:
                    # Revert? Or keep? 
                    # Keeping might help for future steps, but it changes the state arbitrarily.
                    # Reverting keeps the search focused on "adding one".
                    # However, if we revert, we try the next r_new.
                    # If we keep, we might have lost the ability to add something else.
                    # Let's revert to be safe and systematic.
                    current_selected[idx_old][0] = r_old
            
            # If we didn't find a config that allows adding m_target, we revert to original state
            # (which was already done in the loop if we reverted, or if we broke out with success)
            # If we didn't break out with success, current_selected[idx_old][0] is r_old (either reverted or never changed)
            
    # Final verification and formatting
    # Ensure we return the best solution
    if len(current_selected) > len(best_solution):
        best_solution = current_selected

    if not best_solution:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_solution]
