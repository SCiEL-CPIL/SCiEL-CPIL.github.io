import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Range of moduli
    min_m, max_m = 20, 200
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute gcd for all pairs to speed up conflict check
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    n_mods = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # We will store the current state as a dictionary: modulus -> residue
    # And a set of active moduli for quick lookup
    # To allow fast "can we add m with residue r" check, we iterate over active moduli.
    
    def is_compatible(active, m_new, r_new):
        """Check if (m_new, r_new) is compatible with all in active."""
        for m_old, r_old in active.items():
            g = math.gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True

    def get_score(active):
        return sum(1.0 / m for m in active)

    # Initialize with a good starting point: Greedy Add
    active = {}
    # Sort moduli by 1/m descending (i.e., m ascending) to prioritize high value
    for m in sorted(moduli, key=lambda x: 1.0/x, reverse=True):
        # Try to find a compatible residue
        # We can just pick r=0 and check, or try a few
        # Actually, if we just add in order, we might fill up residues quickly.
        # Let's try to find ANY compatible residue.
        found = False
        for r in range(m):
            if is_compatible(active, m, r):
                active[m] = r
                found = True
                break
        if not found:
            # Try to see if we can swap? 
            # For now, just skip in initialization. Local search will handle swaps.
            pass
            
    best_active = dict(active)
    best_score = get_score(best_active)
    
    # Local Search / Simulated Annealing
    current_active = dict(active)
    current_score = best_score
    
    # Precompute gcds for active set? No, active changes.
    # But we can cache gcd(m1, m2) in a dict.
    gcd_cache = {}
    def get_gcd(m1, m2):
        if m1 > m2: m1, m2 = m2, m1
        key = (m1, m2)
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(m1, m2)
        return gcd_cache[key]

    def is_compatible_fast(active, m_new, r_new):
        for m_old, r_old in active.items():
            g = get_gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True

    # Re-evaluate initial state with fast check just in case, though logic is same.
    
    random.seed(42)
    
    while time.time() < deadline:
        # Randomly pick a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a new modulus
            # Pick a random modulus not in active
            inactive_mods = [m for m in moduli if m not in current_active]
            if inactive_mods:
                m_new = random.choice(inactive_mods)
                # Try a random residue
                r_new = random.randint(0, m_new - 1)
                if is_compatible_fast(current_active, m_new, r_new):
                    current_active[m_new] = r_new
                    new_score = current_score + 1.0 / m_new
                    if new_score > best_score:
                        best_score = new_score
                        best_active = dict(current_active)
                    current_score = new_score
                else:
                    # Try to find a valid residue by scanning? 
                    # Might be expensive if many active. 
                    # Just skip for now, or try a few randoms.
                    for _ in range(5):
                        r_try = random.randint(0, m_new - 1)
                        if is_compatible_fast(current_active, m_new, r_try):
                            current_active[m_new] = r_try
                            new_score = current_score + 1.0 / m_new
                            if new_score > best_score:
                                best_score = new_score
                                best_active = dict(current_active)
                            current_score = new_score
                            break
            
        elif move_type < 0.7:
            # Remove a modulus
            if current_active:
                m_old = random.choice(list(current_active.keys()))
                del current_active[m_old]
                current_score -= 1.0 / m_old
                # Note: removing always makes it valid, but lowers score.
                # We don't update best here as score dropped.
                
        elif move_type < 0.9:
            # Change residue of an existing modulus
            if current_active:
                m_old = random.choice(list(current_active.keys()))
                r_old = current_active[m_old]
                # Try a new random residue
                r_new = random.randint(0, m_old - 1)
                if r_new == r_old:
                    continue
                # Check compatibility with all OTHERS
                compatible = True
                for m_other, r_other in current_active.items():
                    if m_other == m_old:
                        continue
                    g = get_gcd(m_old, m_other)
                    if r_new % g == r_other % g:
                        compatible = False
                        break
                if compatible:
                    current_active[m_old] = r_new
                    # Score doesn't change
                    # But this might allow future additions
                    # We don't update best score directly, but state is improved for exploration
                else:
                    # Revert (do nothing)
                    pass
                    
        else:
            # Swap: Remove one, Add one
            if current_active and len(current_active) < n_mods:
                m_remove = random.choice(list(current_active.keys()))
                # Pick a random inactive modulus
                inactive_mods = [m for m in moduli if m not in current_active]
                if inactive_mods:
                    m_add = random.choice(inactive_mods)
                    # Try to find a residue for m_add that is compatible with 
                    # (current_active minus m_remove)
                    temp_active = {k: v for k, v in current_active.items() if k != m_remove}
                    found_r = None
                    # Try random residues
                    for _ in range(20):
                        r_try = random.randint(0, m_add - 1)
                        if is_compatible_fast(temp_active, m_add, r_try):
                            found_r = r_try
                            break
                    if found_r is not None:
                        # Perform swap
                        del current_active[m_remove]
                        current_active[m_add] = found_r
                        current_score -= 1.0 / m_remove
                        current_score += 1.0 / m_add
                        if current_score > best_score:
                            best_score = current_score
                            best_active = dict(current_active)
    
    # Format output
    # classes [[m, r], ...]
    result = [[m, r] for m, r in best_active.items()]
    return result
