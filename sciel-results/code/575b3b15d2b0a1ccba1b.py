import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 150s limit
    
    # Moduli range
    min_m = 20
    max_m = 200
    all_moduli = list(range(min_m, max_m + 1))
    
    # Precompute GCDs
    # gcd_table[m1][m2]
    # We can use a dict of dicts or a 2D list.
    # 181x181 is small.
    gcd_table = {}
    for m1 in all_moduli:
        gcd_table[m1] = {}
        for m2 in all_moduli:
            gcd_table[m1][m2] = math.gcd(m1, m2)
            
    # Helper: Check if (m, r) is disjoint from all in current_classes
    # current_classes is a list of (m, r)
    def is_disjoint(m, r, current_classes):
        for m_c, r_c in current_classes:
            if m == m_c:
                # Should not happen if m is new, or if we are changing residue of m itself
                # But in "Change" move, we exclude self.
                continue
            g = gcd_table[m][m_c]
            if (r - r_c) % g != 0:
                continue
            else:
                return False
        return True

    # Helper: Find a valid residue for m given current_classes
    # Returns a random valid r, or None if no valid r exists
    def find_valid_residue(m, current_classes):
        # Shuffled residues for randomization
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_disjoint(m, r, current_classes):
                return r
        return None

    # Initialize with greedy
    current_classes = []
    for m in all_moduli:
        r = find_valid_residue(m, current_classes)
        if r is not None:
            current_classes.append((m, r))
            
    # Calculate score
    def calc_score(classes):
        return sum(1.0 / m for m, r in classes)
        
    current_score = calc_score(current_classes)
    best_classes = [c for c in current_classes]
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # We will run until deadline
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher T? Or just stop?
            # Let's just stop if T is too low, or restart.
            # Given the time limit, simple cooling is fine.
            break
            
        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Add
            # Pick a random m not in current_classes
            current_moduli_set = {m for m, r in current_classes}
            available_moduli = [m for m in all_moduli if m not in current_moduli_set]
            if not available_moduli:
                continue
            m_add = random.choice(available_moduli)
            r_add = find_valid_residue(m_add, current_classes)
            if r_add is None:
                continue # No valid residue, cannot add
            
            # Calculate delta
            delta = 1.0 / m_add
            # Always accept if delta > 0?
            # In SA, we accept if delta > 0 or with prob exp(delta/T)
            # Here delta is always positive for Add.
            # So we always accept valid Adds.
            current_classes.append((m_add, r_add))
            current_score += delta
            
        elif move_type < 0.6:
            # Remove
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_rem, r_rem = current_classes[idx]
            delta = -1.0 / m_rem
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                current_classes.pop(idx)
                current_score += delta
            # Else reject (do nothing)
            
        else:
            # Change Residue
            if not current_classes:
                continue
            idx = random.randint(0, len(current_classes) - 1)
            m_chg, r_old = current_classes[idx]
            # Pick new residue
            r_new = random.randint(0, m_chg - 1)
            if r_new == r_old:
                continue
            
            # Check validity of new residue
            # We need to check against all OTHER classes
            # Temporarily remove self? Or just skip self in check.
            valid = True
            for i, (m_c, r_c) in enumerate(current_classes):
                if i == idx:
                    continue
                g = gcd_table[m_chg][m_c]
                if (r_new - r_c) % g == 0:
                    valid = False
                    break
            
            if not valid:
                continue # Reject move implicitly by not changing state
            
            # Delta is 0 for change residue (same m)
            # So we always accept valid changes?
            # Yes, because score doesn't change, but it might enable future adds.
            # This is a "neutral" move.
            current_classes[idx] = (m_chg, r_new)
            
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_classes = [c for c in current_classes]
            
        # Cool down
        T *= cooling_rate
        
    # Return best
    # Format: [[m, r], ...]
    return [[m, r] for m, r in best_classes]
