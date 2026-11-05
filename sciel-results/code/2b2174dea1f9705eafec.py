"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    best_score = 0.0
    best_classes = []
    
    # Number of restarts to attempt
    # We estimate each run takes ~0.01-0.05s, so we can do many.
    # Let's do as many as we can.
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            # Check if r_new == r_old (mod g)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m_new, selected_list):
        """Return list of valid residues for m_new given selected_list."""
        valids = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected_list):
                valids.append(r)
        return valids

    def run_greedy_randomized():
        selected = []
        score = 0.0
        for m in moduli:
            valids = get_valid_residues(m, selected)
            if valids:
                # Randomly pick a valid residue
                r = random.choice(valids)
                selected.append([m, r])
                score += 1.0 / m
        return selected, score

    # Run multiple randomized trials
    while time.time() < deadline:
        classes, score = run_greedy_randomized()
        if score > best_score:
            best_score = score
            best_classes = classes
        
        # Optional: If we found a very high score, we might reduce randomness
        # or break early if we hit a theoretical bound (which we don't know exactly).
        # Theoretical max is sum(1/m) for m=20..200 which is ~1.5. 
        # Wait, 1/20 + ... + 1/200 is the harmonic sum H_200 - H_19.
        # H_n ~ ln(n) + gamma.
        # ln(200) + 0.577 - (ln(19) + 0.577) = ln(200/19) = ln(10.5) ~ 2.35.
        # So max possible score is ~2.35.
        # If we get close to 2.35, we can stop.
        if best_score > 2.30:
            break

    return best_classes

# EVOLVE-BLOCK-END
