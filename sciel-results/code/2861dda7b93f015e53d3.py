import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    best_selected = []
    best_score = 0.0

    def is_compatible(m_new, r_new, selected_list):
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def run_random_greedy():
        selected = []
        # We process in increasing order of m to prioritize high value terms
        for m in moduli:
            # Find all valid residues
            valid_residues = []
            for r in range(m):
                if is_compatible(m, r, selected):
                    valid_residues.append(r)
            
            if valid_residues:
                # Pick a random valid residue
                r = random.choice(valid_residues)
                selected.append([m, r])
            # If no valid residue, we skip this modulus.
            # This is a limitation: we don't backtrack.
        
        score = sum(Fraction(1, m) for m, r in selected)
        return selected, float(score)

    def run_simulated_annealing():
        # Start with a random greedy solution
        selected, score = run_random_greedy()
        best_selected_sa = list(selected)
        best_score_sa = score
        
        # SA parameters
        T = 1.0  # Initial temperature
        cooling_rate = 0.995
        min_temperature = 0.01
        max_steps = 5000
        
        step = 0
        while T > min_temperature and time.time() < deadline - 10.0:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a new modulus with random residue
                available_moduli = [m for m in moduli if m not in [s[0] for s in selected]]
                if available_moduli:
                    m_new = random.choice(available_moduli)
                    valid_residues = [r for r in range(m_new) if is_compatible(m_new, r, selected)]
                    if valid_residues:
                        r_new = random.choice(valid_residues)
                        selected.append([m_new, r_new])
                        new_score = sum(Fraction(1, m) for m, r in selected)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            score = new_score
                            if score > best_score_sa:
                                best_score_sa = score
                                best_selected_sa = list(selected)
                        else:
                            selected.pop()
                            step += 1
                            T *= cooling_rate
                            continue
            elif move_type < 0.6:
                # Change residue of an existing modulus
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    m_old, r_old = selected[idx]
                    valid_residues = []
                    for r in range(m_old):
                        if r == r_old:
                            continue
                        # Check compatibility with all other selected
                        compatible = True
                        for j, (m_j, r_j) in enumerate(selected):
                            if j == idx:
                                continue
                            g = math.gcd(m_old, m_j)
                            if (r - r_j) % g == 0:
                                compatible = False
                                break
                        if compatible:
                            valid_residues.append(r)
                    if valid_residues:
                        r_new = random.choice(valid_residues)
                        selected[idx] = [m_old, r_new]
                        new_score = sum(Fraction(1, m) for m, r in selected)
                        delta = new_score - score
                        if delta > 0 or random.random() < math.exp(delta / T):
                            score = new_score
                            if score > best_score_sa:
                                best_score_sa = score
                                best_selected_sa = list(selected)
                        else:
                            selected[idx] = [m_old, r_old]
            else:
                # Remove a modulus
                if len(selected) > 1:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected.pop(idx)
                    new_score = sum(Fraction(1, m) for m, r in selected)
                    delta = new_score - score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        score = new_score
                        if score > best_score_sa:
                            best_score_sa = score
                            best_selected_sa = list(selected)
                    else:
                        selected.insert(idx, removed)
            
            step += 1
            T *= cooling_rate
        
        return best_selected_sa, best_score_sa

    # Run multiple restarts with both methods
    restarts = 0
    while time.time() < deadline - 5.0:
        # Alternate between greedy and SA
        if restarts % 3 == 0:
            selected, score = run_simulated_annealing()
        else:
            selected, score = run_random_greedy()
        
        if score > best_score:
            best_score = score
            best_selected = selected
        restarts += 1
        
    # If we ran very few restarts, maybe we should try a deterministic greedy first?
    # The deterministic greedy is just one specific random choice (e.g., always pick smallest).
    # We can add a deterministic run to ensure we have a valid baseline.
    if restarts == 0:
        selected, score = run_random_greedy()
        best_selected = selected
        best_score = score

    return best_selected
