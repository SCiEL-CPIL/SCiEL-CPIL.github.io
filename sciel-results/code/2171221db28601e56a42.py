"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin for evaluator's 150s limit
    
    best_score = Fraction(0, 1)
    best_classes = []
    
    def find_valid_residue(m, moduli, residues, randomize=False):
        """
        Find a valid residue r for m given existing moduli and residues.
        Returns r or None.
        If randomize is True, returns a random valid residue.
        """
        valid_residues = []
        for r in range(m):
            ok = True
            for i in range(len(moduli)):
                m2 = moduli[i]
                g = gcd(m, m2)
                if (r - residues[i]) % g == 0:
                    ok = False
                    break
            if ok:
                if randomize:
                    valid_residues.append(r)
                else:
                    return r
        if randomize and valid_residues:
            return random.choice(valid_residues)
        return None

    def can_add(m, moduli, residues):
        """Check if m can be added to the set with existing residues. Returns valid r or None."""
        return find_valid_residue(m, moduli, residues)

    def score_of(moduli_list):
        return sum(Fraction(1, m) for m in moduli_list)

    all_moduli = list(range(20, 201))
    
    # Strategy 1: Multiple Greedy Attempts with different residue choices
    best_greedy_classes = []
    best_greedy_score = Fraction(0, 1)
    
    for attempt in range(30):
        if time.time() > deadline - 40:
            break
            
        current_moduli = []
        current_residues = []
        
        use_random_residue = (attempt % 3 == 1)
        
        for m in all_moduli:
            if time.time() > deadline - 40:
                break
            
            r = find_valid_residue(m, current_moduli, current_residues, randomize=use_random_residue)
                
            if r is not None:
                current_moduli.append(m)
                current_residues.append(r)
        
        if current_moduli:
            s = score_of(current_moduli)
            if s > best_greedy_score:
                best_greedy_score = s
                best_greedy_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
    
    if best_greedy_classes:
        best_score = best_greedy_score
        best_classes = best_greedy_classes
    
    # Strategy 2: Enhanced Local Search
    if best_classes and time.time() < deadline - 5:
        current_moduli = [c[0] for c in best_classes]
        current_residues = [c[1] for c in best_classes]
        current_score = best_score
        
        selected_set = set(current_moduli)
        available_moduli = [m for m in all_moduli if m not in selected_set]
        
        def try_swap(out_idx, m_in, current_moduli, current_residues, current_score):
            """Try to swap current_moduli[out_idx] with m_in. Returns (new_moduli, new_residues, new_score) or None."""
            if m_in >= current_moduli[out_idx]:
                return None  # Only accept if m_in < m_out (higher score)
            
            trial_moduli = []
            trial_residues = []
            for i in range(len(current_moduli)):
                if i != out_idx:
                    trial_moduli.append(current_moduli[i])
                    trial_residues.append(current_residues[i])
            
            r_in = find_valid_residue(m_in, trial_moduli, trial_residues)
            
            if r_in is not None:
                delta = Fraction(1, m_in) - Fraction(1, current_moduli[out_idx])
                if delta > 0:
                    new_moduli = current_moduli[:]
                    new_residues = current_residues[:]
                    new_moduli[out_idx] = m_in
                    new_residues[out_idx] = r_in
                    new_score = current_score + delta
                    return (new_moduli, new_residues, new_score)
            return None
        
        def try_add(m_in, current_moduli, current_residues, current_score):
            """Try to add m_in to the current set. Returns (new_moduli, new_residues, new_score) or None."""
            if m_in in selected_set:
                return None
            r_in = find_valid_residue(m_in, current_moduli, current_residues)
            if r_in is not None:
                new_moduli = current_moduli + [m_in]
                new_residues = current_residues + [r_in]
                new_score = current_score + Fraction(1, m_in)
                return (new_moduli, new_residues, new_score)
            return None
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Pass 1: Try all swaps systematically
            for out_idx in range(len(current_moduli)):
                if time.time() > deadline - 5:
                    break
                m_out = current_moduli[out_idx]
                # Try all available moduli smaller than m_out
                for m_in in available_moduli:
                    if m_in >= m_out:
                        continue
                    result = try_swap(out_idx, m_in, current_moduli, current_residues, current_score)
                    if result:
                        current_moduli, current_residues, current_score = result
                        selected_set.remove(m_out)
                        selected_set.add(m_in)
                        available_moduli.remove(m_in)
                        available_moduli.append(m_out)
                        improved = True
                        best_score = current_score
                        best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
                        break  # Restart swap search with new state
            
            # Pass 2: Try to add any available modulus
            for m_in in sorted(available_moduli):  # Try smallest first
                if time.time() > deadline - 5:
                    break
                result = try_add(m_in, current_moduli, current_residues, current_score)
                if result:
                    current_moduli, current_residues, current_score = result
                    selected_set.add(m_in)
                    available_moduli.remove(m_in)
                    improved = True
                    best_score = current_score
                    best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
            
            # Pass 3: Random swaps for diversification
            if time.time() < deadline - 5 and not improved:
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    if not current_moduli or not available_moduli:
                        break
                    out_idx = random.randint(0, len(current_moduli) - 1)
                    m_in = random.choice(available_moduli)
                    result = try_swap(out_idx, m_in, current_moduli, current_residues, current_score)
                    if result:
                        m_out = current_moduli[out_idx]
                        current_moduli, current_residues, current_score = result
                        selected_set.remove(m_out)
                        selected_set.add(m_in)
                        available_moduli.remove(m_in)
                        available_moduli.append(m_out)
                        improved = True
                        best_score = current_score
                        best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
    
    # Ensure we have at least the seed result
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        best_score = sum(Fraction(1, 20*j) for j in range(1, 11))
    
    return best_classes
# EVOLVE-BLOCK-END
