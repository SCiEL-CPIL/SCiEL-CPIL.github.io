import math
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_pairwise_disjoint(moduli, residues):
    """Check if the set of (m, r) pairs is pairwise disjoint."""
    n = len(moduli)
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            if residues[i] % g == residues[j] % g:
                return False
    return True

def _find_residues(moduli, max_iters=5000):
    """
    Try to find residues for a given list of moduli such that they are pairwise disjoint.
    Uses randomized greedy assignment with backtracking-like retries.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # Sort by modulus descending to assign more constrained ones first? 
    # Actually, assigning residues for larger moduli might be easier or harder?
    # Let's just randomize the order of assignment in each attempt.
    
    for _ in range(max_iters):
        residues = [-1] * n
        order = list(range(n))
        random.shuffle(order)
        
        success = True
        for idx in order:
            m = moduli[idx]
            # Find a residue r in [0, m) that doesn't conflict with already assigned ones
            valid_r = []
            for r in range(m):
                conflict = False
                for prev_idx in order[:order.index(idx)]:
                    if residues[prev_idx] != -1:
                        g = math.gcd(m, moduli[prev_idx])
                        if r % g == residues[prev_idx] % g:
                            conflict = True
                            break
                if not conflict:
                    valid_r.append(r)
            
            if not valid_r:
                success = False
                break
            residues[idx] = random.choice(valid_r)
        
        if success:
            return residues
    
    return None

def _calc_score(moduli, residues):
    """Calculate the exact score as a Fraction."""
    score = Fraction(0, 1)
    for m in moduli:
        score += Fraction(1, m)
    return score

def _greedy_moduli_selection(candidate_moduli, max_size=30):
    """
    Greedily select moduli to maximize reciprocal sum, checking feasibility.
    This is a heuristic to get a good starting point.
    """
    selected_moduli = []
    # Sort candidates by value 1/m descending (i.e., m ascending)
    candidates = sorted(set(candidate_moduli), key=lambda x: x)
    
    for m in candidates:
        if len(selected_moduli) >= max_size:
            break
        trial_moduli = selected_moduli + [m]
        # Quick check: are there enough residues?
        # A necessary condition: for any subset, the constraints must be satisfiable.
        # A simple necessary condition: for any pair, gcd > 1? No, gcd=1 is fine (always disjoint).
        # But if gcd is large, it's restrictive.
        # Let's just try to find residues.
        res = _find_residues(trial_moduli, max_iters=200)
        if res is not None:
            selected_moduli = trial_moduli
    
    return selected_moduli

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Candidate moduli in [20, 200]
    all_moduli = list(range(20, 201))
    
    # Initialize with a greedy selection
    current_moduli = _greedy_moduli_selection(all_moduli, max_size=25)
    if not current_moduli:
        # Fallback to a simple valid set
        current_moduli = [20, 30, 40, 50, 60]
    
    current_residues = _find_residues(current_moduli, max_iters=5000)
    if current_residues is None:
        # Fallback
        return [[20, 0], [30, 1], [40, 2], [50, 3], [60, 4]]
        
    current_score = _calc_score(current_moduli, current_residues)
    best_moduli = current_moduli.copy()
    best_residues = current_residues.copy()
    best_score = current_score

    # Local Search: Swap, Add, Remove
    # We will perform a random walk with hill climbing.
    while time.time() < deadline:
        move_type = random.choice(['swap', 'add', 'remove', 'swap'])
        
        if move_type == 'swap':
            if len(current_moduli) < 2:
                continue
            # Pick a random index in current_moduli to remove, and a random candidate to add
            idx_to_remove = random.randint(0, len(current_moduli) - 1)
            m_removed = current_moduli[idx_to_remove]
            
            # Pick a candidate not in current_moduli
            candidates_to_add = [m for m in all_moduli if m not in current_moduli]
            if not candidates_to_add:
                continue
            m_added = random.choice(candidates_to_add)
            
            new_moduli = current_moduli.copy()
            new_moduli.remove(m_removed)
            new_moduli.append(m_added)
            
            new_res = _find_residues(new_moduli, max_iters=500)
            if new_res is not None:
                new_score = _calc_score(new_moduli, new_res)
                # Accept if better, or with some probability if worse (simulated annealing lite)
                if new_score > current_score:
                    current_moduli = new_moduli
                    current_residues = new_res
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_moduli = current_moduli.copy()
                        best_residues = current_residues.copy()
                elif random.random() < 0.1:
                    current_moduli = new_moduli
                    current_residues = new_res
                    current_score = new_score

        elif move_type == 'add':
            candidates_to_add = [m for m in all_moduli if m not in current_moduli]
            if not candidates_to_add:
                continue
            m_added = random.choice(candidates_to_add)
            new_moduli = current_moduli + [m_added]
            
            new_res = _find_residues(new_moduli, max_iters=500)
            if new_res is not None:
                new_score = _calc_score(new_moduli, new_res)
                if new_score > current_score:
                    current_moduli = new_moduli
                    current_residues = new_res
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_moduli = current_moduli.copy()
                        best_residues = current_residues.copy()

        elif move_type == 'remove':
            if len(current_moduli) <= 2:
                continue
            idx_to_remove = random.randint(0, len(current_moduli) - 1)
            m_removed = current_moduli[idx_to_remove]
            new_moduli = current_moduli.copy()
            new_moduli.remove(m_removed)
            
            # Removing always preserves disjointness, but we need to re-find residues?
            # Actually, if we remove one, the existing residues are still valid.
            # But we can try to find a "better" residue assignment for the remaining set?
            # The score depends only on moduli, not residues. So removal just lowers score.
            # However, it might allow adding a different modulus later.
            # So we just accept removal to explore the space, but it lowers current score.
            # We only accept removal if we are in a "stuck" phase or to diversify.
            # For simplicity, let's only remove if it leads to a better future? Hard to know.
            # Let's just do it with low probability to allow restructuring.
            if random.random() < 0.05:
                current_moduli = new_moduli
                current_residues = _find_residues(current_moduli, max_iters=200)
                if current_residues is None:
                    # Should not happen
                    continue
                current_score = _calc_score(current_moduli, current_residues)

        # Occasionally re-optimize residues for the current moduli set?
        # Not necessary since score depends only on moduli.
        
    # Return the best found
    return [[m, r] for m, r in zip(best_moduli, best_residues)]
# EVOLVE-BLOCK-END
