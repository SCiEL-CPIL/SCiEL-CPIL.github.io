"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # Precompute gcd matrix for all pairs of moduli
    gcd_table = [[math.gcd(moduli[i], moduli[j]) for j in range(n_moduli)] for i in range(n_moduli)]
    
    def check_pairwise_disjoint(selected):
        """Check if all pairs in selected are disjoint. selected is list of (m_idx, r)."""
        n = len(selected)
        for i in range(n):
            mi, ri = selected[i]
            for j in range(i + 1, n):
                mj, rj = selected[j]
                g = gcd_table[mi][mj]
                if (ri - rj) % g == 0:
                    return False
        return True
    
    def score_solution(selected):
        """Calculate sum of 1/m for the selected classes. selected is list of (m_idx, r)."""
        return sum(1.0 / moduli[mi] for mi, r in selected)
    
    def find_valid_residue(m_idx, selected):
        """Find a residue r for modulus m_idx that is compatible with all in selected."""
        m = moduli[m_idx]
        for r in range(m):
            valid = True
            for mi_old, r_old in selected:
                g = gcd_table[m_idx][mi_old]
                if (r - r_old) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def find_best_residue(m_idx, selected):
        """Find a residue r for m_idx that is compatible and potentially 'better' (arbitrary tie-break)."""
        # For now, same as find_valid_residue, but could be extended to prefer residues that leave more room for others
        return find_valid_residue(m_idx, selected)

    def greedy_initial(order=None):
        """Greedy construction: try moduli in given order, pick first valid residue."""
        if order is None:
            order = list(range(n_moduli))
        selected = []
        for mi in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(mi, selected)
            if r is not None:
                selected.append((mi, r))
        return selected

    def polish_solution(selected, max_time):
        """
        Local search polish: 
        1. Try to add any missing modulus.
        2. Try to swap residues to allow additions.
        3. Try to swap moduli (remove one, add another) if it improves score.
        """
        current_score = score_solution(selected)
        best_score = current_score
        best_selected = list(selected)
        selected_set = set(mi for mi, r in selected)
        
        # 1. Greedy fill
        changed = True
        while changed:
            changed = False
            for mi in range(n_moduli):
                if mi not in selected_set:
                    r = find_valid_residue(mi, selected)
                    if r is not None:
                        selected.append((mi, r))
                        selected_set.add(mi)
                        current_score += 1.0 / moduli[mi]
                        changed = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
        
        # 2. Local Search: Try swaps
        # We limit the number of swap attempts based on time
        attempts = 0
        while time.time() < max_time:
            attempts += 1
            if attempts % 1000 == 0 and time.time() > max_time:
                break
                
            # Pick a random element to remove
            if not selected:
                break
            idx_remove = random.randrange(len(selected))
            mi_remove, r_remove = selected[idx_remove]
            
            # Try to add a different modulus
            candidates = [mi for mi in range(n_moduli) if mi not in selected_set or mi == mi_remove]
            if not candidates:
                continue
            mi_add = random.choice(candidates)
            
            if mi_add == mi_remove:
                continue # No change in modulus set
            
            # Check if adding mi_add is possible if we remove mi_remove
            # We need to find a residue for mi_add that is compatible with all *other* selected items
            temp_selected = [s for i, s in enumerate(selected) if i != idx_remove]
            r_add = find_valid_residue(mi_add, temp_selected)
            
            if r_add is not None:
                # Calculate new score
                new_score = current_score - (1.0 / moduli[mi_remove]) + (1.0 / moduli[mi_add])
                if new_score > current_score + 1e-12:
                    # Accept swap
                    selected[idx_remove] = (mi_add, r_add)
                    current_score = new_score
                    selected_set.discard(mi_remove)
                    selected_set.add(mi_add)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    
        return best_selected, best_score

    def simulated_annealing(initial_solution, max_time):
        """
        Simulated annealing with moves:
        1. Add a new modulus
        2. Remove a modulus
        3. Reassign residue
        4. Swap modulus (remove one, add another) - NEW
        """
        selected = list(initial_solution)
        current_score = score_solution(selected)
        best_selected = list(selected)
        best_score = current_score
        
        T = 0.05  # Initial temperature
        cooling = 0.999
        min_T = 0.001
        
        selected_set = set(mi for mi, r in selected)
        
        while time.time() < max_time and T > min_T:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Move 1: Add a new modulus
                candidates = [mi for mi in range(n_moduli) if mi not in selected_set]
                if not candidates:
                    T *= cooling
                    continue
                mi = random.choice(candidates)
                r = find_valid_residue(mi, selected)
                if r is None:
                    T *= cooling
                    continue
                new_selected = selected + [(mi, r)]
                new_score = current_score + 1.0 / moduli[mi]
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(-delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.add(mi)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                        
            elif move_type < 0.5:
                # Move 2: Remove a modulus
                if not selected:
                    T *= cooling
                    continue
                # Weight removal toward larger moduli (less valuable)
                weights = [1.0 / moduli[mi] for mi, r in selected]
                total_w = sum(weights)
                if total_w == 0:
                    T *= cooling
                    continue
                r_rand = random.random() * total_w
                cum = 0
                remove_idx = len(selected) - 1
                for i, w in enumerate(weights):
                    cum += w
                    if r_rand <= cum:
                        remove_idx = i
                        break
                mi_removed, r_removed = selected[remove_idx]
                new_selected = selected[:remove_idx] + selected[remove_idx + 1:]
                new_score = current_score - 1.0 / moduli[mi_removed]
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(-delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.discard(mi_removed)
                    
            elif move_type < 0.7:
                # Move 3: Reassign residue
                if not selected:
                    T *= cooling
                    continue
                idx = random.randrange(len(selected))
                mi, old_r = selected[idx]
                m = moduli[mi]
                new_r = old_r
                attempts = 0
                while new_r == old_r and attempts < 50:
                    new_r = random.randrange(m)
                    attempts += 1
                if new_r == old_r:
                    T *= cooling
                    continue
                valid = True
                for j, (mi_old, r_old) in enumerate(selected):
                    if j == idx:
                        continue
                    g = gcd_table[mi][mi_old]
                    if (new_r - r_old) % g == 0:
                        valid = False
                        break
                if not valid:
                    T *= cooling
                    continue
                # Accept reassignment to explore state space
                # It doesn't change score, but might enable future additions
                selected[idx] = (mi, new_r)
                
            else:
                # Move 4: Swap Modulus (Remove one, Add another)
                if not selected:
                    T *= cooling
                    continue
                idx_remove = random.randrange(len(selected))
                mi_remove, r_remove = selected[idx_remove]
                
                candidates = [mi for mi in range(n_moduli) if mi not in selected_set or mi == mi_remove]
                if not candidates:
                    T *= cooling
                    continue
                mi_add = random.choice(candidates)
                
                if mi_add == mi_remove:
                    T *= cooling
                    continue
                
                # Check if mi_add fits if mi_remove is gone
                temp_selected = [s for i, s in enumerate(selected) if i != idx_remove]
                r_add = find_valid_residue(mi_add, temp_selected)
                
                if r_add is None:
                    T *= cooling
                    continue
                
                new_score = current_score - (1.0 / moduli[mi_remove]) + (1.0 / moduli[mi_add])
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(-delta / T):
                    selected[idx_remove] = (mi_add, r_add)
                    current_score = new_score
                    selected_set.discard(mi_remove)
                    selected_set.add(mi_add)
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)

            T *= cooling
            
            # Periodic greedy fill
            if random.random() < 0.05:
                for mi in range(n_moduli):
                    if mi not in selected_set:
                        r = find_valid_residue(mi, selected)
                        if r is not None:
                            selected.append((mi, r))
                            selected_set.add(mi)
                            current_score += 1.0 / moduli[mi]
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = list(selected)
        
        return best_selected, best_score
    
    # Phase 1: Multiple Greedy Initializations
    best_solution = []
    best_score = 0.0
    
    # Ascending greedy (standard)
    sol_asc = greedy_initial(list(range(n_moduli)))
    sc_asc = score_solution(sol_asc)
    if sc_asc > best_score:
        best_score = sc_asc
        best_solution = sol_asc
        
    # Descending greedy
    sol_desc = greedy_initial(list(range(n_moduli - 1, -1, -1)))
    sc_desc = score_solution(sol_desc)
    if sc_desc > best_score:
        best_score = sc_desc
        best_solution = sol_desc
        
    # Randomized greedy
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        shuffled = list(range(n_moduli))
        random.shuffle(shuffled)
        sol_rand = greedy_initial(shuffled)
        sc_rand = score_solution(sol_rand)
        if sc_rand > best_score:
            best_score = sc_rand
            best_solution = sol_rand
            
    # Phase 2: Simulated Annealing
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_solution, sa_score = simulated_annealing(best_solution, deadline - 5)
        if sa_score > best_score:
            best_score = sa_score
            best_solution = sa_solution
            
    # Phase 3: Polish
    if time.time() < deadline - 2:
        polished_solution, polished_score = polish_solution(best_solution, deadline - 1)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished_solution
            
    # Convert to required format [[m, r], ...]
    result = [[moduli[mi], r] for mi, r in best_solution]
    return result
# EVOLVE-BLOCK-END
