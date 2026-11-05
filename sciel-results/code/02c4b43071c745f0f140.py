"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    all_moduli_set = set(moduli)
    
    def gcd(a, b):
        return math.gcd(a, b)
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def score_solution(selected):
        """Calculate sum of 1/m for the selected classes."""
        return sum(1.0 / m for m, r in selected)
    
    def find_valid_residue(m, selected_list):
        """Find the first valid residue for modulus m given selected_list."""
        for r in range(m):
            if is_compatible(m, r, selected_list):
                return r
        return -1
    
    def greedy_initial(order=None):
        """Greedy construction: pick moduli in given order, first valid residue."""
        if order is None:
            order = moduli
        selected = []
        for m in order:
            if time_left() < 5.0:
                break
            r = find_valid_residue(m, selected)
            if r >= 0:
                selected.append([m, r])
        return selected
    
    def local_search(selected, max_iter=500):
        """
        Local search with:
        1. Add moves: try to add any missing modulus
        2. 1-swap moves: remove one, try to add one (or change residue)
        3. Residue optimization: for each selected, try all residues
        """
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        improved = True
        iteration = 0
        
        while improved and time_left() > 10.0 and iteration < max_iter:
            improved = False
            iteration += 1
            selected_set = set(m for m, r in selected)
            
            # 1. Try to add any missing modulus
            for m in sorted(moduli):
                if m in selected_set:
                    continue
                r = find_valid_residue(m, selected)
                if r >= 0:
                    new_selected = selected + [[m, r]]
                    new_score = score_solution(new_selected)
                    if new_score > current_score + 1e-15:
                        selected = new_selected
                        current_score = new_score
                        improved = True
                        selected_set.add(m)
            
            # 2. Residue optimization: for each selected, try to find a better residue
            #    that allows adding more moduli
            for i in range(len(selected)):
                if time_left() < 5.0:
                    break
                m_i = selected[i][0]
                best_r = selected[i][1]
                # Try all residues for this modulus
                for r in range(m_i):
                    if r == best_r:
                        continue
                    # Check compatibility with others
                    compatible = True
                    for j in range(len(selected)):
                        if j == i:
                            continue
                        g = math.gcd(m_i, selected[j][0])
                        if (r - selected[j][1]) % g == 0:
                            compatible = False
                            break
                    if compatible:
                        # Check if this residue allows adding more moduli
                        selected[i][1] = r
                        added_count = 0
                        for m_add in sorted(moduli):
                            if m_add in selected_set:
                                continue
                            r_add = find_valid_residue(m_add, selected)
                            if r_add >= 0:
                                added_count += 1
                        if added_count > 0:
                            best_r = r
                        else:
                            selected[i][1] = best_r
                        break  # Only try first valid alternative for speed
            
            # 3. 1-swap: remove a large modulus, try to add a smaller one
            # Sort by value (1/m) descending - we want to keep small moduli
            selected_sorted = sorted(enumerate(selected), key=lambda x: 1.0/x[1][0], reverse=True)
            for rem_idx, rem_m, rem_r in [(idx, sel[0], sel[1]) for idx, sel in selected_sorted]:
                if time_left() < 5.0:
                    break
                if rem_m < 20:
                    break
                # Try removing this and adding a smaller modulus
                for m_add in sorted(moduli):
                    if m_add >= rem_m:
                        break
                    if m_add in selected_set:
                        continue
                    # Remove rem_m
                    new_selected = [x for j, x in enumerate(selected) if j != rem_idx]
                    r_add = find_valid_residue(m_add, new_selected)
                    if r_add >= 0:
                        new_selected.append([m_add, r_add])
                        new_score = score_solution(new_selected)
                        if new_score > current_score + 1e-15:
                            selected = new_selected
                            current_score = new_score
                            improved = True
                            selected_set = set(m for m, r in selected)
                            break
            
            # 4. 2-swap: remove two, try to add two (only if time permits)
            if time_left() > 30.0:
                n = len(selected)
                for i in range(n):
                    if time_left() < 10.0:
                        break
                    for j in range(i+1, n):
                        if time_left() < 10.0:
                            break
                        m_i, r_i = selected[i]
                        m_j, r_j = selected[j]
                        # Try removing both
                        new_selected = [x for k, x in enumerate(selected) if k != i and k != j]
                        # Try adding two new moduli
                        added_any = False
                        for m_add1 in sorted(moduli):
                            if time_left() < 10.0:
                                break
                            if m_add1 in selected_set or m_add1 == m_i or m_add1 == m_j:
                                continue
                            r_add1 = find_valid_residue(m_add1, new_selected)
                            if r_add1 >= 0:
                                temp_selected = new_selected + [[m_add1, r_add1]]
                                for m_add2 in sorted(moduli):
                                    if time_left() < 10.0:
                                        break
                                    if m_add2 in selected_set or m_add2 == m_i or m_add2 == m_j or m_add2 == m_add1:
                                        continue
                                    r_add2 = find_valid_residue(m_add2, temp_selected)
                                    if r_add2 >= 0:
                                        final_selected = temp_selected + [[m_add2, r_add2]]
                                        new_score = score_solution(final_selected)
                                        if new_score > current_score + 1e-15:
                                            selected = final_selected
                                            current_score = new_score
                                            improved = True
                                            added_any = True
                                            selected_set = set(m for m, r in selected)
                                            break
                                if added_any:
                                    break
                        if added_any:
                            break
                    if improved:
                        break
        
        return selected
    
    def simulated_annealing(selected, steps=5000):
        """
        Simulated annealing to escape local optima.
        Moves:
        1. Add a random missing modulus
        2. Remove a random selected modulus
        3. Change residue of a random selected modulus
        4. Swap: remove one, add another
        """
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        best_selected = [list(x) for x in selected]
        best_score = current_score
        
        T = 0.1  # Initial temperature
        cooling = 0.9995  # Cooling rate
        selected_set = set(m for m, r in selected)
        
        for step in range(steps):
            if time_left() < 5.0:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.25 and len(selected) < 181:
                # Add a random missing modulus
                missing = [m for m in moduli if m not in selected_set]
                if not missing:
                    continue
                m_add = random.choice(missing)
                r_add = find_valid_residue(m_add, selected)
                if r_add < 0:
                    continue
                new_selected = selected + [[m_add, r_add]]
                new_score = score_solution(new_selected)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.add(m_add)
            elif move_type < 0.45 and len(selected) > 5:
                # Remove a random selected modulus
                idx = random.randint(0, len(selected)-1)
                m_rem = selected[idx][0]
                new_selected = selected[:idx] + selected[idx+1:]
                new_score = score_solution(new_selected)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.discard(m_rem)
            elif move_type < 0.70:
                # Change residue of a random selected modulus
                if not selected:
                    continue
                idx = random.randint(0, len(selected)-1)
                m_i = selected[idx][0]
                old_r = selected[idx][1]
                new_r = random.randint(0, m_i - 1)
                if new_r == old_r:
                    continue
                # Check compatibility
                compatible = True
                for j in range(len(selected)):
                    if j == idx:
                        continue
                    g = math.gcd(m_i, selected[j][0])
                    if (new_r - selected[j][1]) % g == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                selected[idx][1] = new_r
                # Score doesn't change with residue, but compatibility might allow future adds
                # For now, just accept if compatible
            elif move_type < 0.90:
                # Swap: remove one, add another
                if not selected:
                    continue
                idx = random.randint(0, len(selected)-1)
                m_rem = selected[idx][0]
                missing = [m for m in moduli if m not in selected_set]
                if not missing:
                    continue
                m_add = random.choice(missing)
                new_selected = selected[:idx] + selected[idx+1:]
                r_add = find_valid_residue(m_add, new_selected)
                if r_add < 0:
                    continue
                new_selected.append([m_add, r_add])
                new_score = score_solution(new_selected)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.discard(m_rem)
                    selected_set.add(m_add)
            else:
                # 2-swap: remove two, add two
                if len(selected) < 2:
                    continue
                idx1, idx2 = random.sample(range(len(selected)), 2)
                m_rem1, m_rem2 = selected[idx1][0], selected[idx2][0]
                new_selected = [x for k, x in enumerate(selected) if k != idx1 and k != idx2]
                missing = [m for m in moduli if m not in selected_set and m != m_rem1 and m != m_rem2]
                if len(missing) < 2:
                    continue
                m_add1, m_add2 = random.sample(missing, 2)
                r_add1 = find_valid_residue(m_add1, new_selected)
                if r_add1 < 0:
                    continue
                temp_selected = new_selected + [[m_add1, r_add1]]
                r_add2 = find_valid_residue(m_add2, temp_selected)
                if r_add2 < 0:
                    continue
                new_selected = temp_selected + [[m_add2, r_add2]]
                new_score = score_solution(new_selected)
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    selected = new_selected
                    current_score = new_score
                    selected_set.discard(m_rem1)
                    selected_set.discard(m_rem2)
                    selected_set.add(m_add1)
                    selected_set.add(m_add2)
            
            T *= cooling
            
            if current_score > best_score + 1e-15:
                best_score = current_score
                best_selected = [list(x) for x in selected]
        
        return best_selected, best_score
    
    # Phase 1: Multiple greedy starts with different orderings
    best_solution = []
    best_score = 0.0
    
    # Standard greedy (ascending moduli)
    sol1 = greedy_initial()
    s1 = score_solution(sol1)
    if s1 > best_score:
        best_solution = sol1
        best_score = s1
    
    # Greedy with prime moduli first
    primes = [m for m in moduli if all(m % i != 0 for i in range(2, int(math.sqrt(m)) + 1))]
    non_primes = [m for m in moduli if m not in primes]
    order2 = primes + non_primes
    sol2 = greedy_initial(order2)
    s2 = score_solution(sol2)
    if s2 > best_score:
        best_solution = sol2
        best_score = s2
    
    # Greedy with descending moduli (to see if it helps)
    order3 = list(reversed(moduli))
    sol3 = greedy_initial(order3)
    s3 = score_solution(sol3)
    if s3 > best_score:
        best_solution = sol3
        best_score = s3
    
    # Phase 2: Local search on best solution
    if time_left() > 20.0:
        improved = local_search(best_solution, max_iter=200)
        s_improved = score_solution(improved)
        if s_improved > best_score:
            best_solution = improved
            best_score = s_improved
    
    # Phase 3: Simulated annealing
    if time_left() > 15.0:
        sa_solution, sa_score = simulated_annealing(best_solution, steps=10000)
        if sa_score > best_score:
            best_solution = sa_solution
            best_score = sa_score
    
    # Phase 4: More local search after SA
    if time_left() > 10.0:
        improved2 = local_search(best_solution, max_iter=100)
        s_improved2 = score_solution(improved2)
        if s_improved2 > best_score:
            best_solution = improved2
            best_score = s_improved2
    
    # Phase 5: Another SA round if time permits
    if time_left() > 15.0:
        sa_solution2, sa_score2 = simulated_annealing(best_solution, steps=5000)
        if sa_score2 > best_score:
            best_solution = sa_solution2
            best_score = sa_score2
    
    # Final local search
    if time_left() > 5.0:
        final = local_search(best_solution, max_iter=50)
        s_final = score_solution(final)
        if s_final > best_score:
            best_solution = final
            best_score = s_final
    
    return best_solution
# EVOLVE-BLOCK-END
