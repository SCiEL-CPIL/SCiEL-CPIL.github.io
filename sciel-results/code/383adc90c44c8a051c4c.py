"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def score_solution(selected):
        return sum(1.0 / m for m, r in selected)
    
    def greedy_initial(seed_choice=0):
        """Greedy: pick smallest moduli, choose residue based on seed_choice strategy."""
        selected = []
        for m in moduli:
            if time.time() > deadline:
                break
            if seed_choice == 0:
                candidates = list(range(m))
            elif seed_choice == 1:
                candidates = list(range(m))
                random.shuffle(candidates)
            else:
                candidates = list(range(m))
                candidates.sort(key=lambda r: r % m)  # same as natural order
            
            for r in candidates:
                if is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def try_add_modulus(selected, m):
        """Try to add modulus m by finding a valid residue."""
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None
    
    def local_search(selected, max_iters=50000):
        """
        Local search: try to improve by:
        1. Adding missing moduli
        2. Changing residues of existing moduli to free space for new ones
        3. Swapping: remove one, add another
        """
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        selected_set = set(m for m, r in selected)
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add any missing modulus
            for m in moduli:
                if m in selected_set:
                    continue
                if time.time() > deadline:
                    return selected
                r = try_add_modulus(selected, m)
                if r is not None:
                    selected.append([m, r])
                    selected_set.add(m)
                    current_score = score_solution(selected)
                    improved = True
                    break
            
            # 2. If can't add, try changing residues to free up space
            if not improved:
                # Find which missing moduli are blocked
                missing = [m for m in moduli if m not in selected_set]
                if not missing:
                    break
                
                # For each missing modulus (smallest first), try to free space
                for m_add in missing:
                    if time.time() > deadline:
                        return selected
                    
                    # Find which selected moduli block m_add
                    blockers = []
                    for idx, (m_old, r_old) in enumerate(selected):
                        g = math.gcd(m_add, m_old)
                        # Check if ALL residues for m_add are blocked by this m_old
                        # Actually, we need to check if there's NO valid r for m_add
                        # given the current configuration
                        # Simpler: check if m_add is blocked
                        pass
                    
                    # Try changing each selected modulus's residue
                    for idx in range(len(selected)):
                        m_old, r_old = selected[idx]
                        # Try changing this residue
                        for r_new in range(m_old):
                            if r_new == r_old:
                                continue
                            # Check if new residue is compatible with all others
                            compatible = True
                            for jdx, (m_j, r_j) in enumerate(selected):
                                if jdx == idx:
                                    continue
                                g = math.gcd(m_old, m_j)
                                if (r_new - r_j) % g == 0:
                                    compatible = False
                                    break
                            if not compatible:
                                continue
                            
                            # Try this change
                            old_r = selected[idx][1]
                            selected[idx][1] = r_new
                            
                            # Now try to add m_add
                            r_add = try_add_modulus(selected, m_add)
                            if r_add is not None:
                                selected.append([m_add, r_add])
                                selected_set.add(m_add)
                                new_score = score_solution(selected)
                                if new_score > current_score:
                                    current_score = new_score
                                    improved = True
                                    break
                                else:
                                    # Revert
                                    selected.pop()
                                    selected_set.discard(m_add)
                            selected[idx][1] = old_r
                        
                        if improved:
                            break
                    
                    if improved:
                        break
        
        return selected
    
    # Run initial greedy
    best_solution = greedy_initial(0)
    best_score = score_solution(best_solution)
    
    # Try local search
    if time.time() < deadline - 5.0:
        improved_solution = local_search(best_solution)
        improved_score = score_solution(improved_solution)
        if improved_score > best_score:
            best_solution = improved_solution
            best_score = improved_score
    
    # Try random restarts with different residue orderings
    if time.time() < deadline - 10.0:
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            sol = greedy_initial(1)  # random residue order
            sc = score_solution(sol)
            if sc > best_score:
                best_solution = sol
                best_score = sc
            
            # Local search on this
            if time.time() < deadline - 5.0:
                sol2 = local_search(sol, max_iters=10000)
                sc2 = score_solution(sol2)
                if sc2 > best_score:
                    best_solution = sol2
                    best_score = sc2
    
    return best_solution
# EVOLVE-BLOCK-END
