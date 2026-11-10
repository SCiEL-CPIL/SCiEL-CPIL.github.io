import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    MAX_M = 128
    candidates = list(range(2, MAX_M + 1))
    
    best_solution = []
    best_count = 0
    
    # Precompute gcds and valid residues for pairs? 
    # It's better to check on the fly.
    
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
        
    def try_add(r, m, current_set):
        for (ri, mi) in current_set:
            if not is_disjoint(r, m, ri, mi):
                return False
        return True
        
    def find_valid_residue(m, current_set):
        # Try all residues for m
        for r in range(m):
            if try_add(r, m, current_set):
                return r
        return None
        
    def greedy_construct(order):
        current_set = []
        for m in order:
            r = find_valid_residue(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set
        
    def local_search(current_set, time_limit):
        # Convert to list of [r, m] for consistency
        sol = [[r, m] for r, m in current_set]
        selected_moduli = {m for r, m in current_set}
        
        # Local search moves:
        # 1. Try to add a new modulus
        # 2. Try to swap a modulus
        # 3. Try to change residue to allow future additions? (Hard to predict)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any unselected modulus
            unselected = [m for m in candidates if m not in selected_moduli]
            random.shuffle(unselected)
            for m in unselected:
                r = find_valid_residue(m, current_set)
                if r is not None:
                    current_set.append((r, m))
                    selected_moduli.add(m)
                    improved = True
                    break # Restart search loop after improvement
                    
            if improved:
                continue
                
            # If no addition possible, try swapping
            # Pick a random selected modulus to remove, and a random unselected to add
            if current_set:
                idx_remove = random.randint(0, len(current_set) - 1)
                r_old, m_old = current_set[idx_remove]
                current_set.pop(idx_remove)
                selected_moduli.remove(m_old)
                
                # Try to add a random unselected one
                unselected = [m for m in candidates if m not in selected_moduli]
                random.shuffle(unselected)
                added = False
                for m_new in unselected:
                    r_new = find_valid_residue(m_new, current_set)
                    if r_new is not None:
                        current_set.append((r_new, m_new))
                        selected_moduli.add(m_new)
                        improved = True
                        added = True
                        break
                
                if not added:
                    # Revert swap if it didn't lead to improvement (count didn't increase, but we might have changed the set)
                    # Actually, if we removed one and didn't add one, count decreased. Revert.
                    current_set.append((r_old, m_old))
                    selected_moduli.add(m_old)
                    
        return current_set

    # Run multiple random restarts
    while time.time() < deadline:
        random_order = candidates[:]
        random.shuffle(random_order)
        
        # Greedy phase
        current_set = greedy_construct(random_order)
        
        # Local search phase
        # Give local search a small time slice
        local_deadline = min(deadline, time.time() + 5.0)
        current_set = local_search(current_set, local_deadline)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_solution = [[r, m] for r, m in current_set]
            
    # Return in the required format: list of [residue, modulus]
    return best_solution

