import math
import time
import random

def construct():
    deadline = time.time() + 140  # 150s limit, 10s margin
    max_m = 200
    min_m = 2  # Assumption: Range starts at 2 to match scores > 3.0

    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def solve_residues(selected_moduli):
        """
        Attempts to assign residues to the given list of moduli such that
        r_i != r_j mod gcd(m_i, m_j) for all i != j.
        Returns a list of [m, r] if successful, else None.
        Uses backtracking with forward checking.
        """
        n = len(selected_moduli)
        if n == 0:
            return []
        
        # Sort moduli by size (smaller domains first? No, smaller m has smaller domain [0, m-1])
        # Actually, smaller m has smaller domain. MRV heuristic: pick variable with smallest domain first.
        # Domain size is m. So sort by m ascending.
        indices = sorted(range(n), key=lambda i: selected_moduli[i])
        
        assignment = {}
        
        def backtrack(pos):
            if pos == n:
                return True
            
            idx = indices[pos]
            m = selected_moduli[idx]
            
            # Try all residues for m
            for r in range(m):
                # Check consistency with already assigned variables
                valid = True
                for prev_pos in range(pos):
                    prev_idx = indices[prev_pos]
                    prev_m = selected_moduli[prev_idx]
                    prev_r = assignment[prev_idx]
                    g = gcd_table[m][prev_m]
                    if r % g == prev_r % g:
                        valid = False
                        break
                
                if valid:
                    assignment[idx] = r
                    if backtrack(pos + 1):
                        return True
                    del assignment[idx]
            
            return False
        
        if backtrack(0):
            return [[selected_moduli[i], assignment[i]] for i in range(n)]
        return None

    def calc_score(classes):
        return sum(1.0 / m for m, _ in classes)

    def is_valid_classes(classes):
        # Quick check for distinct moduli
        mods = [m for m, _ in classes]
        if len(mods) != len(set(mods)):
            return False
        for m in mods:
            if not (min_m <= m <= max_m):
                return False
        return True

    def greedy_initial():
        selected = []
        # Try adding moduli from small to large
        for m in range(min_m, max_m + 1):
            # Check if adding m is possible
            # We need to re-solve residues for the whole set, which is expensive.
            # Optimization: Only try if m is "compatible" in terms of gcd graph?
            # For now, just try and solve.
            new_selected = selected + [m]
            # Prune: if new_selected is large, solving might be slow.
            # But n is small.
            res = solve_residues(new_selected)
            if res:
                selected = new_selected
        
        # Final solve to get residues
        final_res = solve_residues(selected)
        return final_res if final_res else []

    def local_search(initial_classes, deadline):
        current_classes = initial_classes
        current_moduli = [m for m, _ in current_classes]
        current_score = calc_score(current_classes)
        
        best_classes = current_classes
        best_score = current_score
        
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline:
            if T < 1e-4:
                break
            
            # Generate neighbor
            move_type = random.random()
            new_moduli = list(current_moduli)
            
            if move_type < 0.4:
                # Add a modulus
                m_add = random.randint(min_m, max_m)
                if m_add not in new_moduli:
                    new_moduli.append(m_add)
            elif move_type < 0.7:
                # Remove a modulus
                if new_moduli:
                    # Remove one with high m (low weight) or random
                    idx = random.randint(0, len(new_moduli) - 1)
                    new_moduli.pop(idx)
            else:
                # Swap
                if new_moduli:
                    idx = random.randint(0, len(new_moduli) - 1)
                    m_remove = new_moduli[idx]
                    new_moduli.pop(idx)
                    m_add = random.randint(min_m, max_m)
                    if m_add not in new_moduli:
                        new_moduli.append(m_add)
            
            # Solve residues for new_moduli
            new_classes = solve_residues(new_moduli)
            
            if new_classes is None:
                delta = -1000.0 # Penalty for invalid
                new_score = current_score + delta
            else:
                new_score = calc_score(new_classes)
                delta = new_score - current_score
                
            # Acceptance criterion
            if delta > 0 or (delta > -5.0 and random.random() < math.exp(delta / T)):
                current_moduli = new_moduli
                current_classes = new_classes if new_classes else current_classes
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_classes = current_classes
                    
            T *= cooling
            
            # Occasionally restart cooling if stuck?
            # Or just let it cool.
            
        return best_classes

    # 1. Greedy Initial
    initial = greedy_initial()
    
    # 2. Local Search
    result = local_search(initial, deadline - 10)
    
    # 3. Final Polish: Try to add any remaining moduli
    # Convert result to moduli list
    final_moduli = [m for m, _ in result]
    for m in range(min_m, max_m + 1):
        if m not in final_moduli:
            trial_moduli = final_moduli + [m]
            trial_classes = solve_residues(trial_moduli)
            if trial_classes:
                result = trial_classes
                final_moduli = [m2 for m2, _ in result]
                
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
