import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 115s to be safe
    deadline = time.time() + 115
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
            
    moduli = list(range(min_m, max_m + 1))
    
    # Helper to check validity of (m, r) against a set of (m, r) pairs
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(m, selected):
        valid_rs = []
        for r in range(m):
            if is_valid(m, r, selected):
                valid_rs.append(r)
        return valid_rs

    def calculate_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_construct(randomize=False):
        selected = []
        # Process moduli in ascending order (highest weight first)
        for m in moduli:
            valid_rs = get_valid_residues(m, selected)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    # Heuristic: pick r that minimizes conflicts with next 5 moduli
                    # This is a simple look-ahead
                    best_r = valid_rs[0]
                    min_conflicts = float('inf')
                    for r in valid_rs:
                        conflicts = 0
                        # Check next 5 moduli
                        next_moduli = [mm for mm in moduli if mm > m][:5]
                        for mm in next_moduli:
                            g = gcd_table[m][mm]
                            # Count how many residues of mm are blocked by (m, r)
                            # A residue r_mm is blocked if r_mm % g == r % g
                            # Number of such residues is mm // g
                            # So adding (m, r) blocks (mm // g) residues of mm
                            conflicts += mm // g
                        if conflicts < min_conflicts:
                            min_conflicts = conflicts
                            best_r = r
                    r = best_r
                selected.append((m, r))
        return selected

    def local_search(initial_selected):
        selected = [tuple(p) for p in initial_selected]
        current_score = calculate_score(selected)
        selected_set = set(selected)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any unselected modulus
            # We want to add the smallest possible m (highest weight)
            for m in moduli:
                if m in [s[0] for s in selected]:
                    continue
                valid_rs = get_valid_residues(m, selected)
                if valid_rs:
                    # Adding m improves score
                    new_selected = selected + [(m, valid_rs[0])]
                    new_score = current_score + 1.0 / m
                    if new_score > current_score:
                        selected = new_selected
                        current_score = new_score
                        improved = True
                        break # Restart loop as set changed
            
            if improved:
                continue

            # 2. Try to swap a selected modulus with a smaller unselected one
            # Sort selected by m descending to try replacing large m with small m
            selected_sorted = sorted(selected, key=lambda x: x[0], reverse=True)
            
            swap_found = False
            for m_old, r_old in selected_sorted:
                # Try to find a smaller m_new not in selected
                for m_new in moduli:
                    if m_new >= m_old:
                        break
                    if m_new in [s[0] for s in selected]:
                        continue
                    
                    # Check if m_new can be added if m_old is removed
                    temp_selected = [s for s in selected if s[0] != m_old]
                    valid_rs = get_valid_residues(m_new, temp_selected)
                    if valid_rs:
                        # Calculate new score
                        new_score = current_score - 1.0 / m_old + 1.0 / m_new
                        if new_score > current_score:
                            # Perform swap
                            selected = temp_selected + [(m_new, valid_rs[0])]
                            current_score = new_score
                            improved = True
                            swap_found = True
                            break
                if swap_found:
                    break
            
            if improved:
                continue
                
            # 3. Try to change residue of existing moduli to enable additions?
            # This is complex. Instead, try random restarts of local search if stuck.
            # Or, try a "kick": randomly change a residue or remove an item to escape local opt.
            if time.time() < deadline - 10:
                # Random kick: remove a random item and try to re-add or swap
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    removed = selected[idx]
                    selected = [s for s in selected if s != removed]
                    current_score = calculate_score(selected)
                    improved = True # Force loop to try adding again

        return selected, current_score

    best_solution = []
    best_score = 0.0
    
    # Run multiple iterations of Greedy + Local Search
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Alternate between deterministic and randomized greedy
        is_rand = (i % 2 == 1)
        initial = greedy_construct(randomize=is_rand)
        
        # Run local search
        sol, score = local_search(initial)
        
        if score > best_score:
            best_score = score
            best_solution = sol
            
        # Early exit if we found a very good solution? 
        # Theoretical max is sum 1/m for all m if no conflicts.
        # Sum 1/m for 20..200 is approx ln(200) - ln(19) + gamma ~ 5.3 - 2.9 + 0.5 ~ 2.9?
        # Wait, sum 1/n from 20 to 200 is H_200 - H_19.
        # H_200 ~ ln(200) + 0.577 = 5.3 + 0.577 = 5.87
        # H_19 ~ ln(19) + 0.577 = 2.94 + 0.577 = 3.52
        # Diff ~ 2.35.
        # But conflicts reduce this.
        # If we get > 2.2, it's likely good.

    # Convert to required format
    if best_solution:
        return [[m, r] for m, r in best_solution]
    
    # Fallback: simple greedy
    initial = greedy_construct(randomize=False)
    return [[m, r] for m, r in initial]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
