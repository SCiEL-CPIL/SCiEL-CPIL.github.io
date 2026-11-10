import math
import time
import random

def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin for 60s limit
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    
    # Precompute GCDs
    gcds = [[0] * (max_modulus + 1) for _ in range(max_modulus + 1)]
    for i in range(2, max_modulus + 1):
        for j in range(i, max_modulus + 1):
            g = math.gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g

    def get_valid_residues(m, S):
        """
        Returns a list of valid residues for modulus m given existing set S.
        S is a list of [residue, modulus] pairs.
        Uses marking array for efficiency.
        """
        valid = [True] * m
        for r_prime, m_prime in S:
            g = gcds[m][m_prime]
            if g == 1:
                # If gcd is 1, they are never disjoint, so this pair is invalid anyway.
                # But the problem statement implies we only select disjoint classes.
                # If we are building a set of disjoint classes, we should never have gcd=1 pairs.
                # If we do, the whole set is invalid.
                # However, the verifier checks pairwise disjointness.
                # So we must ensure all pairs in S have gcd > 1.
                # If gcd=1, no residue r works.
                return []
            target = r_prime % g
            # Mark r as invalid if r % g == target
            for r in range(target, m, g):
                valid[r] = False
        
        return [r for r, v in enumerate(valid) if v]

    def is_valid_set(S):
        """
        Checks if all pairs in S are disjoint.
        """
        n = len(S)
        for i in range(n):
            r1, m1 = S[i]
            for j in range(i + 1, n):
                r2, m2 = S[j]
                g = gcds[m1][m2]
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True

    def greedy_build(order):
        current_set = []
        for m in order:
            # Check if m can be added
            # We need to ensure gcd(m, m') > 1 for all existing m'
            # If any gcd is 1, m cannot be added to this set.
            compatible_modulus = True
            for _, m_prime in current_set:
                if gcds[m][m_prime] == 1:
                    compatible_modulus = False
                    break
            
            if not compatible_modulus:
                continue

            valid_residues = get_valid_residues(m, current_set)
            if valid_residues:
                # Pick a random valid residue to diversify
                current_set.append([random.choice(valid_residues), m])
        return current_set

    def local_search_improve(S, max_iterations=1000):
        """
        Improves S using add-one and remove-one-add-two moves.
        """
        current = [pair[:] for pair in S]
        current_size = len(current)
        
        # Precompute used moduli for faster lookup
        used_moduli = {m for _, m in current}
        
        for _ in range(max_iterations):
            if time.time() - start_time > max_time - 2.0:
                break
            
            improved = False
            
            # 1. Try to add any missing modulus
            available_moduli = [m for m in moduli if m not in used_moduli]
            random.shuffle(available_moduli)
            
            for m in available_moduli:
                # Quick check: gcd with all existing
                compatible = True
                for _, m_prime in current:
                    if gcds[m][m_prime] == 1:
                        compatible = False
                        break
                if not compatible:
                    continue

                valid_residues = get_valid_residues(m, current)
                if valid_residues:
                    r = random.choice(valid_residues)
                    current.append([r, m])
                    used_moduli.add(m)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try remove-one-add-two
            # Pick a random element to remove
            for _attempt in range(5):  # Reduced attempts per iteration for speed
                if time.time() - start_time > max_time - 2.0:
                    break
                
                if not current:
                    break
                    
                i = random.randint(0, len(current) - 1)
                removed_pair = current[i]
                removed_m = removed_pair[1]
                
                # Create temp set
                temp_set = current[:i] + current[i+1:]
                temp_used = {m for _, m in temp_set}
                
                # Try to add two moduli
                candidates1 = []
                available_for_add = [m for m in moduli if m not in temp_used]
                
                # Filter available_for_add by gcd compatibility with temp_set
                for m1 in available_for_add:
                    compatible = True
                    for _, m_prime in temp_set:
                        if gcds[m1][m_prime] == 1:
                            compatible = False
                            break
                    if compatible:
                        valid_r1 = get_valid_residues(m1, temp_set)
                        if valid_r1:
                            candidates1.append((m1, valid_r1))
                
                if len(candidates1) < 2:
                    continue

                random.shuffle(candidates1)
                
                found_improvement = False
                for m1, valid_r1 in candidates1[:10]: # Limit candidates
                    r1 = random.choice(valid_r1)
                    temp_set2 = temp_set + [[r1, m1]]
                    temp_used2 = temp_used | {m1}
                    
                    # Try to add a second modulus m2
                    # m2 must be in available_for_add and m2 != m1
                    # And compatible with temp_set2
                    for m2 in available_for_add:
                        if m2 == m1:
                            continue
                        # Check gcd compatibility with temp_set2
                        compatible = True
                        for _, m_prime in temp_set2:
                            if gcds[m2][m_prime] == 1:
                                compatible = False
                                break
                        if not compatible:
                            continue

                        valid_r2 = get_valid_residues(m2, temp_set2)
                        if valid_r2:
                            r2 = random.choice(valid_r2)
                            # Success: we removed 1 and added 2
                            current = temp_set2 + [[r2, m2]]
                            used_moduli = temp_used2 | {m2}
                            current_size += 1
                            improved = True
                            found_improvement = True
                            break
                    
                    if found_improvement:
                        break
                
                if found_improvement:
                    break
            
            if not improved:
                break
        
        return current

    # Initial greedy with fixed order
    best_solution = greedy_build(moduli)
    best_size = len(best_solution)
    
    # Local search on initial
    improved = local_search_improve(best_solution, max_iterations=500)
    if len(improved) > best_size:
        best_size = len(improved)
        best_solution = improved

    # Randomized restarts
    restarts = 0
    while time.time() - start_time < max_time - 5.0:
        restarts += 1
        random.shuffle(moduli)
        current_set = greedy_build(moduli)
        
        # Local search
        improved_set = local_search_improve(current_set, max_iterations=500)
        
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_solution = [pair[:] for pair in improved_set]
            
        # If we have a good solution, try to improve it further with more iterations
        if best_size > 20 and time.time() - start_time < max_time - 10.0:
            further_improved = local_search_improve(best_solution, max_iterations=1000)
            if len(further_improved) > best_size:
                best_size = len(further_improved)
                best_solution = [pair[:] for pair in further_improved]

    # Final verification step (just in case)
    if not is_valid_set(best_solution):
        # Fallback to a trivial valid set
        best_solution = [[0, 2]]

    return best_solution
