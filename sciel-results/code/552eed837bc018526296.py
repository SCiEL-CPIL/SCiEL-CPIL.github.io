import math
import time
import random

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    
    Key Insight:
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff (r1 % gcd(m1, m2)) != (r2 % gcd(m1, m2)).
    
    Strategy:
    1. Use constraint-based backtracking with smart variable ordering.
    2. Order moduli by their "constraint density" (number of shared factors with other moduli).
    3. Use forward checking to prune invalid residues early.
    4. Combine with randomized restarts to escape local optima.
    """
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Precompute gcd for all pairs of moduli
    all_moduli = list(range(2, 129))
    n_moduli = len(all_moduli)
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i + 1, n_moduli):
            g = math.gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each modulus, precompute which residues are compatible with which residue of another modulus
    # compat[i][j][r_i] = set of residues r_j such that r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)
    # This is expensive, so we'll compute on the fly but cache
    
    def get_compatible_residues(m_i, r_i, m_j, r_j):
        """Check if residue r_i of modulus m_i is disjoint from residue r_j of modulus m_j."""
        g = math.gcd(m_i, m_j)
        return (r_i % g) != (r_j % g)
    
    def find_valid_residue(m, assigned, moduli_list):
        """
        Find a residue r in [0, m) for modulus m that is disjoint from all assigned moduli.
        assigned: list of (modulus, residue) pairs already assigned
        moduli_list: the full list of moduli being considered
        """
        for r in range(m):
            valid = True
            for m_assigned, r_assigned in assigned:
                g = math.gcd(m, m_assigned)
                if (r % g) == (r_assigned % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def solve_greedy(moduli_order, start_from=None):
        """
        Greedily assign residues to moduli in the given order.
        Returns list of (modulus, residue) pairs.
        """
        assigned = list(start_from) if start_from else []
        assigned_moduli = set(m for m, r in assigned)
        
        for m in moduli_order:
            if m in assigned_moduli:
                continue
            r = find_valid_residue(m, assigned, moduli_order)
            if r is not None:
                assigned.append((m, r))
                assigned_moduli.add(m)
        
        return assigned
    
    def solve_backtracking(moduli_order, max_depth=None):
        """
        Backtracking search with forward checking.
        moduli_order: list of moduli to consider, in order
        Returns best solution found.
        """
        best = []
        
        def backtrack(idx, assigned):
            nonlocal best
            if time.time() > deadline:
                return
            if idx == len(moduli_order):
                if len(assigned) > len(best):
                    best = list(assigned)
                return
            
            m = moduli_order[idx]
            # Try each possible residue
            for r in range(m):
                # Check if r is valid with all assigned
                valid = True
                for m_a, r_a in assigned:
                    g = math.gcd(m, m_a)
                    if (r % g) == (r_a % g):
                        valid = False
                        break
                if valid:
                    assigned.append((m, r))
                    backtrack(idx + 1, assigned)
                    assigned.pop()
                    if time.time() > deadline:
                        return
            
            # Also try skipping this modulus
            if len(assigned) > len(best):
                best = list(assigned)
            backtrack(idx + 1, assigned)
        
        backtrack(0, [])
        return best
    
    # Strategy: Use a combination of approaches
    # 1. First, get a good baseline with greedy
    # 2. Then try to improve with local search (swap residues, add/remove moduli)
    
    # Compute a "difficulty" score for each modulus: number of moduli that share a factor > 1
    difficulty = {}
    for m in all_moduli:
        count = 0
        for m2 in all_moduli:
            if m != m2 and math.gcd(m, m2) > 1:
                count += 1
        difficulty[m] = count
    
    # Sort moduli by difficulty (most constrained first)
    moduli_by_difficulty = sorted(all_moduli, key=lambda m: difficulty[m], reverse=True)
    moduli_by_difficulty_asc = sorted(all_moduli, key=lambda m: difficulty[m])
    
    # Get baseline solutions
    best_solution = []
    
    # Try greedy with different orderings
    orderings = [
        moduli_by_difficulty,
        moduli_by_difficulty_asc,
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
    ]
    
    for order in orderings:
        sol = solve_greedy(order)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Also try starting from subsets
    for start_size in range(0, min(5, len(all_moduli))):
        random_subset = random.sample(all_moduli, start_size)
        sol = solve_greedy(moduli_by_difficulty, start_from=random_subset)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    best_size = len(best_solution)
    
    # Local search: try to improve by swapping residues or adding/removing moduli
    def local_search(solution, max_iterations=1000):
        """
        Local search to improve the solution.
        """
        current = list(solution)
        current_size = len(current)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try adding a modulus
            current_moduli = set(m for m, r in current)
            available = [m for m in all_moduli if m not in current_moduli]
            
            if available:
                m_add = random.choice(available)
                r_add = find_valid_residue(m_add, current, all_moduli)
                if r_add is not None:
                    new_sol = current + [(m_add, r_add)]
                    if len(new_sol) > current_size:
                        current = new_sol
                        current_size = len(current)
                        continue
            
            # Try swapping a residue
            if current:
                idx_swap = random.randint(0, len(current) - 1)
                m_swap, r_swap = current[idx_swap]
                # Try a different residue for this modulus
                new_r = (r_swap + 1) % m_swap
                valid = True
                for i, (m_a, r_a) in enumerate(current):
                    if i == idx_swap:
                        continue
                    g = math.gcd(m_swap, m_a)
                    if (new_r % g) == (r_a % g):
                        valid = False
                        break
                if valid:
                    new_current = list(current)
                    new_current[idx_swap] = (m_swap, new_r)
                    # Check if this allows adding more moduli
                    # (simplified: just accept if valid)
                    current = new_current
            
            # Try removing a modulus and re-adding it with a different residue
            if len(current) > 1:
                idx_remove = random.randint(0, len(current) - 1)
                m_rem, r_rem = current[idx_remove]
                remaining = [x for i, x in enumerate(current) if i != idx_remove]
                # Try to re-add m_rem with a different residue
                for r_try in range(m_rem):
                    if r_try == r_rem:
                        continue
                    valid = True
                    for m_a, r_a in remaining:
                        g = math.gcd(m_rem, m_a)
                        if (r_try % g) == (r_a % g):
                            valid = False
                            break
                    if valid:
                        new_sol = remaining + [(m_rem, r_try)]
                        # Check if we can add more
                        current_moduli_new = set(m for m, r in new_sol)
                        available_new = [m for m in all_moduli if m not in current_moduli_new]
                        if available_new:
                            m_add2 = random.choice(available_new)
                            r_add2 = find_valid_residue(m_add2, new_sol, all_moduli)
                            if r_add2 is not None:
                                new_sol.append((m_add2, r_add2))
                        if len(new_sol) > current_size:
                            current = new_sol
                            current_size = len(current)
                        break
            
            # Try a random perturbation: remove a random subset and re-solve
            if len(current) > 2:
                remove_count = min(2, len(current) - 1)
                indices_to_remove = random.sample(range(len(current)), remove_count)
                remaining = [x for i, x in enumerate(current) if i not in set(indices_to_remove)]
                # Re-solve for the removed moduli and possibly add more
                removed_moduli = [current[i][0] for i in indices_to_remove]
                new_sol = list(remaining)
                for m_rem in removed_moduli:
                    r_rem = find_valid_residue(m_rem, new_sol, all_moduli)
                    if r_rem is not None:
                        new_sol.append((m_rem, r_rem))
                # Try to add more
                current_moduli_new = set(m for m, r in new_sol)
                available_new = [m for m in all_moduli if m not in current_moduli_new]
                for m_add in available_new[:5]:  # Try a few
                    r_add = find_valid_residue(m_add, new_sol, all_moduli)
                    if r_add is not None:
                        new_sol.append((m_add, r_add))
                if len(new_sol) > current_size:
                    current = new_sol
                    current_size = len(current)
        
        return current
    
    # Run local search
    improved = local_search(best_solution, max_iterations=500)
    if len(improved) > best_size:
        best_solution = improved
        best_size = len(best_solution)
    
    # Convert to required format
    result = []
    for m, r in best_solution:
        result.append([r, m])
    
    return result
