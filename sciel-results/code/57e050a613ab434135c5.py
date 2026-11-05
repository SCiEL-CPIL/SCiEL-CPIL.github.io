import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_mod = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for i in range(n_mod):
        for j in range(i+1, n_mod):
            gcd_table[(i, j)] = math.gcd(moduli[i], moduli[j])
    
    def check_disjoint(r1, m1, r2, m2):
        """Check if residue classes (r1,m1) and (r2,m2) are disjoint."""
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def is_compatible(r_new, m_new, selected):
        """Check if (r_new, m_new) is compatible with all selected classes."""
        for r_k, m_k in selected:
            g = math.gcd(m_new, m_k)
            if (r_new - r_k) % g == 0:
                return False
        return True
    
    def count_valid_residues(m_new, selected):
        """Count how many residues for m_new are compatible with selected."""
        count = 0
        for r in range(m_new):
            if is_compatible(r, m_new, selected):
                count += 1
        return count
    
    def find_valid_residue(m_new, selected):
        """Find a valid residue for m_new given selected classes."""
        for r in range(m_new):
            if is_compatible(r, m_new, selected):
                return r
        return None
    
    # Strategy 1: Greedy with multiple orderings
    def greedy_build(order):
        selected = []
        for m in order:
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((r, m))
        return selected
    
    # Strategy 2: Local search - try to swap or add
    def local_search(initial_solution, max_iter=20000):
        selected = list(initial_solution)
        selected_moduli = set(m for r, m in selected)
        remaining = [m for m in moduli if m not in selected_moduli]
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() > deadline:
                break
            
            # Try adding a remaining modulus
            for m in remaining[:]:
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append((r, m))
                    selected_moduli.add(m)
                    remaining.remove(m)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one selected, add two remaining
            for i, (r_i, m_i) in enumerate(selected):
                if time.time() > deadline:
                    return selected
                # Try removing m_i and adding a different modulus
                for m_j in remaining[:]:
                    # Check if we can add m_j without m_i
                    test_selected = selected[:i] + selected[i+1:]
                    r_j = find_valid_residue(m_j, test_selected)
                    if r_j is not None:
                        # Check if we can also add another modulus
                        new_selected = test_selected + [(r_j, m_j)]
                        for m_k in remaining:
                            if m_k == m_j:
                                continue
                            r_k = find_valid_residue(m_k, new_selected)
                            if r_k is not None:
                                # Do the swap: remove m_i, add m_j and m_k
                                selected = new_selected + [(r_k, m_k)]
                                selected_moduli = set(m for r, m in selected)
                                remaining = [m for m in moduli if m not in selected_moduli]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        return selected
    
    # Strategy 3: Backtracking with MRV heuristic and constraint propagation
    def backtrack_mrv(selected, remaining):
        """Backtracking with Minimum Remaining Values heuristic."""
        if time.time() > deadline:
            return selected
        
        if not remaining:
            return selected
        
        # MRV: pick the modulus with fewest valid residues
        min_count = float('inf')
        best_idx = 0
        for idx, m in enumerate(remaining):
            count = count_valid_residues(m, selected)
            if count < min_count:
                min_count = count
                best_idx = idx
            if count == 0:
                break
        
        if min_count == 0:
            return selected
        
        m = remaining[best_idx]
        new_remaining = remaining[:best_idx] + remaining[best_idx+1:]
        
        # Try each valid residue
        for r in range(m):
            if is_compatible(r, m, selected):
                result = backtrack_mrv(selected + [(r, m)], new_remaining)
                if len(result) > len(selected):
                    return result
        
        # If no residue works, skip this modulus
        return backtrack_mrv(selected, new_remaining)
    
    # Strategy 4: Max-flow inspired approach
    # Model: We want to select maximum moduli and assign residues.
    # Key insight: moduli with gcd=1 to all others are "free" - always compatible.
    # Group moduli by their "conflict structure" and solve each group.
    
    def compute_conflict_degree(m, all_moduli):
        """Count how many moduli have gcd > 1 with m."""
        count = 0
        for other in all_moduli:
            if other != m and math.gcd(m, other) > 1:
                count += 1
        return count
    
    # Priority: moduli with fewer conflicts first (more "free")
    conflict_degrees = {m: compute_conflict_degree(m, moduli) for m in moduli}
    
    # Run multiple strategies and keep the best
    best_solution = []
    best_len = 0
    
    # Greedy with different orderings
    orders = []
    orders.append(sorted(moduli, key=lambda m: conflict_degrees[m]))  # Fewest conflicts first
    orders.append(sorted(moduli, key=lambda m: -conflict_degrees[m]))  # Most conflicts first
    orders.append(sorted(moduli, key=lambda m: -math.gcd(m, 2)))  # Even first
    orders.append(sorted(moduli, key=lambda m: -math.gcd(m, 3)))  # Multiples of 3 first
    orders.append(sorted(moduli))
    orders.append(sorted(moduli, reverse=True))
    
    # Add some random orders
    random.seed(123)
    for _ in range(20):
        shuffled = moduli.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy_build(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
    
    # Local search from best greedy
    if best_len > 0:
        improved_sol = local_search(best_solution)
        if len(improved_sol) > best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
    
    # Backtracking with MRV from best solution
    if best_len > 0 and time.time() < deadline - 10:
        selected_moduli = set(m for r, m in best_solution)
        remaining = [m for m in moduli if m not in selected_moduli]
        
        # Try backtracking from empty with MRV
        bt_sol = backtrack_mrv([], moduli)
        if len(bt_sol) > best_len:
            best_len = len(bt_sol)
            best_solution = bt_sol
        
        # Try backtracking extending best solution
        if time.time() < deadline - 5:
            ext_sol = backtrack_mrv(list(best_solution), remaining)
            if len(ext_sol) > best_len:
                best_len = len(ext_sol)
                best_solution = ext_sol
    
    # Final local search pass
    if best_len > 0 and time.time() < deadline - 2:
        final_sol = local_search(best_solution, max_iter=50000)
        if len(final_sol) > best_len:
            best_len = len(final_sol)
            best_solution = final_sol
    
    # Ensure valid output
    if not best_solution:
        # Fallback: just pick coprime moduli
        best_solution = [(0, 2)]
    
    return [[r, m] for r, m in best_solution]
