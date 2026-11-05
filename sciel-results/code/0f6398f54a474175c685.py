import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    MAX_M = 128
    # Focus on even moduli as they form a large pairwise non-coprime set
    candidates = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    
    n = len(candidates)
    
    # Precompute gcds for the candidate set
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcd_tab[i][j] = math.gcd(candidates[i], candidates[j])

    def is_compatible(selected, new_idx, new_r):
        """Check if (new_idx, new_r) is disjoint from all in selected."""
        for (idx, r) in selected:
            g = gcd_tab[new_idx][idx]
            if (new_r - r) % g == 0:
                return False
        return True

    def count_blocked(selected, new_idx, new_r, unselected_set):
        """Count how many unselected moduli have this residue blocked."""
        blocked = 0
        for other_idx in unselected_set:
            if other_idx == new_idx:
                continue
            g = gcd_tab[new_idx][other_idx]
            # Check if any residue for other_idx would be compatible with new_r
            # Actually, we want to count how many residues of other_idx are blocked by new_r
            # A residue r_other is blocked if (new_r - r_other) % g == 0
            # Number of such r_other in [0, other_m) is:
            # r_other ≡ new_r (mod g), so count = ceil(other_m / g) or floor((other_m - 1 - new_r) / g) + 1
            # Simpler: count = (other_m + g - 1) // g  ... no, that's not right either
            # Let me just count directly for accuracy since m <= 128
            other_m = candidates[other_idx]
            cnt = 0
            for r_o in range(other_m):
                if (new_r - r_o) % g == 0:
                    cnt += 1
            blocked += cnt
        return blocked

    def solve_greedy_damage_minimizing(order, unselected_set):
        """Greedy that picks the residue minimizing damage to unselected candidates."""
        selected = []
        remaining = set(unselected_set)
        for idx in order:
            if idx not in remaining:
                continue
            m = candidates[idx]
            best_r = -1
            best_damage = float('inf')
            # Find all compatible residues, then pick the one with minimum damage
            compatible_residues = []
            for r in range(m):
                if is_compatible(selected, idx, r):
                    compatible_residues.append(r)
            
            if not compatible_residues:
                continue
            
            # Evaluate damage for each compatible residue
            for r in compatible_residues:
                dmg = count_blocked(selected, idx, r, remaining)
                if dmg < best_damage:
                    best_damage = dmg
                    best_r = r
            
            if best_r >= 0:
                selected.append((idx, best_r))
                remaining.discard(idx)
        return selected

    best_solution = []
    best_len = 0
    
    # Multiple random restarts with damage-minimizing greedy
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 20:
        order = list(range(n))
        random.shuffle(order)
        sol = solve_greedy_damage_minimizing(order, set(range(n)))
        if len(sol) > best_len:
            best_solution = sol
            best_len = len(sol)
        restart_count += 1
        # Early termination if we've found a large solution
        if best_len >= 15:
            break

    # Local search: try to add more
    selected_idx_set = set(idx for (idx, _) in best_solution)
    
    # Try to add any missing even moduli
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(n):
            if idx in selected_idx_set:
                continue
            m = candidates[idx]
            # Try to find a residue with minimum damage
            compatible_residues = []
            for r in range(m):
                if is_compatible(best_solution, idx, r):
                    compatible_residues.append(r)
            
            if compatible_residues:
                # Pick the one with minimum damage
                best_r = compatible_residues[0]
                best_dmg = float('inf')
                for r in compatible_residues:
                    dmg = count_blocked(best_solution, idx, r, set(range(n)) - selected_idx_set)
                    if dmg < best_dmg:
                        best_dmg = dmg
                        best_r = r
                best_solution.append((idx, best_r))
                selected_idx_set.add(idx)
                improved = True
                break

    # Final polish: try to maximize by re-running with best solution as seed
    if time.time() < deadline - 5:
        # Try to add missing moduli again with difficulty ordering
        unselected = [idx for idx in range(n) if idx not in selected_idx_set]
        
        def count_compatible(idx, selected):
            count = 0
            m = candidates[idx]
            for r in range(m):
                if is_compatible(selected, idx, r):
                    count += 1
            return count
        
        unselected_sorted = sorted(unselected, key=lambda idx: count_compatible(idx, best_solution))
        
        improved2 = True
        while improved2 and time.time() < deadline - 3:
            improved2 = False
            for idx in unselected_sorted:
                if idx in selected_idx_set:
                    continue
                m = candidates[idx]
                compatible_residues = []
                for r in range(m):
                    if is_compatible(best_solution, idx, r):
                        compatible_residues.append(r)
                if compatible_residues:
                    best_r = compatible_residues[0]
                    best_dmg = float('inf')
                    for r in compatible_residues:
                        dmg = count_blocked(best_solution, idx, r, set(range(n)) - selected_idx_set)
                        if dmg < best_dmg:
                            best_dmg = dmg
                            best_r = r
                    best_solution.append((idx, best_r))
                    selected_idx_set.add(idx)
                    improved2 = True
                    break
    
    result = []
    for (idx, r) in best_solution:
        result.append([r, candidates[idx]])
        
    return result
