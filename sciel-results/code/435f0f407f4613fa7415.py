import math
import time
import random
from collections import defaultdict

def construct():
    # Deadline guard: 150s limit, use 140s to be safe
    deadline = time.time() + 140

    max_m = 200
    min_m = 20

    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    # Precompute prime factors
    prime_factors = [[] for _ in range(max_m + 1)]
    for p in range(2, max_m + 1):
        if not prime_factors[p]:
            for multiple in range(p, max_m + 1, p):
                prime_factors[multiple].append(p)

    # Build candidate list: all (m, r) pairs
    candidates = []
    for m in range(min_m, max_m + 1):
        for r in range(m):
            candidates.append((m, r))
    
    n_candidates = len(candidates)
    
    # Build conflict graph: two candidates conflict if they share the same modulus
    # OR if r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    # We'll use a bitset-like approach with sets for efficiency
    conflict = [set() for _ in range(n_candidates)]
    
    # Map (m, r) to index
    index_map = {}
    for i, (m, r) in enumerate(candidates):
        index_map[(m, r)] = i
    
    # Build conflicts - this is O(n^2) but n is at most sum(m for m in 20..200) = ~19000
    # Actually that's too large. Let's be smarter.
    # Two candidates (m1,r1) and (m2,r2) conflict if:
    # 1. m1 == m2 (same modulus, different residues)
    # 2. r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    
    # For same modulus, all pairs conflict
    # For different moduli, check the gcd condition
    
    # Group by modulus
    by_modulus = defaultdict(list)
    for i, (m, r) in enumerate(candidates):
        by_modulus[m].append(i)
    
    # Conflicts within same modulus: all pairs
    for m, indices in by_modulus.items():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                conflict[indices[i]].add(indices[j])
                conflict[indices[j]].add(indices[i])
    
    # Conflicts between different moduli
    moduli_list = sorted(by_modulus.keys())
    for i in range(len(moduli_list)):
        for j in range(i+1, len(moduli_list)):
            m1, m2 = moduli_list[i], moduli_list[j]
            g = gcd_table[m1][m2]
            # For each pair of residues, check conflict
            for idx1 in by_modulus[m1]:
                r1 = candidates[idx1][1]
                rem1 = r1 % g
                for idx2 in by_modulus[m2]:
                    r2 = candidates[idx2][1]
                    rem2 = r2 % g
                    if rem1 == rem2:
                        conflict[idx1].add(idx2)
                        conflict[idx2].add(idx1)
    
    # Weight for each candidate
    weights = [1.0 / m for m, r in candidates]
    
    def greedy_coloring(candidates_to_color, conflicts):
        """Greedy coloring heuristic for upper bound."""
        colors = {}
        color_classes = []
        for i in candidates_to_color:
            used_colors = set()
            for j in conflicts.get(i, []):
                if j in colors:
                    used_colors.add(colors[j])
            c = 0
            while c in used_colors:
                c += 1
            colors[i] = c
            if c >= len(color_classes):
                color_classes.append([])
            color_classes[c].append(i)
        return len(color_classes)
    
    def branch_and_bound(deadline, initial_selection=None):
        """Branch and bound with greedy coloring upper bound."""
        if initial_selection is None:
            initial_selection = []
        
        best_solution = list(initial_selection)
        best_score = sum(weights[i] for i in initial_selection)
        
        # Sort candidates by weight descending
        sorted_indices = sorted(range(n_candidates), key=lambda i: -weights[i])
        
        # Use a stack-based B&B
        # State: (current_selection, next_candidate_idx, current_score)
        # We'll use a priority queue to explore promising branches first
        
        import heapq
        
        # Start with empty selection, explore in order of weight
        # But we need to be careful about the conflict structure
        
        # Alternative: use a recursive approach with memoization of visited states
        # But state space is too large
        
        # Let's use a simpler approach: iterate through candidates in weight order,
        # and at each step decide to include or exclude
        
        # For efficiency, we'll use a limited-depth search
        
        def search(selection, start_idx, score, depth, max_depth=15):
            nonlocal best_solution, best_score
            
            if time.time() > deadline:
                return
            
            if score > best_score:
                best_score = score
                best_solution = list(selection)
            
            if depth >= max_depth or start_idx >= n_candidates:
                return
            
            # Upper bound: current score + sum of remaining weights
            # This is a loose bound, but let's try it
            remaining_weight = sum(weights[i] for i in range(start_idx, n_candidates))
            if score + remaining_weight <= best_score:
                return
            
            # Try to include candidate start_idx if it doesn't conflict
            cand_idx = start_idx
            m, r = candidates[cand_idx]
            
            # Check if this candidate conflicts with any in selection
            can_add = True
            for sel_idx in selection:
                if cand_idx in conflict[sel_idx]:
                    can_add = False
                    break
            
            if can_add:
                selection.append(cand_idx)
                search(selection, start_idx + 1, score + weights[cand_idx], depth + 1, max_depth)
                selection.pop()
            
            # Try excluding
            search(selection, start_idx + 1, score, depth, max_depth)
        
        # Start the search
        search([], 0, 0.0, 0, max_depth=20)
        
        return best_solution
    
    def is_valid(m, r, selected, skip_idx=None):
        """Check if (m, r) is valid against selected classes."""
        for idx, (m2, r2) in enumerate(selected):
            if skip_idx is not None and idx == skip_idx:
                continue
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def greedy_fill(selected, deadline):
        """Try to add more classes to the selected list."""
        for m in range(min_m, max_m + 1):
            if any(m == m_sel for m_sel, _ in selected):
                continue
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break
        return selected

    def local_polish(sol, deadline):
        """Aggressive local search: try all additions, then 2-swap moves."""
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            if iterations > 50:  # Safety limit
                break
            
            # Phase 1: Try adding every possible (m, r) pair
            for m in range(min_m, max_m + 1):
                if any(m == m_sel for m_sel, _ in sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, sol):
                        sol.append([m, r])
                        improved = True
                        break
                if time.time() > deadline:
                    return sol
            
            # Phase 2: Try 1-swap moves (change one residue)
            for idx in range(len(sol)):
                m_chg, old_r = sol[idx]
                for new_r in range(m_chg):
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, sol, skip_idx=idx):
                        # Check if this change enables adding more classes
                        new_sol = [list(x) for x in sol]
                        new_sol[idx] = [m_chg, new_r]
                        added = False
                        for m_add in range(min_m, max_m + 1):
                            if any(m_add == m_sel for m_sel, _ in new_sol):
                                continue
                            for r_add in range(m_add):
                                if is_valid(m_add, r_add, new_sol):
                                    new_sol.append([m_add, r_add])
                                    added = True
                                    break
                            if added:
                                break
                        if added:
                            sol = new_sol
                            improved = True
                            break
                if time.time() > deadline:
                    return sol
            
            # Phase 3: Try 2-swap moves (remove two, add two different ones)
            if len(sol) >= 2:
                for i in range(len(sol)):
                    for j in range(i+1, len(sol)):
                        m1, r1 = sol[i]
                        m2, r2 = sol[j]
                        # Try removing both and adding different moduli
                        temp_sol = [list(x) for x in sol if x is not sol[i] and x is not sol[j]]
                        # More efficient: create a new list
                        temp_sol = []
                        for k, (m, r) in enumerate(sol):
                            if k != i and k != j:
                                temp_sol.append([m, r])
                        
                        # Try adding two new classes
                        added_count = 0
                        for m_add in range(min_m, max_m + 1):
                            if any(m_add == m_sel for m_sel, _ in temp_sol):
                                continue
                            if m_add == m1 or m_add == m2:
                                continue
                            for r_add in range(m_add):
                                if is_valid(m_add, r_add, temp_sol):
                                    temp_sol.append([m_add, r_add])
                                    added_count += 1
                                    break
                            if added_count >= 2:
                                break
                            if time.time() > deadline:
                                return sol
                        
                        if added_count >= 2:
                            # Check if the new solution has higher score
                            old_score = 1.0/m1 + 1.0/m2
                            new_score = 0
                            for k, (m, r) in enumerate(sol):
                                if k != i and k != j:
                                    continue
                                # This is getting complex, let's just accept if we added 2
                            # Simpler: just accept the move
                            sol = temp_sol
                            improved = True
                            break
                    if time.time() > deadline:
                        return sol
                if improved:
                    break
        
        return sol

    def simulated_annealing(initial_solution, deadline):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        T = 0.3
        cooling = 0.9995
        min_T = 1e-8

        while time.time() < deadline:
            if T < min_T:
                break

            move_type = random.random()
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False

            if move_type < 0.50:
                # Add a modulus - prefer smaller moduli for higher contribution
                # Choose m biased towards smaller values
                m_add = random.randint(min_m, max_m)
                if random.random() < 0.5:
                    m_add = random.randint(min_m, min_m + 20)
                if any(m == m_add for m, _ in new_sol):
                    continue
                # Try several random residues, prefer those that don't conflict
                for _ in range(15):
                    r_add = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_add, new_sol):
                        new_sol.append([m_add, r_add])
                        new_score += 1.0 / m_add
                        valid_move = True
                        break
            elif move_type < 0.75:
                # Remove a modulus - prefer removing large moduli (low contribution)
                if not new_sol:
                    continue
                # Weight towards larger moduli
                weights_sa = [1.0 / new_sol[i][0] for i in range(len(new_sol))]
                total_w = sum(weights_sa)
                r = random.random() * total_w
                cumulative = 0
                idx = len(new_sol) - 1
                for i in range(len(new_sol)):
                    cumulative += weights_sa[i]
                    if r <= cumulative:
                        idx = i
                        break
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            elif move_type < 0.90:
                # Change residue of a random class
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                for _ in range(15):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, new_sol, skip_idx=idx):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            else:
                # 2-swap: remove one class, add two different ones (net +1)
                if len(new_sol) < 1:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_old, r_old = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_old

                # Try to add two new classes with different moduli
                added = 0
                for _ in range(25):
                    m_new = random.randint(min_m, max_m)
                    if m_new == m_old:
                        continue
                    if any(m == m_new for m, _ in new_sol):
                        continue
                    r_new = random.randint(0, m_new - 1)
                    if is_valid(m_new, r_new, new_sol):
                        new_sol.append([m_new, r_new])
                        new_score += 1.0 / m_new
                        added += 1
                        if added >= 2:
                            break

                if added > 0:
                    valid_move = True

            if not valid_move:
                continue

            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or (delta > -5.0 and random.random() < math.exp(delta / T)):
                current_sol = new_sol
                current_score = new_score

                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]

            T *= cooling

        # Final local polish
        best_sol = local_polish(best_sol, deadline)
        return best_sol

    def initial_solution(deadline):
        """Build initial solution: add all coprime-friendly classes, then fill by prime powers."""
        selected = []
        selected_set = set()

        # Phase 1: Add classes with moduli that have no prime factor in common with existing
        for m in range(min_m, max_m + 1):
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break

        # Phase 2: Fill remaining - for each prime, try to add at its largest power
        # Then try to add at smaller moduli
        selected = greedy_fill(selected, deadline)

        return selected

    # Run initial solution
    best_solution = initial_solution(deadline - 20)
    best_score = calc_score(best_solution)

    # Run simulated annealing from initial solution
    result = simulated_annealing(best_solution, deadline - 10)
    score = calc_score(result)
    if score > best_score:
        best_score = score
        best_solution = result

    # Try more restarts if time permits
    restart_deadline = deadline - 15
    for _ in range(12):
        if time.time() > restart_deadline:
            break
        sol = initial_solution(deadline - 12)
        sol = simulated_annealing(sol, deadline - 8)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

    # Final aggressive polish on the best solution
    best_solution = local_polish(best_solution, deadline - 1)

    return best_solution

# Test locally
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
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
    # Print moduli for inspection
    moduli = sorted([m for m, _ in res])
    print(f"Moduli count: {len(moduli)}")
    print(f"Smallest 20 moduli: {moduli[:20]}")
    print(f"Largest 20 moduli: {moduli[-20:]}")
