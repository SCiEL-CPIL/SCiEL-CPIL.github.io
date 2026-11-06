import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 100s to be safe
    deadline = time.time() + 100

    max_m = 200
    min_m = 20

    # Precompute gcd table for fast lookup
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    def is_valid(m, r, selected, skip_idx=None):
        """Check if (m, r) is valid against selected classes."""
        if not selected:
            return True
        
        # Fast path: if skip_idx is None, check all
        if skip_idx is None:
            for m_sel, r_sel in selected:
                g = gcd_table[m][m_sel]
                if r % g == r_sel % g:
                    return False
            return True
        else:
            # Check all except skip_idx
            for i in range(len(selected)):
                if i == skip_idx:
                    continue
                m_sel, r_sel = selected[i]
                g = gcd_table[m][m_sel]
                if r % g == r_sel % g:
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

    def local_search_polish(selected, deadline):
        """Fast local search: add missing small moduli and 1-swaps."""
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        improved = True
        
        # Limit iterations to prevent hanging
        max_iters = 20
        iters = 0
        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try adding each missing modulus (prioritize small m)
            for m in range(min_m, max_m + 1):
                if any(m == m_sel for m_sel, _ in current_sol):
                    continue
                for r in range(m):
                    if is_valid(m, r, current_sol):
                        current_sol.append([m, r])
                        current_score += 1.0 / m
                        improved = True
                        break
            
            # 2. 1-swap: remove one class, add another with higher weight
            for idx in range(len(current_sol)):
                m_rem, r_rem = current_sol[idx]
                current_sol.pop(idx)
                current_score -= 1.0 / m_rem
                
                best_add = None
                best_add_score = -1
                # Only consider swaps if we can find something better
                for m_add in range(min_m, max_m + 1):
                    if m_add == m_rem:
                        continue
                    if any(m_add == m_sel for m_sel, _ in current_sol):
                        continue
                    # Quick check: if 1/m_add <= 1/m_rem, no point in swapping unless we are stuck
                    # But we want to maximize sum, so we only swap if we gain.
                    if 1.0 / m_add <= 1.0 / m_rem:
                        continue
                    
                    for r_add in range(m_add):
                        if is_valid(m_add, r_add, current_sol):
                            best_add = [m_add, r_add]
                            best_add_score = 1.0 / m_add
                            break
                
                if best_add:
                    current_sol.append(best_add)
                    current_score += best_add_score
                    improved = True
                else:
                    current_sol.insert(idx, [m_rem, r_rem])
                    current_score += 1.0 / m_rem
                    
        return current_sol, current_score

    def simulated_annealing(initial_solution, deadline):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score

        T = 0.5
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
                if random.random() < 0.3:
                    m_add = random.randint(min_m, min_m + 30)
                if any(m == m_add for m, _ in new_sol):
                    continue
                # Try several random residues
                for _ in range(8):
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
                weights = [1.0 / new_sol[i][0] for i in range(len(new_sol))]
                total_w = sum(weights)
                r = random.random() * total_w
                cumulative = 0
                idx = len(new_sol) - 1
                for i in range(len(new_sol)):
                    cumulative += weights[i]
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
                for _ in range(10):
                    new_r = random.randint(0, m_chg - 1)
                    if new_r == old_r:
                        continue
                    if is_valid(m_chg, new_r, new_sol, skip_idx=idx):
                        new_sol[idx] = [m_chg, new_r]
                        valid_move = True
                        break
            else:
                # Swap: remove one class, try to add another (different m)
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_old, r_old = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_old

                # Try to add a new class with different m
                for _ in range(15):
                    m_new = random.randint(min_m, max_m)
                    if m_new == m_old:
                        continue
                    if any(m == m_new for m, _ in new_sol):
                        continue
                    r_new = random.randint(0, m_new - 1)
                    if is_valid(m_new, r_new, new_sol):
                        new_sol.append([m_new, r_new])
                        new_score += 1.0 / m_new
                        valid_move = True
                        break

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

        return best_sol

    def initial_solution(deadline):
        """Build initial solution: prioritize small moduli, then fill greedily."""
        selected = []
        
        # Phase 1: Add moduli in increasing order (small m = high weight)
        for m in range(min_m, max_m + 1):
            for r in range(m):
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
            if time.time() > deadline:
                break

        # Phase 2: Try to add any remaining moduli
        selected = greedy_fill(selected, deadline)

        return selected

    # Run initial solution
    best_solution = initial_solution(deadline - 20)
    best_score = calc_score(best_solution)

    # Local search polish first (fast, deterministic)
    if time.time() < deadline - 15:
        polished, polished_score = local_search_polish(best_solution, deadline - 15)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

    # Run simulated annealing from polished solution
    # Reduce time budget for SA to ensure we finish
    sa_deadline = deadline - 10
    if time.time() < sa_deadline:
        result = simulated_annealing(best_solution, sa_deadline)
        score = calc_score(result)
        if score > best_score:
            best_score = score
            best_solution = result

    # Final local search polish
    if time.time() < deadline - 2:
        polished, polished_score = local_search_polish(best_solution, deadline - 2)
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished

    # Try a few more restarts if time permits
    restart_deadline = deadline - 5
    for _ in range(2):
        if time.time() > restart_deadline:
            break
        sol = initial_solution(deadline - 20)
        if time.time() < deadline - 15:
            sol, _ = local_search_polish(sol, deadline - 15)
        if time.time() < deadline - 10:
            sol = simulated_annealing(sol, deadline - 10)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol

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
