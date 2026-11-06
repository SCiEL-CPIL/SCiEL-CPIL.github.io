import math
import time
import random
from functools import lru_cache

def construct():
    # Deadline guard: 150s limit, use 135s to be safe
    deadline = time.time() + 135
    
    max_m = 200
    min_m = 20
    
    # Precompute gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Helper to check if (m, r) is valid against a list of selected (m, r)
    def is_valid(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    # Greedy initialization: Pick smallest m with any valid r
    def greedy_init():
        selected = []
        for m in range(min_m, max_m + 1):
            # Try to find a valid r
            valid_r = None
            for r in range(m):
                if is_valid(m, r, selected):
                    valid_r = r
                    break
            if valid_r is not None:
                selected.append([m, valid_r])
        return selected

    # Simulated Annealing
    def simulated_annealing(initial_solution, time_limit=10):
        current_sol = [list(x) for x in initial_solution]
        current_score = calc_score(current_sol)
        best_sol = [list(x) for x in current_sol]
        best_score = current_score
        
        T = 0.1
        cooling = 0.9995
        min_T = 1e-8
        
        all_moduli = list(range(min_m, max_m + 1))
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < min_T:
                break
                
            move_type = random.random()
            
            new_sol = [list(x) for x in current_sol]
            new_score = current_score
            valid_move = False
            
            if move_type < 0.5:
                selected_ms = set(m for m, _ in new_sol)
                unselected = [m for m in all_moduli if m not in selected_ms]
                if not unselected:
                    continue
                
                m_add = random.choice(unselected)
                
                r_add = None
                for _ in range(10):
                    r_try = random.randint(0, m_add - 1)
                    if is_valid(m_add, r_try, new_sol):
                        r_add = r_try
                        break
                
                if r_add is None:
                    for r_try in range(m_add):
                        if is_valid(m_add, r_try, new_sol):
                            r_add = r_try
                            break
                
                if r_add is not None:
                    new_sol.append([m_add, r_add])
                    new_score += 1.0 / m_add
                    valid_move = True
            elif move_type < 0.8:
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_rem, r_rem = new_sol[idx]
                new_sol.pop(idx)
                new_score -= 1.0 / m_rem
                valid_move = True
            else:
                if not new_sol:
                    continue
                idx = random.randint(0, len(new_sol) - 1)
                m_chg, old_r = new_sol[idx]
                
                new_r = None
                for _ in range(5):
                    r_try = random.randint(0, m_chg - 1)
                    if r_try == old_r:
                        continue
                    others = [x for i, x in enumerate(new_sol) if i != idx]
                    if is_valid(m_chg, r_try, others):
                        new_r = r_try
                        break
                
                if new_r is not None:
                    new_sol[idx] = [m_chg, new_r]
                    valid_move = True
            
            if not valid_move:
                continue
                
            delta = new_score - current_score
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_sol = new_sol
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_sol = [list(x) for x in current_sol]
            
            T *= cooling
            
        return best_sol

    # Branch and Bound for Maximum Weight Independent Set
    # Each "node" is a modulus m, and we choose at most one residue per modulus
    # Conflict: (m1, r1) and (m2, r2) conflict if r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    
    def branch_and_bound():
        """
        B&B over moduli. For each modulus, decide: skip it, or pick one of its residues.
        Upper bound: current score + sum of 1/m for all remaining moduli (fractional knapsack style)
        """
        all_moduli = list(range(min_m, max_m + 1))
        n = len(all_moduli)
        
        # Precompute weights
        weights = [1.0 / m for m in all_moduli]
        
        # For upper bound: suffix sums of weights
        suffix_sum = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
        best_score = 0.0
        best_solution = []
        
        # State: index in all_moduli, current selected list
        def bb(idx, current_selected, current_score):
            nonlocal best_score, best_solution
            
            if time.time() > deadline - 2:
                return
            
            # Upper bound
            ub = current_score + suffix_sum[idx]
            if ub <= best_score + 1e-12:
                return
            
            if idx == n:
                if current_score > best_score:
                    best_score = current_score
                    best_solution = [list(x) for x in current_selected]
                return
            
            m = all_moduli[idx]
            
            # Option 1: Skip this modulus
            bb(idx + 1, current_selected, current_score)
            
            # Option 2: Try each residue for this modulus
            # We need to check validity against current_selected
            # To prune, we can try residues in order of... well, all residues have same weight
            # So just try all valid ones
            
            # Collect valid residues
            valid_residues = []
            for r in range(m):
                if is_valid(m, r, current_selected):
                    valid_residues.append(r)
            
            for r in valid_residues:
                new_selected = current_selected + [[m, r]]
                bb(idx + 1, new_selected, current_score + 1.0 / m)
        
        # Start B&B
        bb(0, [], 0.0)
        
        return best_solution, best_score

    # Run multiple strategies
    best_solution = []
    best_score = 0.0
    
    # 1. Deterministic Greedy + SA
    sol = greedy_init()
    sol = simulated_annealing(sol, time_limit=15)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
    
    # 2. Branch and Bound
    remaining_time = deadline - time.time() - 2
    if remaining_time > 5:
        try:
            bb_sol, bb_score = branch_and_bound()
            if bb_score > best_score:
                best_score = bb_score
                best_solution = bb_sol
        except Exception:
            pass
    
    # 3. Randomized Greedy Restarts with SA
    random.seed(42)
    while time.time() < deadline - 2:
        sol = []
        for m in range(min_m, max_m + 1):
            valid_rs = []
            for r in range(m):
                if is_valid(m, r, sol):
                    valid_rs.append(r)
            if valid_rs:
                sol.append([m, random.choice(valid_rs)])
        
        # Short SA polish
        sa_time = min(5, max(1, deadline - time.time() - 2))
        sol = simulated_annealing(sol, time_limit=sa_time)
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
