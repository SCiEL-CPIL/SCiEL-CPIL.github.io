import math
import time
import random

def construct():
    deadline = time.time() + 145
    
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    all_moduli = list(range(20, 201))
    
    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)
    
    def is_valid_add(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_r(m, selected):
        for r in range(m):
            if is_valid_add(m, r, selected):
                return r
        return None
    
    def find_all_valid_rs(m, selected):
        return [r for r in range(m) if is_valid_add(m, r, selected)]
    
    def greedy_fill(order, randomize_residue=False, rng=None):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_rs = find_all_valid_rs(m, selected)
            if valid_rs:
                if randomize_residue and rng:
                    selected.append([m, rng.choice(valid_rs)])
                else:
                    selected.append([m, valid_rs[0]])
        return selected
    
    def local_search_kswap(selected, max_iter=50, k=2):
        """k-swap local search: remove k elements, try to add k smaller ones."""
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline:
            improved = False
            iterations += 1
            current_score = calc_score(selected)
            sorted_sel = sorted(selected, key=lambda x: -x[0])
            n = len(sorted_sel)
            
            # Generate all k-subsets to remove (biased towards larger m)
            if k == 1:
                remove_indices = [range(n)]
            elif k == 2:
                remove_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
            else:
                # For k=3, sample subsets to avoid combinatorial explosion
                remove_indices = []
                for _ in range(min(200, n * (n-1) * (n-2) // 6)):
                    idx = rng.sample(range(n), k)
                    idx = tuple(sorted(idx))
                    if idx not in remove_indices:
                        remove_indices.append(idx)
            
            for rem_combo in remove_indices:
                if time.time() > deadline:
                    break
                if isinstance(rem_combo, tuple):
                    rem_set = set(rem_combo)
                else:
                    rem_set = set(rem_combo)
                
                new_selected = [x for i, x in enumerate(sorted_sel) if i not in rem_set]
                present_ms = set(m for m, _ in new_selected)
                
                # Try to add k smaller moduli
                added = []
                max_remove_m = max(sorted_sel[i][0] for i in rem_set)
                for m_add in range(20, max_remove_m):
                    if m_add in present_ms:
                        continue
                    found_r = find_valid_r(m_add, new_selected)
                    if found_r is not None:
                        new_selected.append([m_add, found_r])
                        added.append(m_add)
                        present_ms.add(m_add)
                        if len(added) == k:
                            break
                
                if len(added) == k:
                    new_score = calc_score(new_selected)
                    if new_score > current_score + 1e-9:
                        selected = new_selected
                        improved = True
                        break
        return selected
    
    def simulated_annealing(selected, iterations=20000):
        """Simulated annealing on the solution."""
        if not selected:
            return selected
        
        current = [x[:] for x in selected]
        current_score = calc_score(current)
        best = [x[:] for x in current]
        best_score = current_score
        
        T = 0.1
        cooling = 0.9999
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Random move: add, remove, or swap
            move_type = rng.random()
            new_sol = [x[:] for x in current]
            
            if move_type < 0.3:
                # Remove a random element
                if new_sol:
                    idx = rng.randrange(len(new_sol))
                    new_sol.pop(idx)
            elif move_type < 0.6:
                # Add a random missing modulus
                present = set(m for m, _ in new_sol)
                missing = [m for m in all_moduli if m not in present]
                if missing:
                    m_add = rng.choice(missing)
                    r_add = find_valid_r(m_add, new_sol)
                    if r_add is not None:
                        new_sol.append([m_add, r_add])
            else:
                # Swap: remove one, add one
                if new_sol:
                    idx = rng.randrange(len(new_sol))
                    m_removed = new_sol[idx][0]
                    new_sol.pop(idx)
                    present = set(m for m, _ in new_sol)
                    missing = [m for m in all_moduli if m not in present]
                    if missing:
                        m_add = rng.choice(missing)
                        r_add = find_valid_r(m_add, new_sol)
                        if r_add is not None:
                            new_sol.append([m_add, r_add])
            
            new_score = calc_score(new_sol)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
                current = new_sol
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = [x[:] for x in current]
            
            T *= cooling
            if T < 1e-6:
                T = 1e-6
        
        return best
    
    rng = random.Random(42)
    
    best_score = 0
    best_solution = []
    
    # Multiple strategies with different orderings
    strategies = [
        ("increasing", list(range(20, 201))),
        ("even_increasing", list(range(20, 201, 2))),
        ("decreasing", list(range(200, 19, -1))),
        ("odd_increasing", list(range(21, 201, 2))),
    ]
    
    for name, order in strategies:
        if time.time() > deadline - 10:
            break
        sol = greedy_fill(order, randomize_residue=False)
        sol = local_search_kswap(sol, max_iter=100, k=1)
        sol = local_search_kswap(sol, max_iter=50, k=2)
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Random restarts
    while time.time() < deadline - 15:
        restart = rng.random()
        
        if restart < 0.3:
            order = sorted(range(20, 201), key=lambda m: m + rng.uniform(0, 80))
        elif restart < 0.6:
            order = list(range(20, 201))
            rng.shuffle(order)
        else:
            order = list(range(20, 201))
            weights = [1.0 / m for m in order]
            remaining = list(order)
            weighted_order = []
            for _ in range(len(remaining)):
                w = [1.0 / m for m in remaining]
                total = sum(w)
                r = rng.random() * total
                cum = 0
                for idx, m in enumerate(remaining):
                    cum += w[idx]
                    if r <= cum:
                        weighted_order.append(m)
                        remaining.pop(idx)
                        break
            order = weighted_order
        
        sol = greedy_fill(order, randomize_residue=True, rng=rng)
        sol = local_search_kswap(sol, max_iter=80, k=1)
        sol = local_search_kswap(sol, max_iter=40, k=2)
        
        # Occasionally use simulated annealing
        if rng.random() < 0.3:
            sol = simulated_annealing(sol, iterations=5000)
        
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final refinement
    if time.time() < deadline - 5:
        best_solution = local_search_kswap(best_solution, max_iter=100, k=1)
        best_solution = local_search_kswap(best_solution, max_iter=50, k=2)
    
    return best_solution

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
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
