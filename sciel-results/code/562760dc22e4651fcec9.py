import math
import time
import random

def construct():
    deadline = time.time() + 115
    
    # Precompute gcd table
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Valid moduli: 20..200
    all_moduli = list(range(20, 201))
    mod_index = {m: i for i, m in enumerate(all_moduli)}
    
    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)
    
    def is_valid_add(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def find_valid_r(m, selected):
        """Find a valid residue for m given selected set. Returns None if none."""
        for r in range(m):
            if is_valid_add(m, r, selected):
                return r
        return None

    def greedy_fill(order):
        selected = []
        for m in order:
            r = find_valid_r(m, selected)
            if r is not None:
                selected.append([m, r])
        return selected

    # Initial solution: Greedy with small moduli first (high weight)
    initial_order = sorted(all_moduli) # 20, 21, ...
    current_sol = greedy_fill(initial_order)
    current_score = calc_score(current_sol)
    
    best_sol = [x[:] for x in current_sol]
    best_score = current_score
    
    rng = random.Random(42)
    
    # Simulated Annealing
    # State: list of [m, r]
    # Moves:
    # 1. Add a random modulus m not in current, with a random valid r.
    # 2. Remove a random modulus.
    # 3. Change residue of a random modulus to another valid one (if exists).
    # 4. Swap: Remove one, add another.
    
    T_start = 1.0
    T_end = 1e-5
    alpha = 0.995 # Cooling rate
    T = T_start
    
    steps = 0
    max_steps = 200000 # Cap steps to ensure we finish, but time is the real guard
    
    while time.time() < deadline - 5 and T > T_end:
        steps += 1
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a modulus
            # Pick a random modulus not in current
            current_ms = set(m for m, _ in current_sol)
            candidates = [m for m in all_moduli if m not in current_ms]
            if candidates:
                m_add = rng.choice(candidates)
                # Try to find a valid r
                # To make it faster, we can try a few random r's or the first valid one
                # Finding ANY valid r is O(m * N). 
                # Let's try to find one efficiently.
                # We can sample r or iterate. Iterating is fine for m<=200.
                r_add = None
                # Randomize order of r to avoid bias?
                rs = list(range(m_add))
                rng.shuffle(rs)
                for r in rs:
                    if is_valid_add(m_add, r, current_sol):
                        r_add = r
                        break
                
                if r_add is not None:
                    new_sol = current_sol + [[m_add, r_add]]
                    new_score = current_score + 1.0 / m_add
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < math.exp(delta / T):
                        current_sol = new_sol
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_sol = [x[:] for x in current_sol]
                            
        elif move_type < 0.7:
            # Try to remove a modulus
            if current_sol:
                idx_remove = rng.randint(0, len(current_sol) - 1)
                m_rem, r_rem = current_sol[idx_remove]
                new_sol = current_sol[:idx_remove] + current_sol[idx_remove+1:]
                new_score = current_score - 1.0 / m_rem
                delta = new_score - current_score
                # Removal always decreases score, so only accept if T is high enough
                # or if it allows future additions (handled by subsequent Add moves)
                if delta > 0 or rng.random() < math.exp(delta / T):
                    current_sol = new_sol
                    current_score = new_score
                    
        elif move_type < 0.9:
            # Try to change residue of an existing modulus
            if current_sol:
                idx = rng.randint(0, len(current_sol) - 1)
                m_cur, r_cur = current_sol[idx]
                # Find an alternative valid residue
                # We need to check validity against all OTHER elements
                others = [x for i, x in enumerate(current_sol) if i != idx]
                
                # Try random residues
                rs = list(range(m_cur))
                rng.shuffle(rs)
                found_alt = False
                for r_new in rs:
                    if r_new == r_cur:
                        continue
                    # Check validity against others
                    valid = True
                    for m2, r2 in others:
                        g = gcd_table[m_cur][m2]
                        if r_new % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        # Score doesn't change, but state does.
                        # Accepting this allows different future Add/Remove moves.
                        # In SA, we usually accept neutral moves or use them to escape.
                        # Let's accept with high probability to explore residue space.
                        if rng.random() < 0.5:
                            current_sol[idx][1] = r_new
                            found_alt = True
                        break
                        
        else:
            # Swap: Remove one, try to add a better/smaller one
            if current_sol:
                idx_remove = rng.randint(0, len(current_sol) - 1)
                m_rem, r_rem = current_sol[idx_remove]
                temp_sol = current_sol[:idx_remove] + current_sol[idx_remove+1:]
                
                # Try to add a random modulus
                current_ms = set(m for m, _ in temp_sol)
                candidates = [m for m in all_moduli if m not in current_ms]
                if candidates:
                    # Prefer smaller moduli for higher score gain
                    # Weighted choice
                    weights = [1.0/m for m in candidates]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    m_add = candidates[0]
                    for m, w in zip(candidates, weights):
                        cum += w
                        if r <= cum:
                            m_add = m
                            break
                    
                    r_add = None
                    rs = list(range(m_add))
                    rng.shuffle(rs)
                    for r in rs:
                        if is_valid_add(m_add, r, temp_sol):
                            r_add = r
                            break
                    
                    if r_add is not None:
                        new_sol = temp_sol + [[m_add, r_add]]
                        new_score = current_score - 1.0/m_rem + 1.0/m_add
                        delta = new_score - current_score
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            current_sol = new_sol
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_sol = [x[:] for x in current_sol]
        
        T *= alpha
        
    # Final local search refinement on best_sol
    # Simple 1-swap improvement
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_score = calc_score(best_sol)
        for i in range(len(best_sol)):
            if time.time() > deadline - 2:
                break
            m_rem, r_rem = best_sol[i]
            temp_sol = best_sol[:i] + best_sol[i+1:]
            present_ms = set(m for m, _ in temp_sol)
            
            for m_add in range(20, 201):
                if m_add in present_ms:
                    continue
                # Try to find valid r
                r_add = None
                for r in range(m_add):
                    if is_valid_add(m_add, r, temp_sol):
                        r_add = r
                        break
                
                if r_add is not None:
                    new_score = current_score - 1.0/m_rem + 1.0/m_add
                    if new_score > current_score + 1e-9:
                        best_sol = temp_sol + [[m_add, r_add]]
                        improved = True
                        break
            if improved:
                break
                
    return best_sol

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
