import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 115s to be safe
    deadline = time.time() + 115
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    n_moduli = len(moduli)
    
    # Precompute GCD table and reciprocal weights
    gcd_table = {}
    for i in moduli:
        for j in moduli:
            gcd_table[(i, j)] = math.gcd(i, j)
    
    inv_m = {m: 1.0 / m for m in moduli}
    
    def get_valid_residues(m, state):
        """Get all valid residues for modulus m given current state {m2: r2}"""
        valid_rs = []
        for r in range(m):
            ok = True
            for m2, r2 in state.items():
                if m2 == m:
                    continue
                g = gcd_table[(m, m2)]
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_rs.append(r)
        return valid_rs
    
    def check_valid(m, r, state, exclude_m=None):
        """Check if (m, r) is valid given state"""
        for m2, r2 in state.items():
            if m2 == m:
                continue
            if exclude_m is not None and m2 == exclude_m:
                continue
            g = gcd_table[(m, m2)]
            if r % g == r2 % g:
                return False
        return True
    
    def compute_score(state):
        return sum(inv_m[m] for m in state)
    
    def greedy_construct(order=None, randomize=False):
        """Greedy construction: add moduli in given order, picking first valid residue."""
        if order is None:
            order = moduli[:]
            if randomize:
                random.shuffle(order)
        
        state = {}
        for m in order:
            if time.time() > deadline - 30:
                break
            valid_rs = get_valid_residues(m, state)
            if valid_rs:
                if randomize:
                    r = random.choice(valid_rs)
                else:
                    r = valid_rs[0]
                state[m] = r
        return state
    
    def simulated_annealing(initial_state, max_time=40):
        """
        Simulated annealing with multiple move types.
        """
        current_state = dict(initial_state)
        current_score = compute_score(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        start_temp = 0.03
        end_temp = 0.00005
        cooling_rate = 0.9995
        
        temp = start_temp
        start_time = time.time()
        
        while time.time() < start_time + max_time and temp > end_temp:
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.30:
                # Add a random unselected modulus, biased toward smaller moduli
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                # Weight by 1/m^2 to strongly prefer small moduli
                weights = [inv_m[m] ** 2 for m in unselected]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights):
                    cum += w
                    if r_rand <= cum:
                        m_new = m
                        break
                valid_rs = get_valid_residues(m_new, current_state)
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = dict(current_state)
                    new_state[m_new] = r_new
                    new_score = current_score + inv_m[m_new]
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                            
            elif move_type < 0.45:
                # Remove a random selected modulus, biased toward larger moduli
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                # Weight by 1/m to prefer removing larger moduli
                weights = [inv_m[m] for m in keys]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights):
                    cum += w
                    if r_rand <= cum:
                        m_old = m
                        break
                new_state = dict(current_state)
                del new_state[m_old]
                new_score = current_score - inv_m[m_old]
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = dict(current_state)
                            
            elif move_type < 0.70:
                # Swap: remove one, add another
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                weights_rem = [inv_m[m] for m in keys]
                total_w_rem = sum(weights_rem)
                r_rand_rem = random.random() * total_w_rem
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights_rem):
                    cum += w
                    if r_rand_rem <= cum:
                        m_old = m
                        break
                
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                weights_add = [inv_m[m] ** 2 for m in unselected]
                total_w_add = sum(weights_add)
                r_rand_add = random.random() * total_w_add
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights_add):
                    cum += w
                    if r_rand_add <= cum:
                        m_new = m
                        break
                
                temp_state = dict(current_state)
                del temp_state[m_old]
                valid_rs = get_valid_residues(m_new, temp_state)
                
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = temp_state
                    new_state[m_new] = r_new
                    new_score = current_score - inv_m[m_old] + inv_m[m_new]
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                            
            elif move_type < 0.85:
                # Change residue of a selected modulus
                if not current_state:
                    temp *= cooling_rate
                    continue
                m_var = random.choice(list(current_state.keys()))
                old_r = current_state[m_var]
                
                # Try to find a new valid residue
                valid_rs = get_valid_residues(m_var, current_state)
                if len(valid_rs) > 1:
                    new_r = random.choice(valid_rs)
                    if new_r != old_r:
                        current_state[m_var] = new_r
                        # Score unchanged, but state improved for future moves

            else:
                # Random restart of a small subset to escape local minima?
                # Or just a no-op to cool down
                pass
            
            # Cool down
            temp *= cooling_rate
            
        return best_state, best_score

    def local_search(state, max_time=10):
        """
        Local search: try to add more moduli or swap to improve.
        """
        current_state = dict(state)
        current_score = compute_score(current_state)
        best_state = dict(current_state)
        best_score = current_score
        start_time = time.time()
        
        while time.time() < start_time + max_time:
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding each unselected modulus
            unselected = [m for m in moduli if m not in current_state]
            random.shuffle(unselected)
            
            for m_new in unselected:
                if time.time() > deadline - 5:
                    break
                valid_rs = get_valid_residues(m_new, current_state)
                if valid_rs:
                    r_new = valid_rs[0]
                    current_state[m_new] = r_new
                    current_score += inv_m[m_new]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_state = dict(current_state)
            
            # Try swapping: remove one large, add one small
            if not improved and current_state:
                keys = sorted(current_state.keys(), reverse=True)  # largest first
                unselected = [m for m in moduli if m not in current_state]
                unselected.sort()  # smallest first
                
                for m_old in keys[:5]:
                    if time.time() > deadline - 5:
                        break
                    for m_new in unselected[:10]:
                        if m_new >= m_old:
                            continue
                        # Check if m_new can be added after removing m_old
                        temp_state = dict(current_state)
                        del temp_state[m_old]
                        valid_rs = get_valid_residues(m_new, temp_state)
                        
                        if valid_rs:
                            r_new = valid_rs[0]
                            new_score = current_score - inv_m[m_old] + inv_m[m_new]
                            if new_score > current_score:
                                current_state = temp_state
                                current_state[m_new] = r_new
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_state = dict(current_state)
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_state, best_score

    best_solution = {}
    best_score = 0.0
    
    # Strategy:
    # 1. Create initial solutions using greedy
    # 2. Run SA from each
    # 3. Apply local search to best SA results
    # 4. Multiple restarts

    # Run 1: Deterministic greedy + SA
    state1 = greedy_construct(randomize=False)
    sol1, score1 = simulated_annealing(state1, max_time=35)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
        
    # Run 2-5: Randomized greedy + SA
    for i in range(4):
        if time.time() > deadline - 30:
            break
            
        state = greedy_construct(randomize=True)
        sol, score = simulated_annealing(state, max_time=20)
        
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final: local search on the best solution
    if time.time() < deadline - 10 and best_solution:
        sol_final, score_final = local_search(best_solution, max_time=15)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final
    
    # One more SA pass on the best
    if time.time() < deadline - 5 and best_solution:
        sol_final2, score_final2 = simulated_annealing(best_solution, max_time=5)
        if score_final2 > best_score:
            best_score = score_final2
            best_solution = sol_final2
    
    # Convert to list of lists
    if best_solution:
        return [[m, r] for m, r in best_solution.items()]
    
    # Fallback
    state = greedy_construct(randomize=False)
    return [[m, r] for m, r in state.items()]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
