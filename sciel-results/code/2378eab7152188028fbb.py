import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 118s to be safe
    deadline = time.time() + 118
    
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
    
    def local_search(state, max_time=20):
        """
        Local search: try to improve by swapping residues and adding/removing moduli.
        """
        current_state = dict(state)
        current_score = compute_score(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < start_time + max_time and no_improve_count < max_no_improve:
            improved = False
            
            # Try adding each unselected modulus
            unselected = [m for m in moduli if m not in current_state]
            for m in unselected:
                if time.time() > deadline - 5:
                    break
                valid_rs = get_valid_residues(m, current_state)
                if valid_rs:
                    r = valid_rs[0]
                    new_state = dict(current_state)
                    new_state[m] = r
                    new_score = current_score + inv_m[m]
                    if new_score > best_score:
                        best_state = new_state
                        best_score = new_score
                        current_state = new_state
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            if improved:
                continue
            
            # Try swapping: remove one, add another
            keys = list(current_state.keys())
            unselected = [m for m in moduli if m not in current_state]
            
            for m_old in keys:
                if time.time() > deadline - 5:
                    break
                temp_state = dict(current_state)
                del temp_state[m_old]
                
                for m_new in unselected:
                    valid_rs = get_valid_residues(m_new, temp_state)
                    if valid_rs:
                        r_new = valid_rs[0]
                        new_state = dict(temp_state)
                        new_state[m_new] = r_new
                        new_score = current_score - inv_m[m_old] + inv_m[m_new]
                        if new_score > best_score:
                            best_state = new_state
                            best_score = new_score
                            current_state = new_state
                            current_score = new_score
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
        
        return best_state, best_score
    
    def simulated_annealing(initial_state, max_time=40):
        """
        Simulated annealing with multiple move types.
        """
        current_state = dict(initial_state)
        current_score = compute_score(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        start_temp = 0.05
        end_temp = 0.0001
        cooling_rate = 0.9997
        
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
                keys = list(current_state.keys())
                m_target = random.choice(keys)
                valid_rs = get_valid_residues(m_target, {k: v for k, v in current_state.items() if k != m_target})
                if valid_rs and len(valid_rs) > 1:
                    r_new = random.choice(valid_rs)
                    if r_new != current_state[m_target]:
                        new_state = dict(current_state)
                        new_state[m_target] = r_new
                        current_state = new_state
                        
            else:
                # Two-swap: remove two, add two
                if len(current_state) < 2:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                m_old1, m_old2 = random.sample(keys, 2)
                
                unselected = [m for m in moduli if m not in current_state]
                if len(unselected) < 2:
                    temp *= cooling_rate
                    continue
                m_new1, m_new2 = random.sample(unselected, 2)
                
                temp_state = dict(current_state)
                del temp_state[m_old1]
                del temp_state[m_old2]
                
                valid_rs1 = get_valid_residues(m_new1, temp_state)
                if valid_rs1:
                    r_new1 = random.choice(valid_rs1)
                    temp_state[m_new1] = r_new1
                    
                    valid_rs2 = get_valid_residues(m_new2, temp_state)
                    if valid_rs2:
                        r_new2 = random.choice(valid_rs2)
                        new_state = dict(temp_state)
                        new_state[m_new2] = r_new2
                        new_score = current_score - inv_m[m_old1] - inv_m[m_old2] + inv_m[m_new1] + inv_m[m_new2]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
            
            temp *= cooling_rate
        
        return best_state, best_score
    
    # Phase 1: Greedy initialization
    best_solution = {}
    best_score = 0.0
    
    # Try a few different greedy orders
    for i in range(15):
        if time.time() > deadline - 60:
            break
        order = moduli[:]
        if i == 0:
            order.sort(key=lambda m: -inv_m[m])  # Smallest m first
        elif i == 1:
            # Sort by number of constraints (fewer constraints first)
            order.sort(key=lambda m: sum(1 for j in moduli if math.gcd(m, j) > 1))
        elif i == 2:
            # Primorial-like: prefer numbers with few prime factors
            order.sort(key=lambda m: len([p for p in range(2, m+1) if m % p == 0 and all(p % q != 0 for q in range(2, p) if p % q == 0)]))
        else:
            random.shuffle(order)
        
        state = greedy_construct(order, randomize=(i > 2))
        score = compute_score(state)
        if score > best_score:
            best_score = score
            best_solution = state
    
    # Phase 2: Local search from best greedy solution
    if best_solution:
        sol, score = local_search(best_solution, max_time=25)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Phase 3: Simulated annealing from best solution
    if best_solution:
        sol, score = simulated_annealing(best_solution, max_time=35)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Phase 4: Multiple restarts
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        num_restarts = int(remaining_time / 15)
        for i in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            order = moduli[:]
            random.shuffle(order)
            state = greedy_construct(order, randomize=True)
            score = compute_score(state)
            
            if score > best_score:
                best_score = score
                best_solution = state
            
            # Local search
            sol, ls_score = local_search(state, max_time=8)
            if ls_score > best_score:
                best_score = ls_score
                best_solution = sol
            
            # Simulated annealing
            sol, sa_score = simulated_annealing(state, max_time=8)
            if sa_score > best_score:
                best_score = sa_score
                best_solution = sol
    
    # Phase 5: Final polish - try to add any missing small moduli
    if best_solution:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for m in sorted(moduli, key=lambda x: -inv_m[x]):
                if m in best_solution:
                    continue
                valid_rs = get_valid_residues(m, best_solution)
                if valid_rs:
                    best_solution[m] = valid_rs[0]
                    best_score += inv_m[m]
                    improved = True
    
    # Convert to required format
    if best_solution:
        return [[m, best_solution[m]] for m in sorted(best_solution.keys())]
    
    # Fallback: simple greedy
    state = greedy_construct(moduli, randomize=False)
    return [[m, state[m]] for m in sorted(state.keys())]

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
