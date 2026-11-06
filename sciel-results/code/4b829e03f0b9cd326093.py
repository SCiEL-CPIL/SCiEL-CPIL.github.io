import time
import math
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute GCDs for speed
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Precompute 1/m as float for fast score comparison
    inv_m = {m: 1.0 / m for m in moduli}
    
    def check_valid(m, r, state, exclude_m=None):
        """Check if (m, r) is valid given state {m2: r2}, optionally excluding m"""
        for m2, r2 in state.items():
            if m2 == m: continue
            if exclude_m is not None and m2 == exclude_m: continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True
        
    def get_valid_residues(m, state, exclude_m=None):
        """Get all valid residues for modulus m given current state"""
        valid_rs = []
        for r in range(m):
            if check_valid(m, r, state, exclude_m):
                valid_rs.append(r)
        return valid_rs
        
    def compute_score_float(state):
        """Compute the score of a state as float for fast comparison"""
        return sum(inv_m[m] for m in state)
    
    def compute_score_exact(state):
        """Compute the exact rational score"""
        return sum(Fraction(1, m) for m in state)
    
    def simulated_annealing(initial_state, max_time=100):
        """
        Simulated annealing to maximize the score.
        Uses float scores for fast comparison during search.
        """
        current_state = dict(initial_state)
        current_score = compute_score_float(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        # Temperature schedule - tuned for this problem
        start_temp = 0.05
        end_temp = 0.0001
        cooling_rate = 0.999
        
        temp = start_temp
        iterations = 0
        start_time = time.time()
        
        while time.time() < start_time + max_time and temp > end_temp:
            iterations += 1
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random unselected modulus - bias toward smaller moduli (higher 1/m)
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    continue
                # Weight by 1/m to prefer smaller moduli
                weights = [inv_m[m] for m in unselected]
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
                        
            elif move_type < 0.55:
                # Remove a random selected modulus - bias toward larger moduli (lower 1/m)
                if not current_state:
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
                        
            elif move_type < 0.85:
                # Swap: remove one, add another
                if not current_state:
                    continue
                keys = list(current_state.keys())
                # Weight removal toward larger moduli
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
                    continue
                # Weight addition toward smaller moduli
                weights_add = [inv_m[m] for m in unselected]
                total_w_add = sum(weights_add)
                r_rand_add = random.random() * total_w_add
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights_add):
                    cum += w
                    if r_rand_add <= cum:
                        m_new = m
                        break
                
                # Try to find a valid residue for m_new after removing m_old
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
                            
            else:
                # Change residue of a random selected modulus
                if not current_state:
                    continue
                m_var = random.choice(list(current_state.keys()))
                old_r = current_state[m_var]
                new_r = random.randint(0, m_var - 1)
                
                if new_r != old_r:
                    # Check if new_r is valid
                    other_state = {k: v for k, v in current_state.items() if k != m_var}
                    if check_valid(m_var, new_r, other_state):
                        new_state = dict(current_state)
                        new_state[m_var] = new_r
                        # Score doesn't change for residue-only moves
                        new_score = current_score
                        delta = 0.0
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
            
            # Cool down
            temp *= cooling_rate
            
            # Periodic check for time limit
            if time.time() > deadline - 10:
                break
        
        return best_state, best_score
    
    def greedy_construct(randomize=False, order=None):
        """
        Greedy construction to create an initial solution.
        """
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
    
    best_solution = {}
    best_score = 0.0
    
    # Strategy:
    # 1. Create a good initial solution using greedy (sorted by 1/m descending)
    # 2. Run simulated annealing from that initial solution
    # 3. Try multiple restarts with different initial solutions
    
    # Run 1: Greedy sorted by 1/m descending (prefer small moduli)
    order_desc = sorted(moduli, key=lambda m: -inv_m[m])
    state1 = greedy_construct(randomize=False, order=order_desc)
    sol1, score1 = simulated_annealing(state1, max_time=35)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
        
    # Run 2: Greedy sorted ascending (prefer large moduli first, leaves room for small)
    order_asc = sorted(moduli, key=lambda m: inv_m[m])
    state2 = greedy_construct(randomize=False, order=order_asc)
    sol2, score2 = simulated_annealing(state2, max_time=25)
    if score2 > best_score:
        best_score = score2
        best_solution = sol2
    
    # Run 3-6: Randomized greedy + SA
    num_restarts = 4
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        state = greedy_construct(randomize=True)
        sol, score = simulated_annealing(state, max_time=15)
        
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final: if we have time, run SA on the best solution found with more iterations
    if time.time() < deadline - 15 and best_solution:
        sol_final, score_final = simulated_annealing(best_solution, max_time=15)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final
    
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
    # Verify validity
    from math import gcd
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Conflict: ({m1},{r1}) and ({m2},{r2})")
    print(f"Valid: {valid}")
