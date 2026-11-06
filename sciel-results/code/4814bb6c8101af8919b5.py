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
        
    def compute_score(state):
        """Compute the score of a state"""
        return sum(Fraction(1, m) for m in state)
    
    def simulated_annealing(initial_state, max_time=100):
        """
        Simulated annealing to maximize the score.
        """
        current_state = dict(initial_state)
        current_score = compute_score(current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        # Temperature schedule
        start_temp = 1.0
        end_temp = 0.001
        cooling_rate = 0.9995
        
        temp = start_temp
        iterations = 0
        start_time = time.time()
        
        while time.time() < start_time + max_time and temp > end_temp:
            iterations += 1
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random unselected modulus
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    continue
                m_new = random.choice(unselected)
                valid_rs = get_valid_residues(m_new, current_state)
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = dict(current_state)
                    new_state[m_new] = r_new
                    new_score = compute_score(new_state)
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_state = dict(current_state)
                        
            elif move_type < 0.6:
                # Remove a random selected modulus
                if not current_state:
                    continue
                m_old = random.choice(list(current_state.keys()))
                new_state = dict(current_state)
                del new_state[m_old]
                new_score = compute_score(new_state)
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
                m_old = random.choice(list(current_state.keys()))
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    continue
                m_new = random.choice(unselected)
                
                # Try to find a valid residue for m_new after removing m_old
                temp_state = dict(current_state)
                del temp_state[m_old]
                valid_rs = get_valid_residues(m_new, temp_state)
                
                if valid_rs:
                    r_new = random.choice(valid_rs)
                    new_state = temp_state
                    new_state[m_new] = r_new
                    new_score = compute_score(new_state)
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
                        new_score = compute_score(new_state)
                        delta = new_score - current_score
                        
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
        
        return best_state, float(best_score)
    
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
    # 1. Create a good initial solution using greedy
    # 2. Run simulated annealing from that initial solution
    # 3. Try multiple restarts with different initial solutions
    
    # Run 1: Deterministic greedy + SA
    state1 = greedy_construct(randomize=False)
    sol1, score1 = simulated_annealing(state1, max_time=40)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
        
    # Run 2-5: Randomized greedy + SA
    num_restarts = 4
    
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        state = greedy_construct(randomize=True)
        sol, score = simulated_annealing(state, max_time=25)
        
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final: if we have time, run SA on the best solution found
    if time.time() < deadline - 15 and best_solution:
        sol_final, score_final = simulated_annealing(best_solution, max_time=10)
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
