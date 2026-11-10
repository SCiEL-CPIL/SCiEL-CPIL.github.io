import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Precompute triple indices for given n
    def get_triples(n):
        return list(itertools.combinations(range(n), 3))
    
    def build_matrices(n):
        triples = get_triples(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        num_fours = len(four_sets)
        num_fives = len(five_sets)
        
        # M: shape (num_fours, 4)
        M = np.zeros((num_fours, 4), dtype=np.int32)
        for i, quad in enumerate(four_sets):
            M[i, 0] = triple_to_idx[(quad[0], quad[1], quad[2])]
            M[i, 1] = triple_to_idx[(quad[0], quad[1], quad[3])]
            M[i, 2] = triple_to_idx[(quad[0], quad[2], quad[3])]
            M[i, 3] = triple_to_idx[(quad[1], quad[2], quad[3])]
            
        # Q: shape (num_fives, 10)
        Q = np.zeros((num_fives, 10), dtype=np.int32)
        for i, quint in enumerate(five_sets):
            idx = 0
            for t in itertools.combinations(quint, 3):
                Q[i, idx] = triple_to_idx[t]
                idx += 1
                
        return triples, M, Q, num_triples

    def calculate_cost(colors, M, Q):
        # colors is 1D numpy array of 0s and 1s
        # Red K4: all 4 triples in M are 0
        # Blue K5: all 10 triples in Q are 1
        red_viol = np.all(colors[M] == 0, axis=1)
        blue_viol = np.all(colors[Q] == 1, axis=1)
        return np.sum(red_viol) + np.sum(blue_viol)

    def solve_for_n(n, time_budget):
        triples, M, Q, num_triples = build_matrices(n)
        
        # Initial random coloring
        colors = np.random.randint(0, 2, num_triples)
        current_cost = calculate_cost(colors, M, Q)
        
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        restart_threshold = 5000  # If no improvement in this many steps, restart
        steps_since_improvement = 0
        
        while time.time() - start_time < time_limit - (115.0 - time_budget):
            if current_cost == 0:
                return colors.tolist()
            
            # Pick random triple
            t_idx = np.random.randint(0, num_triples)
            
            # Flip
            colors[t_idx] = 1 - colors[t_idx]
            
            # Calculate new cost
            new_cost = calculate_cost(colors, M, Q)
            
            # Acceptance
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # Probabilistic accept
                delta = new_cost - current_cost
                if T > min_T:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_cost = new_cost
                        steps_since_improvement += 1
                    else:
                        # Reject, revert
                        colors[t_idx] = 1 - colors[t_idx]
                        steps_since_improvement += 1
                else:
                    # Reject, revert
                    colors[t_idx] = 1 - colors[t_idx]
                    steps_since_improvement += 1
            
            T *= cooling_rate
            
            # Restart logic
            if steps_since_improvement > restart_threshold:
                # Reset temperature and randomize
                T = 1.0
                colors = np.random.randint(0, 2, num_triples)
                current_cost = calculate_cost(colors, M, Q)
                steps_since_improvement = 0
                # If current is better than best, update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()

        if best_cost == 0:
            return best_colors.tolist()
        return None

    # Try n=35 first, then 34, then 33...
    # We allocate time based on likelihood. 35 is hard, 34 is known.
    # Let's give 35 40s, 34 40s, 33 35s.
    
    # Actually, if 35 is impossible (if R(4,5;3) <= 35), we waste 40s.
    # But we must try.
    
    target_ns = [35, 34, 33, 32]
    
    # Dynamic time allocation
    remaining_time = time_limit - (time.time() - start_time)
    
    # Simple heuristic: try 35 with 50% of time, 34 with 30%, 33 with 20%
    # But if 35 fails, we still have time for 34.
    
    for n in target_ns:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Estimate time needed. Just use a fixed chunk per attempt.
        # If we have 100s left, give 40s to 35, 40s to 34, 20s to 33.
        # Adjust if we have less time.
        
        if n == 35:
            budget = min(40.0, remaining_time * 0.4)
        elif n == 34:
            budget = min(40.0, remaining_time * 0.4)
        else:
            budget = min(30.0, remaining_time * 0.2)
            
        if budget < 5.0:
            continue
            
        result = solve_for_n(n, budget)
        
        if result is not None:
            return {'n': n, 'colors': result}
            
        remaining_time = time_limit - (time.time() - start_time)
        
    # Fallback: return a valid small n
    # n=9 is always valid with (a+b+c)%3
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}
