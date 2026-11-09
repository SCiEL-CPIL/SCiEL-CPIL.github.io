import random
import time
import math

def construct():
    n = 180
    elements = list(range(1, n + 1))
    
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c  <=>  (b-a)(c-a) = a^2
    # We iterate a, b and solve for c.
    forbidden_triples = []
    seen = set()
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if b == a:
                continue
            # c = a*b / (b-a)
            den = b - a
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c < 1 or c > n:
                continue
            if c == a or c == b:
                continue
            # Ensure distinctness
            if len({a, b, c}) != 3:
                continue
            triple = tuple(sorted((a, b, c)))
            if triple not in seen:
                seen.add(triple)
                forbidden_triples.append(triple)
                
    # Map element index (0-based) to list of triple indices it belongs to
    # Elements are 1..180, indices 0..179
    elem_to_triples = [[] for _ in range(n)]
    for i, tri in enumerate(forbidden_triples):
        for e in tri:
            elem_to_triples[e - 1].append(i)
            
    num_triples = len(forbidden_triples)
    
    # Helper to calculate violations for a given state (list of 0/1)
    def count_violations(state):
        count = 0
        for tri in forbidden_triples:
            # tri contains 1-based indices
            if state[tri[0]-1] and state[tri[1]-1] and state[tri[2]-1]:
                count += 1
        return count

    # We will run SA multiple times to increase chances of finding a large set.
    # Time limit: 55 seconds total.
    start_time = time.time()
    time_budget = 55.0
    
    best_size = 0
    best_state = None
    
    # We can try different initializations.
    # 1. All 1s
    # 2. Random half
    # 3. Greedy feasible
    
    def run_sa(initial_state, time_limit):
        # initial_state is a list of 0/1
        current_state = list(initial_state)
        current_violations = count_violations(current_state)
        current_size = sum(current_state)
        
        # Energy = -size + penalty * violations
        # We want to minimize Energy.
        # If violations > 0, energy is very high.
        # If violations == 0, energy = -size. Minimizing this maximizes size.
        penalty = 10000 # Must be > n
        
        def get_energy(state, viol):
            return -sum(state) + penalty * viol
            
        current_energy = get_energy(current_state, current_violations)
        
        best_local_size = current_size if current_violations == 0 else -1
        best_local_state = list(current_state) if current_violations == 0 else None
        
        T = 100.0
        T_min = 0.01
        cooling = 0.9995
        
        # Precompute neighbors for faster delta calculation? 
        # Delta for flipping index i:
        # If x_i becomes 1:
        #   Size +1
        #   Violations: count triples containing i that become fully 1.
        # If x_i becomes 0:
        #   Size -1
        #   Violations: count triples containing i that were fully 1 and become not.
        
        # To make this fast, we maintain a count of "how many elements of each triple are currently 1".
        # triple_count[i] = number of 1s in triple i.
        # A triple is violated if triple_count[i] == 3.
        
        triple_counts = [0] * num_triples
        for i, tri in enumerate(forbidden_triples):
            c = 0
            for e in tri:
                if current_state[e-1]:
                    c += 1
            triple_counts[i] = c
            
        current_violations = sum(1 for c in triple_counts if c == 3)
        
        for step in range(1000000):
            if time.time() - start_time > time_limit:
                break
            
            # Choose a random index to flip
            i = random.randint(0, n - 1)
            
            # Calculate delta energy
            # Current state of i:
            x_i = current_state[i]
            
            if x_i == 1:
                # Flipping to 0
                # Size change: -1
                # Violation change: 
                # Triples containing i that are currently violated (count 3) will become count 2 (not violated).
                # Triples containing i that are count 2 will become count 1 (not violated, but were not violated before).
                # So only triples with count 3 matter for violation decrease.
                delta_viol = 0
                for t_idx in elem_to_triples[i]:
                    if triple_counts[t_idx] == 3:
                        delta_viol -= 1
                delta_energy = -(-1) + penalty * delta_viol # -size change + penalty * viol change
                # Wait. Energy = -Size + P*Viol.
                # New Size = Size - 1. Delta Size = -1. Delta Energy_Size = -(-1) = 1? No.
                # E_new - E_old = -(S-1) + P(V+v) - (-S + PV) = S - 1 + PV + Pv - S - PV = -1 + P*v.
                # So delta_energy = -1 + penalty * delta_viol.
                # Here delta_viol is negative (violations decrease).
                delta_energy = -1 + penalty * delta_viol
            else:
                # Flipping to 0 -> 1
                # Size change: +1
                # Violation change:
                # Triples containing i that are currently count 2 will become count 3 (violated).
                # Triples containing i that are count 3 are impossible if x_i=0 (max count 2).
                delta_viol = 0
                for t_idx in elem_to_triples[i]:
                    if triple_counts[t_idx] == 2:
                        delta_viol += 1
                # Delta Energy = -1 (size increase reduces energy by 1) + P * delta_viol
                delta_energy = -1 + penalty * delta_viol
            
            # Acceptance
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                # Apply move
                current_state[i] = 1 - current_state[i]
                current_size += 1 if current_state[i] else -1
                
                # Update triple counts
                for t_idx in elem_to_triples[i]:
                    if current_state[i]:
                        triple_counts[t_idx] += 1
                    else:
                        triple_counts[t_idx] -= 1
                    # Update violation count
                    if current_state[i]:
                        if triple_counts[t_idx] == 3:
                            current_violations += 1
                    else:
                        if triple_counts[t_idx] == 2: # Was 3, now 2
                            current_violations -= 1
                            
                current_energy += delta_energy
                
                # Update best local if feasible
                if current_violations == 0:
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_state = list(current_state)
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_local_state, best_local_size

    # Run multiple restarts
    # Restart 1: All 1s
    init1 = [1] * n
    state1, size1 = run_sa(init1, 10)
    if size1 > best_size:
        best_size = size1
        best_state = state1
        
    # Restart 2: Random
    init2 = [random.randint(0, 1) for _ in range(n)]
    state2, size2 = run_sa(init2, 10)
    if size2 > best_size:
        best_size = size2
        best_state = state2
        
    # Restart 3: Greedy Feasible
    # Start with all 1s, remove elements causing most violations
    init3 = [1] * n
    # Simple greedy cleanup
    state3 = list(init3)
    # Calculate violations
    # We can just use run_sa on init3, but maybe a better init is a "relaxed" greedy.
    # Let's just run SA on init3 again with different seed implicitly by random choices.
    state3, size3 = run_sa(init3, 10)
    if size3 > best_size:
        best_size = size3
        best_state = state3
        
    # If we still have time, run more random restarts
    while time.time() - start_time < time_budget - 5:
        init_r = [random.randint(0, 1) for _ in range(n)]
        # Bias towards 1s?
        init_r = [1 if random.random() < 0.7 else 0 for _ in range(n)]
        state_r, size_r = run_sa(init_r, 5)
        if size_r > best_size:
            best_size = size_r
            best_state = state_r
            
    if not best_state:
        # Fallback
        return [180]
        
    # Convert state to list of elements
    result = [i + 1 for i in range(n) if best_state[i]]
    return result
