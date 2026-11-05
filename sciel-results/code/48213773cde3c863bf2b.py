import numpy as np
import time

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    if not aps:
        return list(range(1, n + 1))

    # Map index to list of AP indices containing it
    # This allows incremental violation counting
    ap_indices_by_elem = {i: [] for i in range(1, n + 1)}
    for i, ap in enumerate(aps):
        for elem in ap:
            ap_indices_by_elem[elem].append(i)
            
    # Initial state: Greedy
    sel_set = set()
    for x in range(1, n + 1):
        is_valid = True
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in sel_set and (x - 2 * d) in sel_set and (x - 3 * d) in sel_set:
                is_valid = False
                break
        if is_valid:
            sel_set.add(x)
            
    state = np.zeros(n + 1, dtype=bool)
    for x in sel_set:
        state[x] = True
        
    # Count initial violations (should be 0 for greedy)
    violations = 0
    for ap in aps:
        if all(state[elem] for elem in ap):
            violations += 1
            
    size = len(sel_set)
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.99995
    lambda_pen = 50.0
    
    current_energy = lambda_pen * violations - size
    best_state = state.copy()
    best_energy = current_energy
    
    start_time = time.time()
    timeout = 85.0
    
    while time.time() - start_time < timeout:
        idx = np.random.randint(1, n + 1)
        
        old_val = state[idx]
        new_val = not old_val
        
        # Calculate change in violations
        viol_delta = 0
        for ap_idx in ap_indices_by_elem[idx]:
            ap = aps[ap_idx]
            # Check if the other 3 elements are in state
            # We need to be careful: state[idx] is still old_val
            # So "other 3 in state" means they are True
            others = [elem for elem in ap if elem != idx]
            others_in_state = all(state[elem] for elem in others)
            
            was_satisfied = old_val and others_in_state
            is_satisfied = new_val and others_in_state
            
            if was_satisfied and not is_satisfied:
                viol_delta -= 1
            elif not was_satisfied and is_satisfied:
                viol_delta += 1
                
        new_violations = violations + viol_delta
        size_delta = 1 if new_val else -1
        new_size = size + size_delta
        new_energy = lambda_pen * new_violations - new_size
        
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            state[idx] = new_val
            violations = new_violations
            size = new_size
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()
        else:
            # Revert (state[idx] was not changed, so no revert needed if we didn't modify it)
            # Wait, I didn't modify state[idx] yet in the calculation above.
            # I only used old_val and new_val.
            # So if rejected, state remains unchanged.
            pass
            
        T *= alpha
        if T < T_min:
            T = T_min

    result = [i for i in range(1, n + 1) if best_state[i]]
    if not result:
        return [1]
    return result
