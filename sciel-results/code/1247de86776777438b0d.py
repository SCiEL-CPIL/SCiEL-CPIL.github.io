import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Precompute all 4-APs in [1, n]
    # A 4-AP is defined by (a, d) -> (a, a+d, a+2d, a+3d)
    # We store them as a numpy array of indices for fast vectorized checking
    aps_list = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    if not aps_list:
        return list(range(1, n + 1))
        
    all_aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(all_aps)
    
    # Helper to check validity and count violations using vectorization
    # state: boolean array of size n+1 (index 0 unused)
    def get_violations(state):
        # Extract bits for all AP terms
        # state[all_aps] gives shape (num_aps, 4)
        # A violation exists if all 4 bits are 1
        terms = state[all_aps]
        # Sum along axis 1. If sum == 4, it's a violation.
        sums = np.sum(terms, axis=1)
        return np.sum(sums == 4)
    
    def is_valid(state):
        return get_violations(state) == 0

    # Initial state: Random subset with moderate density
    # Density of 4-AP free sets is roughly n^{-c}, but greedy gets ~100/512 ~ 0.2
    # Let's start with a random set of size ~150 to give SA room to move.
    initial_size = 150
    current_state = np.zeros(n + 1, dtype=np.int8)
    random.seed(42)
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates[:initial_size]:
        current_state[x] = 1
        
    current_violations = get_violations(current_state)
    current_size = np.sum(current_state)
    
    best_state = current_state.copy()
    best_size = current_size
    best_violations = current_violations
    
    # If initial is valid, it's a good start. If not, SA will fix it.
    # If initial is valid, we might want to start with a denser random set to break it?
    # No, starting valid is good. But if we start valid, we can't "accept" worse moves 
    # unless we allow temporary invalidity.
    
    # Simulated Annealing Parameters
    # We want to maximize size, subject to violations == 0.
    # Objective: Maximize (size - lambda * violations)
    # Or: Minimize (violations * penalty - size)
    # Let's use Energy = violations * 1000 - size. We want to minimize Energy.
    # But we only care about valid solutions (violations=0) for the final answer.
    
    T = 100.0  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.01
    
    # Move generators:
    # 1. Add a random unselected element
    # 2. Remove a random selected element
    # 3. Swap a selected with an unselected
    
    while time.time() < deadline and T > min_T:
        # Determine move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add an element
            unselected = np.where(current_state == 0)[0]
            unselected = unselected[unselected > 0] # Exclude index 0
            if len(unselected) == 0:
                T *= cooling_rate
                continue
            x = random.choice(unselected)
            
            new_state = current_state.copy()
            new_state[x] = 1
            
            new_violations = get_violations(new_state)
            new_size = current_size + 1
            
            delta_viol = new_violations - current_violations
            delta_size = 1
            
            # Energy change: We prefer lower violations and higher size.
            # Let's define Cost = 100 * violations - size
            delta_cost = 100 * delta_viol - delta_size
            
        elif move_type < 0.8:
            # Try to remove an element
            selected = np.where(current_state == 1)[0]
            if len(selected) == 0:
                T *= cooling_rate
                continue
            x = random.choice(selected)
            
            new_state = current_state.copy()
            new_state[x] = 0
            
            new_violations = get_violations(new_state)
            new_size = current_size - 1
            
            delta_viol = new_violations - current_violations
            delta_size = -1
            
            delta_cost = 100 * delta_viol - delta_size
            
        else:
            # Swap
            selected = np.where(current_state == 1)[0]
            unselected = np.where(current_state == 0)[0]
            unselected = unselected[unselected > 0]
            
            if len(selected) == 0 or len(unselected) == 0:
                T *= cooling_rate
                continue
                
            x_remove = random.choice(selected)
            x_add = random.choice(unselected)
            
            new_state = current_state.copy()
            new_state[x_remove] = 0
            new_state[x_add] = 1
            
            new_violations = get_violations(new_state)
            # Size stays same
            
            delta_viol = new_violations - current_violations
            delta_size = 0
            
            delta_cost = 100 * delta_viol - delta_size

        # Acceptance criterion
        if delta_cost < 0:
            # Better solution
            current_state = new_state
            current_violations = new_violations
            current_size = new_size
            
            if new_violations == 0 and new_size > best_size:
                best_state = new_state.copy()
                best_size = new_size
                best_violations = 0
        else:
            # Worse solution, accept with probability exp(-delta_cost / T)
            # delta_cost is positive here
            p = np.exp(-delta_cost / T)
            if random.random() < p:
                current_state = new_state
                current_violations = new_violations
                current_size = new_size
        
        T *= cooling_rate
        
        # Optional: If we find a valid solution, we can try to "polish" it with greedy adds
        # But SA should handle it.
        
    # Final Polish: Greedy add on the best valid state found
    # If best_state is valid, try to add more elements greedily
    if best_violations == 0:
        final_state = best_state.copy()
        final_size = best_size
        # Try to add any remaining elements that don't create violations
        unselected = np.where(final_state == 0)[0]
        unselected = unselected[unselected > 0]
        random.shuffle(unselected)
        
        for x in unselected:
            if time.time() > deadline - 2:
                break
            test_state = final_state.copy()
            test_state[x] = 1
            if get_violations(test_state) == 0:
                final_state = test_state
                final_size += 1
                
        if final_size > best_size:
            best_state = final_state
            best_size = final_size
            best_violations = 0
            
    # If no valid solution was found (best_violations > 0), we must return a valid one.
    # The problem requires a valid set.
    # Fallback: Use the current state if valid, or find any valid subset.
    # A simple fallback is the greedy seed from the original code, or just the empty set (non-empty required).
    # Let's ensure we return a valid set.
    
    if best_violations != 0:
        # This shouldn't happen often if T is low enough at the end, 
        # but if it does, we need a valid set.
        # We can try to fix the current state by removing elements causing violations.
        # Or just use a small valid set like {1, 2, 4, 8...} powers of 2 are 4-AP free?
        # No, 1, 2, 4, 8 is not a 4-AP. But 1, 2, 3, 4 is.
        # Powers of 2: 1, 2, 4, 8, 16...
        # AP: 1, 2, 3, 4? No.
        # 1, 3, 5, 7? No.
        # Actually, any set with no 3-AP is 4-AP free.
        # Cantor set is 3-AP free.
        
        # Let's generate a small Cantor set as a fallback
        fallback = set()
        # Base 3, no digit 2
        # 1, 2 (no), 3 (10), 4 (11), 6 (20 no), 7 (21 no), 9 (100), 10 (101), 12 (110), 13 (111)...
        # Wait, 1 is 1, 2 is 2 (digit 2? no, 2 is '2' in base 3? Yes. 2 in base 3 is '2'. So 2 is excluded.)
        # 1 (1), 3 (10), 4 (11), 9 (100), 10 (101), 12 (110), 13 (111)...
        # Let's just use a simple valid set: {1} is valid.
        # But we want to maximize.
        # Let's try to repair the current state.
        repair_state = current_state.copy()
        repair_violations = get_violations(repair_state)
        
        # Iteratively remove one element from each violating AP until valid
        # This is O(V * N) which is fine.
        max_iter = 1000
        for _ in range(max_iter):
            if repair_violations == 0:
                break
            if time.time() > deadline - 1:
                break
            # Find a violating AP
            terms = repair_state[all_aps]
            sums = np.sum(terms, axis=1)
            viol_idx = np.where(sums == 4)[0]
            if len(viol_idx) == 0:
                break
            # Pick a random violating AP
            v_idx = random.choice(viol_idx)
            ap = all_aps[v_idx]
            # Remove a random element from this AP
            elem_to_remove = random.choice(ap)
            repair_state[elem_to_remove] = 0
            repair_violations = get_violations(repair_state)
            
        if repair_violations == 0:
            best_state = repair_state
            best_size = np.sum(repair_state)
            best_violations = 0
        else:
            # Absolute fallback: {1}
            best_state = np.zeros(n + 1, dtype=np.int8)
            best_state[1] = 1
            best_size = 1
            best_violations = 0

    return [i for i in range(1, n + 1) if best_state[i] == 1]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Quick check
    s = set(res)
    valid = True
    for a in s:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in s and (a+2*d) in s and (a+3*d) in s:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
