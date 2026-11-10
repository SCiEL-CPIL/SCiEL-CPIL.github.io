import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-APs (a, b, c, d) with a < b < c < d, d-a = 3*step
    # Store as list of tuples of 0-indexed positions
    aps = []
    for a in range(N):
        for step in range(1, (N - 1 - a) // 3 + 1):
            b = a + step
            c = a + 2 * step
            d = a + 3 * step
            if d < N:
                aps.append((a, b, c, d))
    
    num_aps = len(aps)
    # Convert to numpy array for fast vectorized operations
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Precompute for each index, which APs contain it
    # For fast incremental cost updates
    index_to_aps = [[] for _ in range(N)]
    for i, (a, b, c, d) in enumerate(aps):
        index_to_aps[a].append(i)
        index_to_aps[b].append(i)
        index_to_aps[c].append(i)
        index_to_aps[d].append(i)
    
    def count_violations(state):
        """Count how many 4-APs have all 4 elements in state."""
        if num_aps == 0:
            return 0
        # state is a boolean numpy array of length N
        # For each AP, check if all 4 positions are True
        # ap_array has shape (num_aps, 4)
        vals = state[ap_array]  # shape (num_aps, 4)
        return int(np.all(vals, axis=1).sum())
    
    def greedy_fill(order, state):
        """Fill state greedily in given order, maintaining feasibility."""
        size = 0
        for idx in order:
            if state[idx]:
                continue
            # Check if adding idx creates any violation
            # Only need to check APs containing idx
            can_add = True
            for ap_i in index_to_aps[idx]:
                a, b, c, d = aps[ap_i]
                # Check if the other 3 are already in state
                others = [x for x in (a, b, c, d) if x != idx]
                if all(state[x] for x in others):
                    can_add = False
                    break
            if can_add:
                state[idx] = True
                size += 1
        return size
    
    best_state = None
    best_size = 0
    best_violations = float('inf')
    
    # Phase 1: Multiple greedy initializations to find good feasible starts
    orders = []
    orders.append(list(range(N - 1, -1, -1)))  # Reverse
    orders.append(list(range(N)))  # Forward
    for _ in range(8):
        r = list(range(N))
        random.shuffle(r)
        orders.append(r)
    
    for order in orders:
        if time.time() > deadline:
            break
        state = np.zeros(N, dtype=bool)
        size = greedy_fill(order, state)
        if size > best_size:
            best_size = size
            best_state = state.copy()
    
    # Phase 2: Max-SAT style stochastic local search (WalkSAT-like)
    # Objective: maximize size - lambda * violations
    # We'll use a two-phase approach:
    # 1. Try to add elements (increase size)
    # 2. If stuck, try to flip elements (accept some violations temporarily)
    # 3. Periodically try to remove violations
    
    if best_state is not None:
        state = best_state.copy()
        size = int(state.sum())
        violations = count_violations(state)
        
        # Track best
        if size > best_size or (size == best_size and violations < best_violations):
            best_size = size
            best_violations = violations
            best_state = state.copy()
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                # Restart from best
                state = best_state.copy()
                size = int(state.sum())
                violations = count_violations(state)
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5)
                continue
            
            # Choose a random action
            action = random.random()
            
            if action < 0.6:
                # Try to add a random element
                idx = random.randint(0, N - 1)
                if state[idx]:
                    no_improve_count += 1
                    continue
                # Check if adding idx creates violations
                new_violations = 0
                for ap_i in index_to_aps[idx]:
                    a, b, c, d = aps[ap_i]
                    others = [x for x in (a, b, c, d) if x != idx]
                    if all(state[x] for x in others):
                        new_violations += 1
                # Accept if it doesn't create too many violations
                if new_violations <= 1:
                    state[idx] = True
                    size += 1
                    violations += new_violations
                    if size > best_size or (size == best_size and violations < best_violations):
                        best_size = size
                        best_violations = violations
                        best_state = state.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            elif action < 0.85:
                # Try to remove a random element (to reduce violations)
                if size == 0:
                    no_improve_count += 1
                    continue
                # Find elements that are part of violations
                # Simply pick random element from state
                filled = np.where(state)[0]
                if len(filled) == 0:
                    no_improve_count += 1
                    continue
                idx = random.choice(filled)
                # Count how many violations this element is part of
                old_violations = 0
                for ap_i in index_to_aps[idx]:
                    a, b, c, d = aps[ap_i]
                    others = [x for x in (a, b, c, d) if x != idx]
                    if all(state[x] for x in others):
                        old_violations += 1
                if old_violations > 0:
                    state[idx] = False
                    size -= 1
                    violations -= old_violations
                    if size > best_size or (size == best_size and violations < best_violations):
                        best_size = size
                        best_violations = violations
                        best_state = state.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Random removal to escape local minima
                    if random.random() < 0.1:
                        state[idx] = False
                        size -= 1
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                        
            else:
                # 2-exchange: remove one, add another
                if size == 0:
                    no_improve_count += 1
                    continue
                filled = np.where(state)[0]
                if len(filled) == 0:
                    no_improve_count += 1
                    continue
                remove_idx = random.choice(filled)
                add_idx = random.randint(0, N - 1)
                if state[add_idx]:
                    no_improve_count += 1
                    continue
                # Check net change in violations
                # Remove remove_idx, add add_idx
                # This is complex; approximate by checking each independently
                # Count violations involving remove_idx
                remove_viol = 0
                for ap_i in index_to_aps[remove_idx]:
                    a, b, c, d = aps[ap_i]
                    others = [x for x in (a, b, c, d) if x != remove_idx]
                    if all(state[x] for x in others):
                        remove_viol += 1
                # Count violations that would be created by add_idx
                # (after removing remove_idx)
                add_viol = 0
                for ap_i in index_to_aps[add_idx]:
                    a, b, c, d = aps[ap_i]
                    others = [x for x in (a, b, c, d) if x != add_idx]
                    # Check if all others are in state, considering remove_idx might be gone
                    all_in = True
                    for x in others:
                        if x == remove_idx:
                            continue  # Will be removed
                        if not state[x]:
                            all_in = False
                            break
                    if all_in:
                        add_viol += 1
                # Net change in size: 0 (remove one, add one)
                # Net change in violations: add_viol - remove_viol
                if add_viol <= remove_viol:
                    state[remove_idx] = False
                    state[add_idx] = True
                    violations += add_viol - remove_viol
                    if size > best_size or (size == best_size and violations < best_violations):
                        best_size = size
                        best_violations = violations
                        best_state = state.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
    
    # Phase 3: Final greedy refill on the best state found
    if best_state is not None:
        # Try to add any element that doesn't create violations
        state = best_state.copy()
        size = int(state.sum())
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if state[idx]:
                    continue
                can_add = True
                for ap_i in index_to_aps[idx]:
                    a, b, c, d = aps[ap_i]
                    others = [x for x in (a, b, c, d) if x != idx]
                    if all(state[x] for x in others):
                        can_add = False
                        break
                if can_add:
                    state[idx] = True
                    size += 1
                    improved = True
        if size > best_size:
            best_size = size
            best_state = state
    
    # Verify the best state has no violations
    if best_state is not None:
        v = count_violations(best_state)
        if v > 0:
            # Remove elements to fix violations
            # Greedy: remove one element from each violated AP
            state = best_state.copy()
            while time.time() < deadline:
                v = count_violations(state)
                if v == 0:
                    break
                # Find a violated AP
                vals = state[ap_array]
                violated_mask = np.all(vals, axis=1)
                if not violated_mask.any():
                    break
                viol_idx = np.where(violated_mask)[0][0]
                a, b, c, d = aps[viol_idx]
                # Remove the element that is in the most APs (to fix more violations)
                counts = [len(index_to_aps[x]) for x in (a, b, c, d)]
                max_i = counts.index(max(counts))
                state[(a, b, c, d)[max_i]] = False
            
            size = int(state.sum())
            if size > best_size:
                best_size = size
                best_state = state
    
    # Convert to list of 1-indexed values
    if best_state is None:
        return [1]
    
    result = [i + 1 for i in range(N) if best_state[i]]
    return sorted(result)
