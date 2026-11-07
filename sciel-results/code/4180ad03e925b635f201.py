import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute sum_mask[x]: bit y is set iff x + y <= N
    sum_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        m = 0
        for y in range(1, N + 1 - x + 1):
            m |= (1 << y)
        sum_mask[x] = m
    
    def is_valid_bitset(mask):
        """Check if no three distinct a,b,c in mask have all pair sums in mask."""
        # For each a < b with a+b in mask, check if exists c > b with a+c and b+c in mask
        # (mask & sum_mask[a]) has bits y where a+y is in mask
        # (mask & sum_mask[b]) has bits y where b+y is in mask
        # Intersection has bits y where BOTH a+y and b+y are in mask
        # We need y > b (i.e., y >= b+1)
        for a in range(1, N + 1):
            if not (mask >> a) & 1:
                continue
            ma = mask & sum_mask[a]  # bits y where a+y is in mask
            for b in range(a + 1, N + 1):
                if not (mask >> b) & 1:
                    continue
                s_ab = a + b
                if s_ab > N:
                    break
                if not (mask >> s_ab) & 1:
                    continue
                # Check if there exists c > b with a+c and b+c both in mask
                mb = mask & sum_mask[b]
                common = ma & mb
                # We need a bit c > b set in common
                if common >> (b + 1):
                    return False
        return True

    def is_valid_set(s_set):
        """Wrapper for set input."""
        mask = 0
        for x in s_set:
            mask |= (1 << x)
        return is_valid_bitset(mask)

    # Initial candidate: upper half
    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)
    
    random.seed(42)
    
    def local_search(current_set, max_steps=20000):
        current_mask = 0
        for x in current_set:
            current_mask |= (1 << x)
        
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            improved = False
            
            # Try adding elements
            candidates = []
            for x in range(1, N + 1):
                if not (current_mask >> x) & 1:
                    candidates.append(x)
            
            random.shuffle(candidates)
            
            for elem in candidates:
                trial_mask = current_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    current_mask = trial_mask
                    improved = True
                    break
            
            if improved:
                continue
                
            # If no single addition works, try swaps (remove 1, add 1)
            if current_mask == 0:
                break
            
            current_elements = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
            if not current_elements:
                break
            to_remove = random.choice(current_elements)
            
            base_mask = current_mask & ~(1 << to_remove)
            
            # Try to add 1 element to base_mask (swap)
            candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
            random.shuffle(candidates)
            
            swap_found = False
            for elem in candidates:
                trial_mask = base_mask | (1 << elem)
                if is_valid_bitset(trial_mask):
                    current_mask = trial_mask
                    swap_found = True
                    break
            
            if not swap_found:
                # Try 2-for-1 swap: remove 1, add 2
                candidates = [x for x in range(1, N + 1) if x != to_remove and not (base_mask >> x) & 1]
                random.shuffle(candidates)
                
                new_mask = base_mask
                added = 0
                for elem in candidates:
                    if added == 2:
                        break
                    trial_mask = new_mask | (1 << elem)
                    if is_valid_bitset(trial_mask):
                        new_mask = trial_mask
                        added += 1
                
                if added == 2:
                    current_mask = new_mask
                    improved = True
        
        # Convert back to set
        result_set = set()
        for x in range(1, N + 1):
            if (current_mask >> x) & 1:
                result_set.add(x)
        return result_set

    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        # Different initial sets
        if restart == 0:
            init = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            # Try a mix: upper half plus some lower
            init = set(range(N // 2 + 1, N + 1))
            lower_candidates = list(range(1, N // 2 + 1))
            random.shuffle(lower_candidates)
            for x in lower_candidates:
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        elif restart % 3 == 2:
            # Random subset biased toward upper half
            init = set()
            for x in range(N // 2 + 1, N + 1):
                if random.random() < 0.95:
                    init.add(x)
            for x in range(1, N // 2 + 1):
                if random.random() < 0.3:
                    init.add(x)
            
            # Ensure validity by removing random elements
            while not is_valid_set(init):
                if len(init) == 0:
                    break
                to_remove = random.choice(list(init))
                init.remove(to_remove)
        else:
            # Greedy random construction
            init = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                trial = init | {x}
                if is_valid_set(trial):
                    init = trial
        
        # Run local search
        improved_set = local_search(init)
        
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Final polish: try to add any remaining element (exhaustive)
    remaining = set(range(1, N + 1)) - best_set
    for elem in sorted(remaining):
        if time.time() > deadline - 2:
            break
        trial = best_set | {elem}
        if is_valid_set(trial):
            best_set = trial
    
    # Second polish pass after first additions
    remaining = set(range(1, N + 1)) - best_set
    for elem in sorted(remaining):
        if time.time() > deadline - 1:
            break
        trial = best_set | {elem}
        if is_valid_set(trial):
            best_set = trial
    
    # Ensure return type is correct
    result = list(best_set)
    result = [int(x) for x in result]
    return result
