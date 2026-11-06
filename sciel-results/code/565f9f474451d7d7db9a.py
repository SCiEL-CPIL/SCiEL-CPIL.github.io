"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 135.0
    
    target = set(range(1, 31))
    
    def get_all_diffs(elements):
        """Get all positive differences"""
        elems = sorted(elements)
        diffs = set()
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diffs.add(elems[j] - elems[i])
        return diffs
    
    def is_sidon(elements):
        """Check if elements form a Sidon set"""
        diffs = get_all_diffs(elements)
        # Check for duplicates by counting
        elems = sorted(elements)
        diff_list = []
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diff_list.append(elems[j] - elems[i])
        return len(diff_list) == len(set(diff_list))
    
    def objective(elements):
        """Lower is better: penalize missing target diffs and large max element"""
        if not is_sidon(elements):
            return float('inf')
        if 0 not in elements:
            return float('inf')
        diffs = get_all_diffs(elements)
        missing = len(target - diffs)
        max_val = max(elements)
        return missing * 1000 + max_val
    
    # Start with a reasonable initial set
    # Use a known construction or greedy initialization
    def greedy_init():
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            # Try to find a small x such that x-d >= 0 and all new diffs are unused
            found = False
            for x in range(d, 200):
                if x in a:
                    continue
                new_diffs = set(abs(x - y) for y in a)
                if new_diffs & used:
                    continue
                a.append(x)
                used |= new_diffs
                found = True
                break
            if not found:
                # Add a large element
                x = 2 * max(a) + d + 1
                a.append(x)
                used |= set(abs(x - y) for y in a)
        return a
    
    best_set = greedy_init()
    best_obj = objective(best_set)
    
    if best_obj < float('inf') and target.issubset(get_all_diffs(best_set)):
        return sorted(best_set)
    
    # Local search: try to improve by swapping or adding/removing elements
    current = list(best_set)
    current_obj = best_obj
    
    for iteration in range(100000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            if len(current) >= 32:
                continue
            new_elem = random.randint(1, 500)
            if new_elem in current:
                continue
            trial = current + [new_elem]
            trial_obj = objective(trial)
            if trial_obj < current_obj:
                current = trial
                current_obj = trial_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_set = list(current)
                    if target.issubset(get_all_diffs(current)):
                        return sorted(current)
        
        elif move_type == 'remove':
            if len(current) <= 8:
                continue
            idx = random.randint(0, len(current) - 1)
            if current[idx] == 0:
                continue
            trial = current[:idx] + current[idx+1:]
            trial_obj = objective(trial)
            if trial_obj < current_obj:
                current = trial
                current_obj = trial_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_set = list(current)
        
        elif move_type == 'swap':
            if len(current) < 2:
                continue
            idx = random.randint(0, len(current) - 1)
            if current[idx] == 0:
                continue
            old_elem = current[idx]
            new_elem = random.randint(1, 500)
            if new_elem in current:
                continue
            trial = current[:idx] + [new_elem] + current[idx+1:]
            trial_obj = objective(trial)
            if trial_obj < current_obj:
                current = trial
                current_obj = trial_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_set = list(current)
                    if target.issubset(get_all_diffs(current)):
                        return sorted(current)
    
    if best_obj < float('inf') and target.issubset(get_all_diffs(best_set)):
        return sorted(best_set)
    
    # Final fallback: return best found
    return sorted(best_set)

# EVOLVE-BLOCK-END
