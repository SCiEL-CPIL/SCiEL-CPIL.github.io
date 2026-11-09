# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element.
    """
    
    REQUIRED_DIFFS = set(range(1, 31))
    MAX_SIZE = 32
    MIN_SIZE = 9
    
    def get_diffs(elements):
        diffs = []
        n = len(elements)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(elements[j] - elements[i])
        return diffs

    def is_valid_set(elements):
        if not elements:
            return False
        diffs = get_diffs(elements)
        if len(diffs) != len(set(diffs)):
            return False
        diff_set = set(diffs)
        return REQUIRED_DIFFS.issubset(diff_set)

    def evaluate(elements):
        if not elements:
            return float('inf')
        
        diffs = get_diffs(elements)
        diff_set = set(diffs)
        
        # Penalty for duplicates
        dup_count = len(diffs) - len(diff_set)
        penalty = 10000 * dup_count
        
        # Penalty for missing diffs
        missing = REQUIRED_DIFFS - diff_set
        penalty += 100 * len(missing)
        
        # Objective: Minimize max element
        max_val = max(elements)
        penalty += max_val
        
        return penalty

    # --- Phase 1: Find a valid set using Simulated Annealing ---
    best_elements = None
    best_penalty = float('inf')
    
    start_time = time.time()
    time_limit = 110.0  # Use most of the time for optimization
    
    # Initialize with a few random restarts to find a valid set quickly
    # Then optimize
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        size = random.randint(9, 12)
        elems = [0]
        candidates = list(range(1, 150))
        random.shuffle(candidates)
        for _ in range(size - 1):
            elems.append(candidates[_])
        elems.sort()
        
        current_penalty = evaluate(elems)
        
        # If already valid, record
        if is_valid_set(elems):
            if max(elems) < (max(best_elements) if best_elements else float('inf')):
                best_elements = elems
                best_penalty = current_penalty
        
        # Simulated Annealing
        temp = 100.0
        cooling = 0.99
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            # Generate neighbor
            move_type = random.choice(['swap', 'shrink', 'add', 'remove'])
            new_elems = list(elems)
            
            if move_type == 'swap' and len(new_elems) > 1:
                idx = random.randint(1, len(new_elems)-1)
                old_val = new_elems[idx]
                new_val = random.randint(1, 200)
                if new_val not in new_elems:
                    new_elems[idx] = new_val
                    new_elems.sort()
                    
            elif move_type == 'shrink' and len(new_elems) > 1:
                idx = random.randint(1, len(new_elems)-1)
                old_val = new_elems[idx]
                new_val = random.randint(1, max(1, old_val - 1))
                if new_val not in new_elems:
                    new_elems[idx] = new_val
                    new_elems.sort()
                    
            elif move_type == 'add' and len(new_elems) < MAX_SIZE:
                new_val = random.randint(1, 200)
                if new_val not in new_elems:
                    new_elems.append(new_val)
                    new_elems.sort()
                    
            elif move_type == 'remove' and len(new_elems) > MIN_SIZE:
                idx = random.randint(1, len(new_elems)-1)
                new_elems.pop(idx)

            new_penalty = evaluate(new_elems)
            
            # Acceptance criterion
            delta = new_penalty - current_penalty
            if delta < 0 or (temp > 1 and random.random() < pow(2.718, -delta / temp)):
                elems = new_elems
                current_penalty = new_penalty
                
            temp *= cooling
            
            # Check if valid and better
            if is_valid_set(elems):
                if max(elems) < (max(best_elements) if best_elements else float('inf')):
                    best_elements = elems
                    best_penalty = current_penalty

    if best_elements is None:
        # Fallback: Greedy
        best_elements = [0]
        used_diffs = set()
        needed = set(REQUIRED_DIFFS)
        
        for _ in range(MAX_SIZE - 1):
            if not needed:
                break
            best_x = None
            best_score = float('inf')
            for x in range(1, 500):
                if x in best_elements:
                    continue
                new_diffs = []
                valid = True
                for e in best_elements:
                    d = abs(x - e)
                    if d in used_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                covers = sum(1 for d in new_diffs if d in needed)
                if covers == 0:
                    continue
                score = (-covers, x)
                if score < best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            best_elements.append(best_x)
            best_elements.sort()
            for e in best_elements[:-1]:
                d = abs(best_x - e)
                used_diffs.add(d)
                needed.discard(d)
                
        if not is_valid_set(best_elements):
            # If still invalid, return the best effort
            pass

    return best_elements

# EVOLVE-BLOCK-END
