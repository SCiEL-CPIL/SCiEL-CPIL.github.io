import random
import time

def construct():
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds safety margin
    
    best_score = -1
    best_sets = None
    
    def is_comparable(a, b):
        # Distinct masks a, b
        # Strict subset relation exists if one is subset of the other
        return ((a & b) == a) or ((a & b) == b)

    def calc_score(sets):
        score = 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                if is_comparable(sets[i], sets[j]):
                    score += 1
        return score

    def generate_initial_sets():
        # Base: B_5 (32 sets)
        base_sets = list(range(32)) # 0 to 31
        
        # Extensions: 8 sets
        extra_sets = []
        for _ in range(8):
            # Start with 31 (11111)
            mask = 31
            # Add 1 to 3 random bits from 5 to 11
            for _ in range(random.randint(1, 3)):
                bit = random.randint(5, 11)
                mask |= (1 << bit)
            extra_sets.append(mask)
            
        all_sets = base_sets + extra_sets
        unique_sets = list(set(all_sets))
        
        while len(unique_sets) < m:
            r = random.randint(0, total_masks - 1)
            if r not in unique_sets:
                unique_sets.append(r)
                
        return unique_sets[:m]

    # We will run a few restarts.
    # To ensure we use time effectively, we run restarts until deadline.
    
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        
        current_sets = generate_initial_sets()
        current_mask_set = set(current_sets)
        current_score = calc_score(current_sets)
        
        if current_score > best_score:
            best_score = current_score
            best_sets = current_sets[:]
            
        # Allocate time for this restart
        # Simple heuristic: give each restart 10 seconds, or remaining time if less
        time_for_restart = 10.0
        if time.time() + time_for_restart > deadline - 2.0:
            time_for_restart = deadline - 2.0 - time.time()
            
        restart_deadline = time.time() + time_for_restart
        no_improve_count = 0
        
        while time.time() < restart_deadline:
            # Pick random index
            i = random.randint(0, m - 1)
            old_mask = current_sets[i]
            
            # Pick random candidate
            new_mask = random.randint(0, total_masks - 1)
            
            if new_mask == old_mask:
                continue
                
            if new_mask in current_mask_set:
                continue
                
            # Calculate delta
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                other = current_sets[j]
                
                if is_comparable(old_mask, other):
                    delta -= 1
                    
                if is_comparable(new_mask, other):
                    delta += 1
            
            if delta > 0:
                current_sets[i] = new_mask
                current_mask_set.remove(old_mask)
                current_mask_set.add(new_mask)
                current_score += delta
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_sets = current_sets[:]
            else:
                no_improve_count += 1
                if no_improve_count > 2000:
                    # Force a move to escape local optimum
                    current_sets[i] = new_mask
                    current_mask_set.remove(old_mask)
                    current_mask_set.add(new_mask)
                    current_score += delta
                    no_improve_count = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_sets = current_sets[:]

    # Convert best_sets to list of lists of ints
    if best_sets is None:
        best_sets = generate_initial_sets()
        
    result = []
    for mask in best_sets:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
