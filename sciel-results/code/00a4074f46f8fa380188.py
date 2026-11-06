import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial set
    current_set = set(random.sample(range(1, 513), 32))
    current_list = list(current_set)
    present = [False] * 513
    for x in current_set:
        present[x] = True
        
    max_diff = 512
    count = [0] * (max_diff + 1)
    
    # Helper to calculate score from scratch (slow, use for init)
    def calc_score():
        for d in range(1, max_diff + 1):
            count[d] = 0
        score = 0
        # O(N^2) check
        for i in range(32):
            a = current_list[i]
            for j in range(i+1, 32):
                b = current_list[j]
                d = b - a
                if a + 2*d <= 512 and present[a + 2*d]:
                    count[d] += 1
        for d in range(1, max_diff + 1):
            if count[d] > 0:
                score += 1
        return score

    current_score = calc_score()
    best_score = current_score
    best_list = list(current_list)
    
    T = 1.0
    cooling = 0.99995 # Slow cooling
    
    while time.time() < deadline and T > 0.01:
        old_score = current_score
        
        # Pick random index to replace
        idx = random.randint(0, 31)
        old_val = current_list[idx]
        
        # Pick random new val
        new_val = random.randint(1, 512)
        if present[new_val]:
            continue
            
        # Remove old_val
        delta = 0
        present[old_val] = False
        for d in range(1, 512):
            # Check if old_val was part of AP with diff d
            # 1. start
            if old_val + 2*d <= 512:
                if present[old_val + d] and present[old_val + 2*d]:
                    count[d] -= 1
                    if count[d] == 0: delta -= 1
            # 2. middle
            if old_val - d >= 1 and old_val + d <= 512:
                if present[old_val - d] and present[old_val + d]:
                    count[d] -= 1
                    if count[d] == 0: delta -= 1
            # 3. end
            if old_val - 2*d >= 1:
                if present[old_val - 2*d] and present[old_val - d]:
                    count[d] -= 1
                    if count[d] == 0: delta -= 1
        current_score += delta
        
        # Add new_val
        present[new_val] = True
        for d in range(1, 512):
            # Check if new_val forms AP with diff d
            # 1. start
            if new_val + 2*d <= 512:
                if present[new_val + d] and present[new_val + 2*d]:
                    count[d] += 1
                    if count[d] == 1: delta += 1
            # 2. middle
            if new_val - d >= 1 and new_val + d <= 512:
                if present[new_val - d] and present[new_val + d]:
                    count[d] += 1
                    if count[d] == 1: delta += 1
            # 3. end
            if new_val - 2*d >= 1:
                if present[new_val - 2*d] and present[new_val - d]:
                    count[d] += 1
                    if count[d] == 1: delta += 1
        current_score += delta
        
        # Accept/Reject
        if current_score > old_score:
            # Always accept improvement
            pass
        else:
            # Accept with probability exp((new - old) / T)
            if random.random() < pow(2.718281828, (current_score - old_score) / T):
                pass
            else:
                # Revert
                # Remove new_val
                present[new_val] = False
                for d in range(1, 512):
                    if new_val + 2*d <= 512:
                        if present[new_val + d] and present[new_val + 2*d]:
                            count[d] -= 1
                            if count[d] == 0: delta -= 1 # Wait, delta logic is tricky for revert
                    # ... This revert logic is duplicating code and prone to errors.
                    # Better to just store the changes or re-calculate?
                    # Re-calculation is O(N^2) = 1024. 
                    # If we revert often, this might be slow.
                    # Let's stick to Hill Climbing first, it's simpler and often sufficient for these small N.
                    # Or, implement proper undo.
                
                # For now, let's just do Hill Climbing (accept only improvements) to ensure correctness and speed.
                # If we get stuck, we can add restarts.
                
                # REVERT
                # Remove new_val
                revert_delta = 0
                for d in range(1, 512):
                    if new_val + 2*d <= 512:
                        if present[new_val + d] and present[new_val + 2*d]:
                            count[d] -= 1
                            if count[d] == 0: revert_delta -= 1
                    if new_val - d >= 1 and new_val + d <= 512:
                        if present[new_val - d] and present[new_val + d]:
                            count[d] -= 1
                            if count[d] == 0: revert_delta -= 1
                    if new_val - 2*d >= 1:
                        if present[new_val - 2*d] and present[new_val - d]:
                            count[d] -= 1
                            if count[d] == 0: revert_delta -= 1
                current_score += revert_delta
                present[new_val] = False
                
                # Add old_val back
                for d in range(1, 512):
                    if old_val + 2*d <= 512:
                        if present[old_val + d] and present[old_val + 2*d]:
                            count[d] += 1
                            if count[d] == 1: revert_delta += 1
                    if old_val - d >= 1 and old_val + d <= 512:
                        if present[old_val - d] and present[old_val + d]:
                            count[d] += 1
                            if count[d] == 1: revert_delta += 1
                    if old_val - 2*d >= 1:
                        if present[old_val - 2*d] and present[old_val - d]:
                            count[d] += 1
                            if count[d] == 1: revert_delta += 1
                current_score += revert_delta
                present[old_val] = True
                current_list[idx] = old_val
                continue # Skip update of best
        
        # Update list
        current_list[idx] = new_val
        
        if current_score > best_score:
            best_score = current_score
            best_list = list(current_list)
            
        T *= cooling
        
    return sorted(best_list)
