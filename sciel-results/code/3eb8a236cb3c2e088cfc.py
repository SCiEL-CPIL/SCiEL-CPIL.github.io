import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Greedy Construction
    chosen = []
    counts = {}
    
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        new_sums = [x + y for y in chosen] + [2 * x]
        for s in new_sums:
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
            
    best_size = len(chosen)
    best_chosen = list(chosen)
    
    # 2. Local Search
    current_chosen = list(chosen)
    current_counts = dict(counts)
    used = set(current_chosen)
    unused = list(set(range(1, 1001)) - used)
    
    random.seed(42)
    
    # Helper to check if adding x is valid given current state
    def can_add(x, ch, cnt):
        for y in ch:
            s = x + y
            if cnt.get(s, 0) + 1 > 2:
                return False
        s = 2 * x
        if cnt.get(s, 0) + 1 > 2:
            return False
        return True

    # Helper to check if swapping y with x is valid
    def can_swap(y, x, ch, cnt):
        # Sums removed: y+z for z in ch, 2y
        # Sums added: x+z for z in ch (z!=y), 2x
        
        # We need to check if any sum exceeds 2 after swap.
        # A sum s is affected if s is in RemovedSums or AddedSums.
        
        # Instead of building sets, we can just check the constraints for the new sums
        # and ensure that removing y doesn't drop any count below 0 (which it won't if y is in ch).
        # The risk is that adding x causes a sum to exceed 2.
        # Also, removing y might "free up" capacity, but we only care about max count <= 2.
        
        # So, we just need to check if adding x to (ch - {y}) is valid.
        # Let ch_prime = ch - {y}.
        # We need to check if x can be added to ch_prime.
        
        # However, the counts in `cnt` reflect `ch`.
        # So we need to simulate the removal of y to get `cnt_prime`, then check x against `cnt_prime`.
        
        # To avoid modifying `cnt` repeatedly, we can calculate the delta.
        # But simpler: just iterate through ch_prime and check against `cnt` adjusted for y's removal.
        
        # Count of s in ch_prime = cnt[s] - (1 if s in sums(y) else 0)
        
        # Sums of y: y+z for z in ch, 2y.
        # Note: y+y = 2y. y+z = z+y.
        
        # Let's just do the check directly.
        for z in ch:
            if z == y:
                continue
            s = x + z
            # What is the count of s in ch_prime?
            # It is cnt[s] minus (1 if s is a sum involving y in ch else 0).
            # s involves y if s == y + z' for some z' in ch.
            # Since z != y, s = x + z.
            # Does y contribute to s?
            # y contributes to s if s = y + w for some w in ch.
            # So if x + z = y + w, then w = x + z - y.
            # If w is in ch, then y contributed to s.
            
            # This is getting complicated.
            
            # Alternative: Just compute the new counts for the affected sums.
            # Affected sums are those involving y or x.
            # Sums involving y: {y+z | z in ch} U {2y}
            # Sums involving x (in new set): {x+z | z in ch, z!=y} U {2x}
            
            # Let's just build a dict of deltas.
            deltas = {}
            
            # Remove y
            for z in ch:
                s = y + z
                deltas[s] = deltas.get(s, 0) - 1
            s = 2 * y
            deltas[s] = deltas.get(s, 0) - 1
            
            # Add x
            for z in ch:
                if z == y:
                    continue
                s = x + z
                deltas[s] = deltas.get(s, 0) + 1
            s = 2 * x
            deltas[s] = deltas.get(s, 0) + 1
            
            # Check if any new count > 2
            for s, d in deltas.items():
                new_count = cnt.get(s, 0) + d
                if new_count > 2:
                    return False
                if new_count < 0:
                    return False # Should not happen
            
            return True

    while time.time() < deadline:
        # Try to add
        if unused:
            idx = random.randint(0, len(unused) - 1)
            x = unused[idx]
            if can_add(x, current_chosen, current_counts):
                # Accept
                for y in current_chosen:
                    s = x + y
                    current_counts[s] = current_counts.get(s, 0) + 1
                s = 2 * x
                current_counts[s] = current_counts.get(s, 0) + 1
                current_chosen.append(x)
                used.add(x)
                unused.pop(idx)
                
                if len(current_chosen) > best_size:
                    best_size = len(current_chosen)
                    best_chosen = list(current_chosen)
        else:
            # No unused elements? Then we can't add.
            # This shouldn't happen unless we have 1000 elements.
            break

        # Try to swap
        if current_chosen and unused:
            y_idx = random.randint(0, len(current_chosen) - 1)
            y = current_chosen[y_idx]
            x_idx = random.randint(0, len(unused) - 1)
            x = unused[x_idx]
            
            if can_swap(y, x, current_chosen, current_counts):
                # Perform swap
                # Update counts
                deltas = {}
                for z in current_chosen:
                    s = y + z
                    deltas[s] = deltas.get(s, 0) - 1
                s = 2 * y
                deltas[s] = deltas.get(s, 0) - 1
                
                for z in current_chosen:
                    if z == y:
                        continue
                    s = x + z
                    deltas[s] = deltas.get(s, 0) + 1
                s = 2 * x
                deltas[s] = deltas.get(s, 0) + 1
                
                for s, d in deltas.items():
                    current_counts[s] = current_counts.get(s, 0) + d
                    if current_counts[s] == 0:
                        del current_counts[s]
                    
                # Update sets
                current_chosen[y_idx] = x
                used.remove(y)
                used.add(x)
                unused[x_idx] = y

    return best_chosen
