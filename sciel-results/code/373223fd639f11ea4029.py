import time
import random

def construct():
    N = 256
    # Greedy initialization
    A = 0
    selected = []
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        ok = True
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))) and (A & (1 << (x + 3 * d))):
                ok = False
                break
        if not ok: continue
        
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        if ok:
            for d in range(1, min(x - 1, N - x) // 2 + 1):
                if (A & (1 << (x - d))) and (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))):
                    ok = False
                    break
        if not ok: continue
        
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        if ok:
            for d in range(1, min(x - 2, N - x) // 1 + 1):
                if (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))) and (A & (1 << (x + d))):
                    ok = False
                    break
        if not ok: continue
        
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        if ok:
            for d in range(1, (x - 1) // 3 + 1):
                if (A & (1 << (x - 3 * d))) and (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))):
                    ok = False
                    break
        if not ok: continue
        
        A |= (1 << x)
        selected.append(x)
    
    # Count 3-APs
    def count_3aps(A):
        count = 0
        for x in range(1, N + 1):
            if not (A & (1 << x)): continue
            # x is middle
            for d in range(1, min(x - 1, N - x) + 1):
                if (A & (1 << (x - d))) and (A & (1 << (x + d))):
                    count += 1
            # x is low end
            for d in range(1, (N - x) // 2 + 1):
                if (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))):
                    count += 1
            # x is high end
            for d in range(1, (x - 1) // 2 + 1):
                if (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))):
                    count += 1
        return count

    best_A = A
    best_score = count_3aps(A)
    
    # Simulated Annealing
    # We need fast incremental gain/loss
    def gain_3ap(x, A):
        g = 0
        # x is middle
        for d in range(1, min(x - 1, N - x) + 1):
            if (A & (1 << (x - d))) and (A & (1 << (x + d))):
                g += 1
        # x is low end
        for d in range(1, (N - x) // 2 + 1):
            if (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))):
                g += 1
        # x is high end
        for d in range(1, (x - 1) // 2 + 1):
            if (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))):
                g += 1
        return g

    def loss_3ap(x, A):
        # Same as gain, because removing x breaks the same progressions
        return gain_3ap(x, A)

    def can_add(x, A):
        # 1. x is 1st term
        for d in range(1, (N - x) // 3 + 1):
            if (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))) and (A & (1 << (x + 3 * d))):
                return False
        # 2. x is 2nd term
        for d in range(1, min(x - 1, N - x) // 2 + 1):
            if (A & (1 << (x - d))) and (A & (1 << (x + d))) and (A & (1 << (x + 2 * d))):
                return False
        # 3. x is 3rd term
        for d in range(1, min(x - 2, N - x) // 1 + 1):
            if (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))) and (A & (1 << (x + d))):
                return False
        # 4. x is 4th term
        for d in range(1, (x - 1) // 3 + 1):
            if (A & (1 << (x - 3 * d))) and (A & (1 << (x - 2 * d))) and (A & (1 << (x - d))):
                return False
        return True

    start_time = time.time()
    timeout = 50.0 # Leave 10s for safety
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    current_A = A
    current_score = best_score
    
    while time.time() - start_time < timeout:
        # Pick a random move: Add, Remove, Swap
        move_type = random.randint(0, 2)
        
        if move_type == 0: # Add
            candidates = [x for x in range(1, N + 1) if not (current_A & (1 << x)) and can_add(x, current_A)]
            if not candidates: continue
            x = random.choice(candidates)
            new_A = current_A | (1 << x)
            delta = gain_3ap(x, current_A)
            new_score = current_score + delta
        elif move_type == 1: # Remove
            in_set = [x for x in range(1, N + 1) if current_A & (1 << x)]
            if not in_set: continue
            x = random.choice(in_set)
            new_A = current_A & ~(1 << x)
            delta = -loss_3ap(x, current_A)
            new_score = current_score + delta
        else: # Swap
            in_set = [x for x in range(1, N + 1) if current_A & (1 << x)]
            out_set = [x for x in range(1, N + 1) if not (current_A & (1 << x))]
            if not in_set or not out_set: continue
            x_remove = random.choice(in_set)
            x_add = random.choice(out_set)
            # Check if x_add can be added to A \ {x_remove}
            temp_A = current_A & ~(1 << x_remove)
            if not can_add(x_add, temp_A):
                continue
            new_A = temp_A | (1 << x_add)
            # Score change: lose progressions with x_remove, gain progressions with x_add in new context
            # Note: gain/loss functions depend on the current set.
            # Loss is calculated on current_A.
            # Gain is calculated on temp_A (set without x_remove).
            # This is an approximation if the two interact, but usually they don't share 3-APs unless close.
            # Actually, a 3-AP cannot contain both x_remove and x_add if they are far apart?
            # If they are in the same 3-AP, say x_remove, m, x_add.
            # Removing x_remove breaks it. Adding x_add to temp_A (which doesn't have x_remove) doesn't form it.
            # So the calculation `delta = -loss(x_remove, current_A) + gain(x_add, temp_A)` is exact?
            # Yes, because the set of 3-APs in new_A is (3APs in current_A) \ (3APs containing x_remove) \ (3APs containing both x_remove and x_add) + (3APs containing x_add in new_A).
            # Since x_add is not in current_A, no 3-AP in current_A contains x_add.
            # So 3APs(new_A) = 3APs(current_A) - 3APs_containing_x_remove + 3APs_containing_x_add_in_new_A.
            # Wait, 3APs_containing_x_remove might include ones with x_add? No, x_add not in current_A.
            # So yes, it is exact.
            delta = -loss_3ap(x_remove, current_A) + gain_3ap(x_add, temp_A)
            new_score = current_score + delta

        if new_score > current_score:
            current_A = new_A
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_A = current_A
        else:
            # Accept with probability exp(delta / T)
            if random.random() < pow(2.71828, delta / T):
                current_A = new_A
                current_score = new_score
        
        T *= cooling

    # Final Polish: Hill Climbing
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Try removing elements that reduce score? No, we want to maximize.
        # Try adding elements that increase score.
        for x in range(1, N + 1):
            if not (best_A & (1 << x)) and can_add(x, best_A):
                delta = gain_3ap(x, best_A)
                if delta > 0:
                    best_A |= (1 << x)
                    best_score += delta
                    improved = True
                    # Restart loop as set changed
                    break
        # Try swapping
        in_set = [x for x in range(1, N + 1) if best_A & (1 << x)]
        out_set = [x for x in range(1, N + 1) if not (best_A & (1 << x))]
        for x_remove in in_set:
            if time.time() - start_time > timeout: break
            temp_A = best_A & ~(1 << x_remove)
            base_loss = loss_3ap(x_remove, best_A)
            for x_add in out_set:
                if not can_add(x_add, temp_A): continue
                gain = gain_3ap(x_add, temp_A)
                if gain > base_loss:
                    best_A = temp_A | (1 << x_add)
                    best_score += gain - base_loss
                    improved = True
                    break
            if improved: break

    return sorted([x for x in range(1, N + 1) if best_A & (1 << x)])

