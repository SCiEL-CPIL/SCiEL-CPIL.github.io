"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def is_maximal(selected):
        """Check if selected is a maximal Sidon set in [1, N]."""
        if not is_sidon(selected):
            return False
        sums = get_sums(selected)
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            # New sums: 2x, x+s for s in selected
            # Collision if 2x in sums OR (x+s) in sums for any s in selected
            # Note: x+s is a sum of x and s. If it exists in sums, it's a collision.
            if 2 * x in sums:
                continue
            can_add = True
            for s in selected:
                if (x + s) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True

    def greedy_blocking_seed(seed_val):
        """
        Greedy construction that picks elements blocking the most others.
        Uses a random tie-breaker based on seed_val.
        """
        random.seed(seed_val)
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        
        # Pre-compute potential conflicts? 
        # For each x, we want to know how many y in alive it blocks.
        # y is blocked by x (given current S) if adding x to S makes y invalid.
        # This is dynamic. We will compute it on the fly.
        
        while alive:
            best_x = None
            max_blocked = -1
            best_rand = -1
            
            # To speed up, we can sample if alive is large, but N=256 is small.
            # We iterate through all alive candidates to find the one with max blocking power.
            for x in alive:
                # Check if x is compatible with current S
                # x is in alive, so it should be. But let's verify to be safe.
                # Actually, alive is maintained to contain only compatible elements.
                
                # Calculate blocking power:
                # How many y in alive (y != x) become incompatible with S ∪ {x}?
                # y is incompatible with S ∪ {x} if:
                # 1. 2y is in Sums ∪ {2x, x+s for s in S}
                # 2. y+s is in Sums ∪ {2x, x+s' for s' in S} for some s in S ∪ {x}
                
                # Since y is in alive, it is compatible with S.
                # So 2y not in Sums, and y+s not in Sums for any s in S.
                
                # New sums from x: NewSumsX = {2x} ∪ {x+s for s in S}
                # y is blocked if:
                # A. 2y in NewSumsX
                # B. (y+x) in Sums  (since y+x is a sum involving x, if it was already in Sums, collision)
                #    Wait, if y+x in Sums, then y+x = a+b for some a,b in S.
                #    This means {x, y, a, b} has a collision.
                #    Is this covered by other conditions?
                #    If y+x = a+b, then adding x creates sum x+y which collides with a+b.
                #    So yes, y is blocked if x+y in Sums.
                # C. y+s in NewSumsX for some s in S.
                #    y+s = 2x  => s = 2x - y. If 2x-y in S, blocked.
                #    y+s = x+s' => s = x + s' - y. If x+s'-y in S, blocked.
                
                # Let's count blocked y's.
                blocked_count = 0
                # Optimization: We can break early if we exceed current max?
                # But we need exact count for comparison.
                
                # Pre-calculate NewSumsX set for fast lookup?
                # NewSumsX size is 1 + |S|.
                # We can just check conditions directly.
                
                # Condition B: x+y in Sums
                if (x + y_placeholder) in Sums: pass # Just logic check
                
                # Let's iterate y
                for y in alive:
                    if y == x:
                        continue
                    
                    # Check A: 2y in {2x} ∪ {x+s for s in S}
                    if 2 * y == 2 * x: # y=x, skipped
                        pass
                    # 2y = x+s => s = 2y - x
                    if (2 * y - x) in S:
                        blocked_count += 1
                        continue
                    # 2y = 2x => y=x, skipped
                    
                    # Check B: x+y in Sums
                    if (x + y) in Sums:
                        blocked_count += 1
                        continue
                    
                    # Check C: y+s in {2x} ∪ {x+s' for s' in S}
                    # y+s = 2x => s = 2x - y
                    if (2 * x - y) in S:
                        blocked_count += 1
                        continue
                    # y+s = x+s' => s = x + s' - y
                    # We need to check if there exists s in S such that (x + s' - y) is in S?
                    # No, the condition is: does the set {y+s | s in S} intersect {2x} ∪ {x+s' | s' in S}?
                    # Intersection with {2x} handled above.
                    # Intersection with {x+s' | s' in S}:
                    # y+s = x+s' => s' = y + s - x.
                    # So we check if for any s in S, (y + s - x) is in S.
                    # This is O(|S|).
                    
                    blocked = False
                    for s in S:
                        if (y + s - x) in S:
                            blocked = True
                            break
                    if blocked:
                        blocked_count += 1
                        continue
                        
                    # If we reach here, y is NOT blocked by x.
                
                # Tie-breaking: use random value to explore different structures
                r = random.random()
                if blocked_count > max_blocked or (blocked_count == max_blocked and r > best_rand):
                    max_blocked = blocked_count
                    best_x = x
                    best_rand = r
            
            if best_x is None:
                break
                
            # Add best_x to S
            S.append(best_x)
            # Update Sums
            Sums.add(2 * best_x)
            for s in S[:-1]: # S without best_x
                Sums.add(best_x + s)
                
            # Remove best_x from alive
            alive.remove(best_x)
            
            # Remove all y in alive that are now blocked by S (which includes best_x)
            # This maintains the invariant that alive contains only elements compatible with S.
            to_remove = []
            for y in alive:
                # Check if y is compatible with S
                # 1. 2y in Sums?
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                # 2. y+s in Sums for any s in S?
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
                    continue
                    
            for y in to_remove:
                alive.remove(y)

        return S

    def repair_maximality(S):
        """
        If S is Sidon but not maximal, try to make it maximal without increasing size.
        This is hard. However, the greedy algorithm *should* produce a maximal set
        because it removes all blocked elements at each step.
        If it's not maximal, it means an element x was not blocked by any sequence of additions?
        Actually, if x is not in S and not blocked, it means x was in 'alive' at the end?
        No, the loop terminates when 'alive' is empty.
        So if the loop terminates, 'alive' is empty, meaning all elements are either in S or blocked.
        Thus, S should be maximal by construction.
        
        However, if there is a bug in the blocking logic, it might fail.
        Let's verify maximality. If not maximal, we can try to swap an element in S
        with an extendable element to maintain size but block the extendable one?
        Or simply return S and hope the verifier is lenient? No, verifier is strict.
        
        If S is not maximal, we must fix it.
        The only way to fix non-maximality without increasing size is to replace an element.
        But replacing might break Sidon property.
        
        Given the strictness, if greedy fails maximality, we fall back to a simpler
        deterministic greedy that is known to be maximal (ascending greedy) but might be larger.
        However, the task is to *minimize* cardinality. A larger set is worse.
        
        Let's trust the greedy blocking logic. It removes ALL blocked elements.
        If an element is not blocked, it stays in 'alive'.
        The loop continues until 'alive' is empty.
        So every element not in S is blocked by S.
        Therefore, S is maximal.
        """
        return S

    # Strategy:
    # 1. Run greedy_blocking with several different seeds to find the smallest valid maximal set.
    # 2. Since greedy is deterministic given the seed (due to random tie-breaking),
    #    different seeds explore different "greedy paths".
    # 3. We keep the best (smallest) result.
    
    best_set = None
    best_size = N + 1
    
    # Run a few iterations with different seeds
    # N=256, greedy is O(N^3) approx. 256^3 = 16M ops.
    # In Python, 16M ops might take 1-5 seconds.
    # We have 55 seconds. We can run ~10-20 seeds.
    
    num_seeds = 20
    for i in range(num_seeds):
        if time.time() > deadline - 5.0:
            break
            
        S = greedy_blocking_seed(i * 12345 + 7)
        
        if not S:
            continue
            
        if is_sidon(S):
            if is_maximal(S):
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = S
            else:
                # This shouldn't happen if logic is correct, but if it does,
                # we can try to force maximality by adding the missing extendable elements?
                # That increases size.
                # Or we can discard this result.
                pass

    if best_set is None:
        # Fallback: Simple ascending greedy (always maximal, likely larger)
        S = []
        Sums = set()
        for x in range(1, N + 1):
            new_sums = {2 * x}
            new_sums.update(x + s for s in S)
            if not (new_sums & Sums):
                S.append(x)
                Sums.update(new_sums)
        best_set = S

    return best_set

# EVOLVE-BLOCK-END
