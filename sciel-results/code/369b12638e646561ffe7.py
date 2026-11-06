import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    # Initial greedy construction
    S = set()
    counts = {}
    
    def get_new_sums(x, S, y=None):
        """Get sums formed by x with S, excluding pairs involving y if y is being removed."""
        sums = []
        for u in S:
            if u == y:
                continue
            sums.append(x + u)
        sums.append(2 * x)
        return sums

    def is_valid_addition(x, S, counts, y=None):
        """
        Check if adding x and removing y (if y is not None) results in a valid set.
        y is in S. x is not in S.
        """
        # Sums that will be removed
        removed_sums = set()
        if y is not None:
            for v in S:
                if v == y:
                    continue
                removed_sums.add(y + v)
            removed_sums.add(2 * y)
            
        # Sums that will be added
        added_sums = get_new_sums(x, S, y)
        
        # Check constraints
        # We need to check sums that are in added_sums
        # And sums that are in removed_sums (to ensure we don't go negative? No, counts are non-negative)
        # Actually, we just need to ensure max count <= 2.
        # Sums not in added_sums and not in removed_sums are unchanged.
        # Sums in added_sums: new_count = old_count - (1 if in removed_sums else 0) + 1
        # Sums in removed_sums but not in added_sums: new_count = old_count - 1
        
        # Optimization: Only check sums in added_sums and removed_sums.
        # But we must be careful: if a sum is in added_sums, its count increases.
        # If a sum is in removed_sums, its count decreases.
        
        # Let's simulate the change for the relevant sums.
        # We can just iterate through added_sums.
        for s in added_sums:
            old_c = counts.get(s, 0)
            rem_c = 1 if s in removed_sums else 0
            new_c = old_c - rem_c + 1
            if new_c > 2:
                return False
                
        # We also need to ensure that removing y doesn't cause any issues?
        # No, removing elements only decreases counts, so it never violates the <= 2 constraint.
        # The only risk is adding x.
        
        return True

    def apply_change(S, counts, x, y):
        """Apply addition of x and removal of y to S and counts."""
        if y is not None:
            # Remove y
            for v in S:
                if v == y:
                    continue
                s = y + v
                counts[s] -= 1
            s = 2 * y
            counts[s] -= 1
            S.remove(y)
            
        # Add x
        for v in S:
            s = x + v
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1
        S.add(x)

    # 1. Initial Greedy
    for x in range(1, N + 1):
        if is_valid_addition(x, S, counts, None):
            apply_change(S, counts, x, None)
            
    current_size = len(S)
    
    # 2. Local Search
    # We will try to improve the set by adding elements or swapping.
    # To make it faster, we can pre-calculate some things or just rely on the speed of the check.
    
    # Helper to check if we can add x directly
    def try_add(x):
        if x in S:
            return False
        if is_valid_addition(x, S, counts, None):
            apply_change(S, counts, x, None)
            return True
        return False

    # Helper to check swap
    def try_swap(x, y):
        if x in S or y not in S:
            return False
        if is_valid_addition(x, S, counts, y):
            apply_change(S, counts, x, y)
            return True
        return False

    # Main loop
    last_improvement = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Try adding any element
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True
                current_size += 1
                break # Restart search after improvement
        
        if improved:
            continue
            
        # Try swapping
        # We need to be careful with the order.
        # Try a few random swaps
        # If we find a swap, it doesn't increase size, but might allow future additions.
        # However, the goal is to maximize size. Swaps are only useful if they lead to larger sets.
        # So we should prioritize additions.
        # If no additions are possible, we are in a local maximum.
        # To escape, we can do a "kick": remove k elements and try to add different ones.
        
        # Let's try a kick: remove 2 random elements, then try to add any element.
        # This is a simple simulated annealing step with T=0 (greedy kick).
        
        if len(S) > 2:
            # Pick 2 elements to remove
            elems = list(S)
            random.shuffle(elems)
            y1, y2 = elems[0], elems[1]
            
            # Save state
            S_backup = set(S)
            counts_backup = dict(counts)
            
            # Remove y1, y2
            # We can just call apply_change with x=None? No, apply_change adds x.
            # Let's write a remove function or just rollback.
            # Easier: just copy S and counts, remove, and try to add.
            
            # Actually, let's just try to find ANY addition.
            # If none, we are stuck.
            # Let's try a more systematic local search:
            # Iterate through all x not in S, and all y in S.
            # If swap is valid, do it?
            # No, swap doesn't increase size.
            # We only care about size.
            # So if we can't add, we should try to change the set composition to allow additions.
            
            # Let's try: for each y in S, remove it, and see if we can add any x.
            # This is O(|S| * |Candidates| * Cost).
            # 50 * 950 * 50 = 2.3M ops. Fast enough.
            
            found_kick = False
            elems = list(S)
            random.shuffle(elems)
            
            for y in elems:
                if time.time() > deadline:
                    break
                    
                # Save state
                S_orig = set(S)
                counts_orig = dict(counts)
                
                # Remove y
                # Manual removal for speed
                for v in S:
                    if v == y: continue
                    s = y + v
                    counts[s] -= 1
                s = 2 * y
                counts[s] -= 1
                S.remove(y)
                
                # Try to add any x
                # We can try the candidates that were previously blocked by y?
                # Or just random shuffle.
                candidates = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(candidates)
                
                added_any = False
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if is_valid_addition(x, S, counts, None):
                        apply_change(S, counts, x, None)
                        added_any = True
                        # Size is same as before kick (removed 1, added 1)
                        # But we might be able to add more later.
                        # Let's accept this state and restart the main loop.
                        break
                        
                if added_any:
                    found_kick = True
                    break
                else:
                    # Rollback
                    S = S_orig
                    counts = counts_orig
            
            if found_kick:
                continue # Restart main loop to try adding more
        
        # If we looped through everything and didn't improve, we might be in a global local max.
        # Break to avoid infinite loop if no progress?
        # Or keep trying random kicks?
        # Let's add a counter for no-improvement steps and break if too many.
        # But we have time. Let's just keep trying kicks.
        
    return list(S)

# EVOLVE-BLOCK-END
