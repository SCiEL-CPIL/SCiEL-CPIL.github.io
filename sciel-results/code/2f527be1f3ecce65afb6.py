import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Greedy initialization
    def greedy_construct():
        selected = set()
        for x in range(1, 257):
            # Check if x can be added
            valid = True
            # x as middle: a + b = 2x
            # x as end: 2y - x in selected (y in selected)
            # We just need to check against existing selected
            for y in selected:
                # Case 1: x is middle term. a, x, b. a+b=2x.
                # Since a,b in selected, we check if any pair sums to 2x.
                # This is O(|S|^2) if done naively. 
                # Faster: precompute sums? Or just iterate.
                # For x as middle: we need a,b in S such that a+b=2x.
                # For x as end: we need y in S such that 2y-x in S.
                
                # Check x as end:
                z = 2*y - x
                if z in selected and z != x:
                    valid = False
                    break
                # Check x as middle: 
                # We need to check if there is a,b in S with a+b=2x.
                # This is expensive to do for every y. 
                # Let's restructure the check.
            # Let's do a clean check function.
            if valid:
                # Now check middle term condition separately for efficiency?
                # Actually, checking x as middle requires checking pairs in S.
                # Let's just do a full check in a helper.
                if can_add(x, selected):
                    selected.add(x)
        return selected

    def can_add(x, S):
        # x is not in S
        # 1. x is middle: exists a, b in S, a != b, a+b = 2x
        #    We can check this by iterating a in S, seeing if 2x-a in S.
        for a in S:
            b = 2*x - a
            if b in S and b != a:
                return False
        # 2. x is end: exists y in S, 2y - x in S (and 2y-x != x)
        #    If 2y-x = x, then y=x, but x not in S. So just check if 2y-x in S.
        for y in S:
            z = 2*y - x
            if z in S:
                return False
        return True

    def get_sum(S):
        return sum(1.0/x for x in S)

    best_set = greedy_construct()
    best_score = get_sum(best_set)
    
    # 2. Local Search (Hill Climbing)
    # We try to improve by swapping elements.
    # Since N=256, S size is roughly ~100-150?
    # Max size of 3-AP free set in [1, N] is ~ N/log N? No, it's related to Roth's theorem, but for small N it's denser.
    # Actually, greedy on [1, 256] picks many small numbers.
    
    while time.time() < deadline:
        improved = False
        current_set = set(best_set)
        current_score = best_score
        
        # Try all possible swaps? 
        # |S| is maybe 100. |Not S| is 150. 15000 swaps.
        # Each swap check is O(|S|) or O(|S|^2)?
        # can_add is O(|S|) for end check, O(|S|) for middle check (if we just iterate).
        # Wait, can_add(x, S) iterates S for middle and S for end. So O(|S|).
        # So one swap check is O(|S|).
        # 15000 * 100 = 1.5M ops per iteration. Very fast.
        
        # Find best swap
        best_swap_score = current_score
        best_swap_u = None
        best_swap_v = None
        
        # To speed up, we can just iterate through candidates.
        # We want to replace u in S with v not in S.
        # Gain = 1/v - 1/u. We want high 1/v and low 1/u.
        # So sort S by 1/x ascending (low value first) and NotS by 1/x descending (high value first).
        
        S_list = sorted(current_set, key=lambda x: 1.0/x) # low value first
        NotS_list = sorted([x for x in range(1, 257) if x not in current_set], key=lambda x: -1.0/x) # high value first
        
        # Try swaps
        found_improvement = False
        for u in S_list:
            if found_improvement:
                break
            for v in NotS_list:
                # Potential gain
                if 1.0/v - 1.0/u <= 0:
                    break # Since NotS is sorted desc, further v will have lower 1/v, so gain even worse
                # Check if (S \ {u}) U {v} is valid
                new_set = set(current_set)
                new_set.remove(u)
                if can_add(v, new_set):
                    new_score = current_score - 1.0/u + 1.0/v
                    if new_score > best_swap_score + 1e-12:
                        best_swap_score = new_score
                        best_swap_u = u
                        best_swap_v = v
                        found_improvement = True
                        break # Move to next u? Or accept this one immediately?
                        # Accepting immediately (First Improvement) is faster.
                        # Let's do First Improvement for speed.
        
        if found_improvement:
            best_set.remove(best_swap_u)
            best_set.add(best_swap_v)
            best_score = best_swap_score
            improved = True
        else:
            # No improvement found via single swaps.
            # Try adding? (Shouldn't be possible if greedy was maximal, but swaps might have opened space)
            # Actually, if we can add, we should have added it in the greedy phase or found it in swaps?
            # Swaps maintain size. Additions increase size.
            # Let's check if we can add any element.
            added = False
            for v in NotS_list: # High value first
                if can_add(v, best_set):
                    best_set.add(v)
                    best_score += 1.0/v
                    added = True
                    break
            if not added:
                # Stuck. Break or restart?
                # With First Improvement, we might get stuck in local minima.
                # Let's try a few random restarts if stuck?
                # Or just break, as greedy+hill climbing is usually good.
                # To be safe, let's do a random restart with a perturbed greedy.
                # But time is limited.
                break

    return sorted(best_set)
