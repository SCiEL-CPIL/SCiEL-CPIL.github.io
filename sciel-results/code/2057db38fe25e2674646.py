import time
import random

# Precompute all 4-APs and 3-APs info
N = 256
all_4aps = []
# For each 4-AP, store the tuple of elements
# We also need for each element, the list of 4-APs it belongs to
element_to_4aps = {i: [] for i in range(1, N+1)}

for d in range(1, (N-1)//3 + 1):
    for a in range(1, N - 3*d + 1):
        b = a + d
        c = a + 2*d
        e = a + 3*d
        ap = (a, b, c, e)
        all_4aps.append(ap)
        element_to_4aps[a].append(ap)
        element_to_4aps[b].append(ap)
        element_to_4aps[c].append(ap)
        element_to_4aps[e].append(ap)

# Precompute 3-APs? 
# Counting 3-APs incrementally is better.
# A 3-AP is (a, b, c) with a+c=2b.
# We can just count them by iterating b and checking pairs.
# But incremental is O(N) per move.

def count_3aps(S):
    """Count number of 3-APs in set S."""
    count = 0
    S_list = sorted(S)
    # For each middle term b
    for b in S_list:
        # Check a = b-d, c = b+d
        # We can just iterate d
        for d in range(1, N):
            if (b-d in S) and (b+d in S):
                count += 1
            if b+d > N:
                break
    return count

def is_safe_add(S, y):
    """Check if adding y to S creates a 4-AP."""
    # y is not in S
    for ap in element_to_4aps[y]:
        # Check if the other 3 elements are in S
        # ap is a tuple of 4
        others = [x for x in ap if x != y]
        if all(x in S for x in others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate initial set using greedy
    S = set()
    for x in range(1, N+1):
        if is_safe_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = count_3aps(best_S)
    
    # 2. Local Search
    # We will try to improve by adding, removing, or swapping elements.
    # To avoid getting stuck, we can use a simple hill climbing with random restarts
    # or simulated annealing. Given the time, simple hill climbing with 
    # "best improvement" or "first improvement" is good.
    # Let's do first improvement hill climbing with random moves.
    
    current_S = set(S)
    current_score = best_score
    
    # To speed up 3-AP count updates, we can just recount from scratch if moves are sparse,
    # but N=256 is small. count_3aps is O(N^2) ~ 65k ops. 
    # If we do 10k iterations, that's 650M ops, which might be too slow in Python.
    # Let's optimize count_3aps or use incremental.
    
    # Incremental 3-AP count:
    # delta_3aps = new_3aps - old_3aps
    # Adding y:
    #   y as first: (y, y+d, y+2d) -> y+d, y+2d in S
    #   y as mid: (y-d, y, y+d) -> y-d, y+d in S
    #   y as last: (y-2d, y-d, y) -> y-2d, y-d in S
    # Removing y:
    #   Reverse of adding
    
    def delta_3aps_add(S, y):
        delta = 0
        # y as first
        for d in range(1, N-y+1):
            if (y+d in S) and (y+2*d in S):
                delta += 1
            if y+2*d > N: break
        # y as middle
        for d in range(1, min(y-1, N-y)+1):
            if (y-d in S) and (y+d in S):
                delta += 1
        # y as last
        for d in range(1, (y+1)//2):
            if (y-2*d in S) and (y-d in S):
                delta += 1
        return delta

    def delta_3aps_remove(S, y):
        # Same as add, but we are removing, so we subtract the 3-APs that y was part of.
        # Note: The 3-APs counted in delta_3aps_add are those *new* 3-APs formed by adding y.
        # These are exactly the 3-APs that will be *lost* when removing y.
        return delta_3aps_add(S, y)

    # Let's verify delta_3aps_add logic.
    # If we add y, we gain 3-APs where y is a member and the other two are in S.
    # If we remove y, we lose 3-APs where y is a member and the other two are in S.
    # So yes, the magnitude is the same.
    
    # We need to update current_score correctly.
    
    while time.time() < deadline:
        improved = False
        
        # Try to add a random element
        # Get candidates that are safe
        # Instead of checking all, pick a random few
        for _ in range(10):
            y = random.randint(1, N)
            if y not in current_S:
                if is_safe_add(current_S, y):
                    new_score = current_score + delta_3aps_add(current_S, y)
                    if new_score > current_score:
                        current_S.add(y)
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_S = set(current_S)
                            best_score = current_score
                        break # Restart search with new state
        
        # Try to remove a random element (only if it doesn't reduce score? 
        # Actually, removing might allow adding more later. 
        # But for pure hill climbing on score, we only accept if score increases.
        # Removing always decreases score (or keeps same if y is in no 3-AP).
        # So simple hill climbing will never remove.
        # We need swaps or simulated annealing to escape local optima.
        
        # Let's try swaps: Remove x, Add y.
        # We need to check if S \ {x} U {y} is 4-AP free.
        # And if score increases.
        
        for _ in range(10):
            x = random.choice(list(current_S))
            y = random.randint(1, N)
            if y == x or y in current_S:
                continue
            
            # Check validity of S' = S - {x} + {y}
            # 1. y must be safe in S - {x}
            #    is_safe_add checks if y creates 4-AP with elements in S.
            #    But we removed x. So we need to check if y creates 4-AP with S \ {x}.
            #    This is equivalent to: 
            #    Does y form a 4-AP with 3 elements from S?
            #    If yes, and one of them is x, it might be ok if x is removed.
            #    So we need to check: Are there 3 elements in S \ {x} that form a 4-AP with y?
            
            safe = True
            for ap in element_to_4aps[y]:
                others = [z for z in ap if z != y]
                # Check if all others are in S \ {x}
                # i.e., all others are in S, and x is not in others (or if x is in others, it's removed so it's fine? No.)
                # If x is in others, then others \ {x} must be in S.
                # Actually, simpler:
                # The 4-AP is formed if {y} U (others \ {x if x in others else set()}) is subset of S'.
                # S' contains y and S \ {x}.
                # So we need (others \ {x}) to be subset of S.
                # And y is in S'.
                # So we just need to check if (others \ {x}) is subset of S.
                
                # Wait, if x is NOT in others, then we need all others in S.
                # If x IS in others, we need (others - {x}) in S.
                
                others_set = set(others)
                if x in others_set:
                    others_set.remove(x)
                
                if others_set.issubset(current_S):
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Calculate score change
            # Remove x: lose delta_3aps_remove(S, x)
            # Add y: gain delta_3aps_add(S \ {x}, y)
            # Note: delta_3aps_add depends on the set.
            # We need delta_3aps_add for S' without y, which is S \ {x}.
            
            # Let's compute delta_remove_x = delta_3aps_add(current_S, x)
            # Let's compute delta_add_y = delta_3aps_add(current_S - {x}, y)
            
            # But current_S - {x} is not a set we have handy for the helper.
            # We can modify delta_3aps_add to take an exclusion? Or just compute it.
            # Since N is small, we can just create a temporary set? 
            # Or pass a "exclude" set.
            
            # Let's just compute the score of S' directly? No, too slow.
            # Let's compute delta_add_y manually with exclusion of x.
            
            d_add = 0
            # y as first
            for d in range(1, N-y+1):
                if (y+d in current_S) and (y+2*d in current_S) and (y+d != x) and (y+2*d != x):
                    d_add += 1
                if y+2*d > N: break
            # y as middle
            for d in range(1, min(y-1, N-y)+1):
                if (y-d in current_S) and (y+d in current_S) and (y-d != x) and (y+d != x):
                    d_add += 1
            # y as last
            for d in range(1, (y+1)//2):
                if (y-2*d in current_S) and (y-d in current_S) and (y-2*d != x) and (y-d != x):
                    d_add += 1
            
            d_rem = delta_3aps_add(current_S, x)
            
            new_score = current_score - d_rem + d_add
            
            if new_score > current_score:
                current_S.remove(x)
                current_S.add(y)
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_S = set(current_S)
                    best_score = current_score
                break # Restart
        
        if not improved:
            # Stuck in local optimum.
            # Random restart? Or perturb?
            # Let's try a random perturbation: remove 2 random elements, add 2 random safe ones?
            # Or just restart from best_S?
            # Let's try to escape by accepting a slightly worse move (Simulated Annealing style)
            # Or just restart from best_S with a random modification.
            
            # Simple strategy: Randomly remove one element from current_S, 
            # then try to add random safe elements to maximize score again.
            if current_S:
                x = random.choice(list(current_S))
                current_S.remove(x)
                current_score -= delta_3aps_add(current_S | {x}, x) # Wait, delta_3aps_add(S, x) assumes x not in S.
                # If we remove x from S, the score drops by the number of 3-APs containing x.
                # That number is delta_3aps_add(S, x) where S is the set WITHOUT x?
                # No, delta_3aps_add(S, x) counts 3-APs formed by adding x to S.
                # So if S contains x, the number of 3-APs lost by removing x is the number of 3-APs containing x and two other elements from S\{x}.
                # This is exactly delta_3aps_add(S\{x}, x).
                
                # So:
                # S_old = S
                # S_new = S \ {x}
                # Score_new = Score_old - delta_3aps_add(S_new, x)
                
                # Let's recalculate properly.
                # We already removed x from current_S.
                # So current_S is S_new.
                # The score drop is delta_3aps_add(current_S, x).
                current_score -= delta_3aps_add(current_S, x)
                
                # Now try to add random safe elements
                for _ in range(20):
                    y = random.randint(1, N)
                    if y not in current_S:
                        if is_safe_add(current_S, y):
                            new_score = current_score + delta_3aps_add(current_S, y)
                            if new_score > current_score:
                                current_S.add(y)
                                current_score = new_score
                                if current_score > best_score:
                                    best_S = set(current_S)
                                    best_score = current_score

    return sorted(best_S)

# Test the construct function briefly
if __name__ == "__main__":
    # Just a sanity check
    res = construct()
    print(len(res))
    # Verify no 4-AP
    S = set(res)
    for ap in all_4aps:
        if all(x in S for x in ap):
            print("Error: 4-AP found", ap)
            break
    print("Done")
