import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a large set of distinct integers in 1..128 such that there are no 
    three distinct members a, b, c for which a+b, a+c, and b+c are also in the set.
    
    Strategy: Greedy search with randomization and local improvement.
    1. Start with a known good base set (e.g., upper half).
    2. Try to add elements from the lower half if they don't violate constraints.
    3. Try to swap elements to improve density.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 90s
    
    N = 128
    best_set = set(range(N//2 + 1, N + 1)) # Initial 64 elements
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S creates a forbidden triple.
        Forbidden: distinct a,b,c in S ∪ {x} such that a+b, a+c, b+c in S ∪ {x}.
        
        Since we are adding x to a valid set S, the new forbidden triple must involve x.
        Let the triple be {x, a, b} with x, a, b distinct.
        The pair sums are: x+a, x+b, a+b.
        All three must be in S ∪ {x}.
        
        Cases:
        1. a+b == x: Then we need x+a and x+b in S.
           If x+a in S and x+b in S, then {x, a, b} is a forbidden triple.
        2. x+a == x: Impossible (a=0 not in 1..128).
        3. x+b == x: Impossible.
        4. a+b == x+a: implies b=x, not distinct.
        5. a+b == x+b: implies a=x, not distinct.
        6. x+a == x+b: implies a=b, not distinct.
        
        Wait, the sums don't have to be distinct from the members or each other?
        "The six displayed values need not be mutually distinct."
        But a, b, c are distinct members.
        
        So if we add x, we look for distinct a, b in S such that:
        - x, a, b are distinct.
        - x+a is in S ∪ {x}
        - x+b is in S ∪ {x}
        - a+b is in S ∪ {x}
        
        Since a,b in S and S doesn't contain x, a+b can be x (if x in S∪{x}) or in S.
        x+a can be x (if a=0, no) or in S.
        x+b can be x (if b=0, no) or in S.
        
        So we need:
        1. a+b == x AND (x+a in S) AND (x+b in S)
        2. a+b in S AND (x+a in S) AND (x+b in S)
        
        Note: If a+b == x, then a+b is in S∪{x} (it is x).
        If x+a == x, a=0 (invalid).
        If x+b == x, b=0 (invalid).
        
        So the condition for x to be invalid to add is:
        Exists distinct a, b in S such that:
        ( (a+b == x and (x+a in S) and (x+b in S)) 
          OR 
          (a+b in S and (x+a in S) and (x+b in S)) )
        """
        for a in S:
            if a == x: continue
            # Check all b > a to avoid duplicates, but order doesn't matter for set membership
            for b in S:
                if b == x or b == a: continue
                # We have distinct x, a, b
                sum_ab = a + b
                sum_xa = x + a
                sum_xb = x + b
                
                # All sums must be in S ∪ {x}
                if (sum_ab in S or sum_ab == x) and \
                   (sum_xa in S or sum_xa == x) and \
                   (sum_xb in S or sum_xb == x):
                    return False
        return True

    def get_score(S):
        return len(S)

    # Try to add elements from 1..N//2
    candidates = list(range(1, N//2 + 1))
    random.shuffle(candidates)
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for x in candidates:
            if x in best_set:
                continue
            if is_valid_addition(best_set, x):
                best_set.add(x)
                added_any = True
                # Re-shuffle remaining candidates? Or just continue?
                # Re-shuffling might help find different combinations
                candidates = [c for c in candidates if c not in best_set]
                random.shuffle(candidates)
                break # Restart loop to check new candidates against updated set
    
    # Local Search: Try to swap elements
    # If we have time, try to replace a large element with a smaller one if it maintains validity and size (or increases it?)
    # Actually, swapping doesn't change size. We want to maximize size.
    # Can we remove an element to add two others?
    
    # Simple improvement: Try adding more if possible (already done above)
    # Let's try a few random restarts if time permits
    
    best_size = len(best_set)
    best_result = list(best_set)
    
    # If we have time, do a few random restarts with different initial sets or orders
    restarts = 0
    while time.time() < deadline and restarts < 5:
        restarts += 1
        current_set = set(range(N//2 + 1, N + 1))
        # Randomly remove a few elements from the upper half to see if we can pack more
        # This is a heuristic: sometimes removing a "blocking" element allows adding many smaller ones
        if restarts > 1:
            # Try a different strategy: start with fewer upper elements
            # e.g., keep only top 60
            start_size = 60 + random.randint(0, 4)
            current_set = set(range(N - start_size + 1, N + 1))
        
        candidates = list(range(1, N - len(current_set) + 1))
        random.shuffle(candidates)
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in candidates:
                if x in current_set:
                    continue
                if is_valid_addition(current_set, x):
                    current_set.add(x)
                    added_any = True
                    candidates = [c for c in candidates if c not in current_set]
                    random.shuffle(candidates)
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = list(current_set)
            
    return best_result
# EVOLVE-BLOCK-END

def construct():
    return construct() # Recursive call? No, that's wrong.
    
# Wait, I defined construct inside the block. I need to make sure the entry point is correct.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     return list(range(P['n']//2+1,P['n']+1))

# So I just need to replace the body of construct.

def construct():
    """
    Constructs a large set of distinct integers in 1..128 such that there are no 
    three distinct members a, b, c for which a+b, a+c, and b+c are also in the set.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin out of 90s
    
    N = 128
    best_set = set(range(N//2 + 1, N + 1)) # Initial 64 elements
    
    def is_valid_addition(S, x):
        """
        Check if adding x to S creates a forbidden triple.
        """
        for a in S:
            if a == x: continue
            for b in S:
                if b == x or b == a: continue
                sum_ab = a + b
                sum_xa = x + a
                sum_xb = x + b
                
                if (sum_ab in S or sum_ab == x) and \
                   (sum_xa in S or sum_xa == x) and \
                   (sum_xb in S or sum_xb == x):
                    return False
        return True

    # Try to add elements from 1..N//2
    candidates = list(range(1, N//2 + 1))
    random.shuffle(candidates)
    
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for x in candidates:
            if x in best_set:
                continue
            if is_valid_addition(best_set, x):
                best_set.add(x)
                added_any = True
                candidates = [c for c in candidates if c not in best_set]
                random.shuffle(candidates)
                break
    
    best_size = len(best_set)
    best_result = list(best_set)
    
    # If we have time, do a few random restarts
    restarts = 0
    while time.time() < deadline and restarts < 10:
        restarts += 1
        # Strategy: Start with a random subset of the upper half?
        # Or just randomize the order of adding lower elements more aggressively?
        
        # Let's try a different initial state:
        # Keep upper half, but randomly remove 1-3 elements to break symmetry/blocking
        current_set = set(range(N//2 + 1, N + 1))
        if restarts > 1:
            removals = random.randint(1, 3)
            for _ in range(removals):
                if current_set:
                    current_set.remove(random.choice(list(current_set)))
        
        candidates = list(range(1, N//2 + 1))
        random.shuffle(candidates)
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for x in candidates:
                if x in current_set:
                    continue
                if is_valid_addition(current_set, x):
                    current_set.add(x)
                    added_any = True
                    candidates = [c for c in candidates if c not in current_set]
                    random.shuffle(candidates)
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = list(current_set)
            
    return best_result
