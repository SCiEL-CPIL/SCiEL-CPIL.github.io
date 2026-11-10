import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial valid set: 63..128
    base_set = set(range(63, 129))
    best_set = base_set.copy()
    best_size = len(best_set)
    
    # Candidates to add: 1..62
    candidates = list(range(1, 63))
    
    # Try many random orders
    # We can also try to add in increasing order, decreasing order, etc.
    # But random is good for escaping local minima (though greedy add doesn't have backtracking, 
    # the order determines which elements get blocked).
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        current_set = base_set.copy()
        cands = candidates.copy()
        random.shuffle(cands)
        
        for x in cands:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Check validity of adding x
            # Violation if exists a, b in current_set such that:
            # 1. a < b
            # 2. a + b in current_set
            # 3. a + x in current_set
            # 4. b + x in current_set
            
            # Optimization:
            # Let A = { a in current_set | a + x in current_set }
            # Let B = { b in current_set | b + x in current_set and b > a }
            # We need a in A, b in B, and a+b in current_set.
            
            # To speed up, we can precompute the set of elements that "conflict" with x?
            # Conflict pair (a,b) exists if a+x, b+x, a+b in S.
            
            valid = True
            # Iterate a
            # We can restrict a to be < x? No.
            # a + x > a.
            
            # Simple double loop
            # To make it faster, we can break early.
            # Also, we can sort current_set? No, set is unordered.
            # Convert to list? No, set iteration is fine.
            
            # Heuristic: Check smaller a first?
            # Let's just iterate.
            
            # Optimization: 
            # If a+x not in current_set, skip a.
            # So we only iterate a where a+x in current_set.
            # Let's build list of such a's.
            a_candidates = [a for a in current_set if a + x in current_set]
            
            for a in a_candidates:
                # We need b > a
                # b + x in current_set
                # a + b in current_set
                
                # Iterate b
                # We can filter b candidates similarly?
                # b_candidates = [b for b in current_set if b > a and b + x in current_set]
                # Then check if a+b in current_set.
                
                # To avoid rebuilding list for each a, we can just loop.
                for b in current_set:
                    if b <= a:
                        continue
                    if b + x in current_set and a + b in current_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            if best_size == 128:
                break

    return list(best_set)
