import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    n = 256
    base_order = list(range(1, n + 1))
    
    # State representation for speed
    # We maintain A, B as lists, and sums_A, sums_B, diffs_A, diffs_B as sets
    
    def try_add(x, side, A, B, sums, diffs):
        """
        Try to add x to side (0 for A, 1 for B).
        Returns True if successful, updating the state.
        """
        S = A if side == 0 else B
        other_diffs = diffs[1 - side]
        my_sums = sums[side]
        my_diffs = diffs[side]
        
        # Check Sidon: new pair sums a+x must be unique and not in my_sums
        # Also x+x must not be in my_sums
        for a in S:
            if (a + x) in my_sums:
                return False
        if (x + x) in my_sums:
            return False
            
        # Check Diff Disjointness: new diffs x-a and a-x must not be in other_diffs
        for a in S:
            if a != x:
                if (x - a) in other_diffs or (a - x) in other_diffs:
                    return False
        
        # If all checks pass, add x
        S.append(x)
        for a in S:
            my_sums.add(a + x)
            if a != x:
                my_diffs.add(x - a)
                my_diffs.add(a - x)
        my_sums.add(x + x)
        return True

    def remove(x, side, A, B, sums, diffs):
        """
        Remove x from side (0 for A, 1 for B).
        """
        S = A if side == 0 else B
        my_sums = sums[side]
        my_diffs = diffs[side]
        
        # Remove sums and diffs involving x
        for a in S:
            if a != x:
                my_sums.discard(a + x)
                my_diffs.discard(x - a)
                my_diffs.discard(a - x)
        my_sums.discard(x + x)
        
        S.remove(x)

    def greedy_construct(order, deadline, prefer_balance=True):
        A = []
        B = []
        sums = [set(), set()]
        diffs = [set(), set()]
        
        for x in order:
            if time.time() > deadline:
                break
                
            placed = False
            # Try to place in A or B
            # Heuristic: try the smaller set first to balance sizes, or randomize
            sides = [0, 1]
            if prefer_balance:
                if len(A) > len(B):
                    sides = [1, 0]
            else:
                random.shuffle(sides)
                
            for side in sides:
                if try_add(x, side, A, B, sums, diffs):
                    placed = True
                    break
                    
        return A, B

    def local_search(A, B, deadline):
        A = list(A)
        B = list(B)
        sums = [set(), set()]
        diffs = [set(), set()]
        
        # Rebuild sums and diffs
        for x in A:
            for y in A:
                if x <= y:
                    sums[0].add(x+y)
                if x != y:
                    diffs[0].add(x-y)
        for x in B:
            for y in B:
                if x <= y:
                    sums[1].add(x+y)
                if x != y:
                    diffs[1].add(x-y)
                    
        best_size = len(A) + len(B)
        improved = True
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Identify blocked elements
            blocked = []
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                # Check if x can be added to A or B
                can_add_A = True
                for a in A:
                    if (a + x) in sums[0] or (x - a) in diffs[1] or (a - x) in diffs[1]:
                        can_add_A = False
                        break
                if can_add_A and (x+x) in sums[0]:
                    can_add_A = False
                    
                can_add_B = True
                for b in B:
                    if (b + x) in sums[1] or (x - b) in diffs[0] or (b - x) in diffs[0]:
                        can_add_B = False
                        break
                if can_add_B and (x+x) in sums[1]:
                    can_add_B = False
                    
                if not can_add_A and not can_add_B:
                    blocked.append(x)
                    
            random.shuffle(blocked)
            
            for x in blocked:
                if time.time() > deadline - 2.0:
                    break
                    
                # Try to remove one element from A or B to allow x
                # We look for an element y in A that conflicts with x
                # If we remove y, can we add x?
                # This is a 1-for-1 swap. Size stays same.
                # To increase size, we need to remove 0 and add 1 (impossible if blocked)
                # Or remove 1 and add 2? Too complex for now.
                # Let's try 1-for-1 swaps to potentially unblock other elements later.
                # Actually, a better heuristic is to try to add x by removing a "less useful" element.
                
                # Let's try a simple hill climbing: try adding x by removing a random conflicting element.
                # If size doesn't increase, we might still accept it if it leads to a better state?
                # No, we want to maximize size.
                
                # Alternative: Try to add x to A by removing an element from A.
                # Then try to add another element y that was blocked by the removed element.
                
                # For now, let's stick to a simpler local search:
                # Try to add x to A. If it fails, find the element a in A causing the conflict.
                # Remove a. Add x.
                # Then try to add a back? No.
                # Try to add some other element z that is now free?
                
                # This is getting complex. Let's rely on multiple random greedy restarts.
                # The local search in the original code was weak.
                pass
        
        return A, B

    best_A = []
    best_B = []
    best_size = 0
    
    # Phase 1: Many random greedy restarts
    random.seed(42)
    num_restarts = 5000
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        order = base_order[:]
        random.shuffle(order)
        
        # Vary the balance preference
        prefer_balance = (i % 2 == 0)
        
        A, B = greedy_construct(order, deadline, prefer_balance)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            
    # Phase 2: Local Search on the best found
    # A more effective local search:
    # Try to add any element. If it fails, try to swap it with an existing element.
    
    def enhanced_local_search(A, B, deadline):
        A = list(A)
        B = list(B)
        sums = [set(), set()]
        diffs = [set(), set()]
        
        # Rebuild
        for x in A:
            for y in A:
                if x <= y:
                    sums[0].add(x+y)
                if x != y:
                    diffs[0].add(x-y)
        for x in B:
            for y in B:
                if x <= y:
                    sums[1].add(x+y)
                if x != y:
                    diffs[1].add(x-y)
                    
        best_size = len(A) + len(B)
        
        # Try to add elements
        while time.time() < deadline - 1.0:
            improved = False
            candidates = [x for x in range(1, n+1) if x not in A and x not in B]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 1.0:
                    break
                    
                # Try adding x to A
                if try_add(x, 0, A, B, sums, diffs):
                    best_size += 1
                    improved = True
                    break # Restart loop with new state
                    
                # Try adding x to B
                if try_add(x, 1, A, B, sums, diffs):
                    best_size += 1
                    improved = True
                    break # Restart loop with new state
                    
            if not improved:
                # Try swaps: remove y from A, add x to A (size same)
                # This might unblock other additions.
                # Let's try a random swap
                if A:
                    y = random.choice(A)
                    remove(y, 0, A, B, sums, diffs)
                    if try_add(x, 0, A, B, sums, diffs):
                        # Size is same, but state changed
                        # We don't accept this as an "improvement" in size,
                        # but it might help later.
                        # To keep it simple and safe, we only accept size improvements.
                        # So we revert.
                        remove(x, 0, A, B, sums, diffs)
                        A.append(y)
                        # Re-add y's sums/diffs
                        for a in A:
                            if a != y:
                                sums[0].add(a+y)
                                diffs[0].add(y-a)
                                diffs[0].add(a-y)
                        sums[0].add(y+y)
                        # We didn't improve, so break out to try another x?
                        # This is inefficient.
                        pass
                break # If no simple addition worked, stop this pass
        
        return A, B

    A, B = enhanced_local_search(best_A, best_B, deadline)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
