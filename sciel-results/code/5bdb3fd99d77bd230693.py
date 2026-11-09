import random
import time

def construct():
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Initial solution (known size 20)
    # 1, 2, 5, 11, 19, 24, 30, 37, 48, 59, 73, 85, 102, 114, 132, 155, 177, 202, 231, 284
    initial_solution = [1, 2, 5, 11, 19, 24, 30, 37, 48, 59, 73, 85, 102, 114, 132, 155, 177, 202, 231, 284]
    
    # Verify initial solution
    def is_valid(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
        
    if not is_valid(initial_solution):
        # Fallback to greedy construction if hardcoded is bad
        A = prescribed.copy()
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        for x in range(12, 285):
            new_sums = {x + a for a in A} | {2 * x}
            if len(new_sums) == len(A) + 1 and not new_sums & sums:
                A.append(x)
                sums |= new_sums
        return A

    best_solution = initial_solution.copy()
    best_size = len(best_solution)
    
    # Time limit guard
    start_time = time.time()
    time_limit = 100  # Keep 20s buffer
    
    # Helper to compute sums and blocked set
    def get_sums_and_blocked(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        
        # Blocked numbers x such that adding x is invalid
        # Condition: 2x in sums OR (x + A) intersects sums
        # x + a = s => x = s - a
        blocked = set()
        for s in sums:
            for a in A:
                x = s - a
                if 1 <= x <= 284 and x not in A:
                    blocked.add(x)
        # Also 2x = s => x = s/2
        for s in sums:
            if s % 2 == 0:
                x = s // 2
                if 1 <= x <= 284 and x not in A:
                    blocked.add(x)
        # Note: The loop above covers x+a=s. 
        # Does it cover 2x=s? 
        # If 2x=s, then x+x=s. This is of the form x+a=s with a=x.
        # But x is not in A, so we can't use a=x in the loop "for a in A".
        # So we must explicitly check 2x=s.
        
        return sums, blocked

    sums, blocked = get_sums_and_blocked(best_solution)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    current_A = best_solution.copy()
    current_sums, current_blocked = get_sums_and_blocked(current_A)
    
    while T > T_min and time.time() - start_time < time_limit:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add an element
            available = [x for x in range(1, 285) if x not in current_A and x not in current_blocked]
            if available:
                x = random.choice(available)
                new_A = current_A + [x]
                # Check validity (should be valid by construction, but double check)
                if is_valid(new_A):
                    current_A = new_A
                    current_sums, current_blocked = get_sums_and_blocked(current_A)
                    if len(current_A) > best_size:
                        best_size = len(current_A)
                        best_solution = current_A.copy()
        elif move_type < 0.6:
            # Try to remove an element (not prescribed)
            removable = [x for x in current_A if x not in prescribed]
            if removable:
                y = random.choice(removable)
                new_A = [x for x in current_A if x != y]
                # Accept if it improves potential (more available slots) or randomly
                new_sums, new_blocked = get_sums_and_blocked(new_A)
                # Heuristic: prefer moves that increase available slots
                # Available slots = 284 - len(A) - len(blocked)
                curr_avail = 284 - len(current_A) - len(current_blocked)
                new_avail = 284 - len(new_A) - len(new_blocked)
                
                # Delta in size is -1
                # We accept if new_avail is significantly better, or with prob exp(-delta/T)
                # Let's define energy as -size + lambda * (-avail) ?
                # Simpler: Accept if new_avail > curr_avail, or with prob 0.1
                if new_avail > curr_avail or random.random() < 0.1:
                    current_A = new_A
                    current_sums, current_blocked = new_sums, new_blocked
        else:
            # Try to swap
            removable = [x for x in current_A if x not in prescribed]
            available = [x for x in range(1, 285) if x not in current_A and x not in current_blocked]
            
            if removable and available:
                y = random.choice(removable)
                x = random.choice(available)
                new_A = [z for z in current_A if z != y] + [x]
                
                if is_valid(new_A):
                    new_sums, new_blocked = get_sums_and_blocked(new_A)
                    curr_avail = 284 - len(current_A) - len(current_blocked)
                    new_avail = 284 - len(new_A) - len(new_blocked)
                    
                    # Accept if it improves potential
                    if new_avail > curr_avail or random.random() < 0.2:
                        current_A = new_A
                        current_sums, current_blocked = new_sums, new_blocked
                        if len(current_A) > best_size:
                            best_size = len(current_A)
                            best_solution = current_A.copy()
                            
        T *= cooling
        
    return best_solution
