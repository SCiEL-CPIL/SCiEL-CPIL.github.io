import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 256
    
    def is_valid_addition(A, x):
        # Check if adding x to A creates a 4-AP
        # 4-AP: a, a+d, a+2d, a+3d
        # Max d: (N-1)//3
        for d in range(1, (N - 1) // 3 + 1):
            # x is a+3d
            if x - 3*d >= 1:
                if (x - 3*d in A and x - 2*d in A and x - d in A):
                    return False
            # x is a+2d
            if x - 2*d >= 1 and x + d <= N:
                if (x - 2*d in A and x - d in A and x + d in A):
                    return False
            # x is a+d
            if x - d >= 1 and x + 2*d <= N:
                if (x - d in A and x + d in A and x + 2*d in A):
                    return False
            # x is a
            if x + 3*d <= N:
                if (x + d in A and x + 2*d in A and x + 3*d in A):
                    return False
        return True

    def count_3aps(A):
        count = 0
        # Max d: (N-1)//2
        for d in range(1, (N - 1) // 2 + 1):
            for a in range(1, N - 2*d + 1):
                if a in A and (a + d) in A and (a + 2*d) in A:
                    count += 1
        return count

    def delta_3ap_add(A, x):
        # Calculate increase in 3-APs if x is added to A
        delta = 0
        for d in range(1, (N - 1) // 2 + 1):
            # x is a+2d (end)
            if x - 2*d >= 1 and x - d in A and x - 2*d in A:
                delta += 1
            # x is a+d (middle)
            if x - d >= 1 and x + d <= N and x - d in A and x + d in A:
                delta += 1
            # x is a (start)
            if x + 2*d <= N and x + d in A and x + 2*d in A:
                delta += 1
        return delta

    # GRASP Initialization
    best_initial_set = set()
    best_initial_score = -1
    
    for _ in range(5): # 5 random greedy inits
        A = set()
        # Randomize order? Or randomized choice from candidates
        # Simple: iterate 1..N, but with some randomness in acceptance?
        # Or just standard greedy is strong.
        # Let's try standard greedy first.
        for x in range(1, N + 1):
            if is_valid_addition(A, x):
                A.add(x)
        
        score = count_3aps(A)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_set = set(A)
            
    # If time allows, try randomized greedy
    if time.time() < start_time + 10:
        for _ in range(10):
            A = set()
            # Randomized Greedy: pick from top candidates?
            # For simplicity, just shuffle and pick first valid? No, that's weak.
            # Let's stick to standard greedy for init, it's fast and good.
            pass

    current_set = best_initial_set
    current_score = best_initial_score
    best_set = set(current_set)
    best_score = current_score
    
    # Tabu Search
    tabu_len = 30
    tabu_dict = {}
    iter_count = 0
    no_improve = 0
    
    while time.time() < deadline:
        iter_count += 1
        best_move_score = current_score
        best_move_type = None
        best_move_elem = None
        
        # Evaluate Moves
        # To speed up, we can skip elements that are clearly bad?
        # No, just evaluate all.
        
        # Add moves
        for x in range(1, N + 1):
            if x in current_set: continue
            if not is_valid_addition(current_set, x): continue
            
            delta = delta_3ap_add(current_set, x)
            new_score = current_score + delta
            is_tabu = tabu_dict.get(x, 0) > iter_count
            
            if (not is_tabu or new_score > best_score) and new_score > best_move_score:
                best_move_score = new_score
                best_move_type = 'add'
                best_move_elem = x
        
        # Remove moves
        for x in list(current_set):
            # Delta for removal is -delta_3ap_add(current_set - {x}, x)
            # But we can compute delta on current_set and subtract 1? No.
            # If x is in A, the APs containing x are those where x is part of the triple.
            # Counting them is same as delta_3ap_add(A\{x}, x).
            # We can approximate or just compute.
            # Since N is small, compute.
            # Temporarily remove x for calculation? No, just pass A\{x}.
            # Creating a new set is O(N).
            # Better: modify delta_3ap_add to take a set and ignore x?
            # No, just use a helper that checks membership in A but assumes x is not there for the "other two" elements.
            # Actually, delta_3ap_add(A, x) checks if neighbors are in A.
            # If x is in A, and we want to know how many APs x forms with A\{x},
            # we just check if the other two are in A. Since x is in A, it doesn't affect the check for the other two.
            # So delta_3ap_add(current_set, x) is exactly the number of APs x forms with other elements in current_set.
            # So removal delta is -delta_3ap_add(current_set, x).
            
            delta = delta_3ap_add(current_set, x)
            new_score = current_score - delta
            is_tabu = tabu_dict.get(x, 0) > iter_count
            
            if (not is_tabu or new_score > best_score) and new_score > best_move_score:
                best_move_score = new_score
                best_move_type = 'remove'
                best_move_elem = x

        if best_move_elem is None:
            # Stuck
            if no_improve > 50:
                break
            # Random kick
            if current_set:
                random_elem = random.choice(list(current_set))
                current_set.remove(random_elem)
                current_score = count_3aps(current_set)
                no_improve = 0
                tabu_dict = {}
            continue

        # Execute
        if best_move_type == 'add':
            current_set.add(best_move_elem)
        else:
            current_set.remove(best_move_elem)
            
        current_score = best_move_score
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            no_improve = 0
        else:
            no_improve += 1
            
        tabu_dict[best_move_elem] = iter_count + tabu_len

    return sorted(best_set)
