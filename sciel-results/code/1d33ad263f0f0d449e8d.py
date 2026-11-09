import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 256
    
    A = set()
    remaining = set(range(1, N + 1))
    
    def is_valid_add(x, A):
        # Check 4-AP
        # x is 1st
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    def count_total_3ap(A):
        count = 0
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    def compute_loss(x, A):
        # Loss is the number of 3-APs involving x that would be destroyed if x is removed.
        # This is equivalent to count_3ap_gain(x, A) but strictly counting pairs in A\{x}.
        # Since x is in A, count_3ap_gain(x, A) counts pairs (u,v) in A such that {x,u,v} is 3AP.
        # If we remove x, these APs are gone.
        return count_3ap_gain(x, A)

    # Greedy Phase: pick element with max 3-AP gain that doesn't create 4-AP
    while remaining and time.time() < deadline - 10:
        best_x = -1
        best_score = -1
        for x in remaining:
            if is_valid_add(x, A):
                score = count_3ap_gain(x, A)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x == -1:
            break
        A.add(best_x)
        remaining.remove(best_x)
    
    current_score = count_total_3ap(A)
    best_A = set(A)
    best_score = current_score
    
    # Simulated Annealing Phase
    if time.time() < deadline - 2:
        T = 10.0
        cooling_rate = 0.9995
        min_T = 0.1
        
        # Precompute list of elements for faster random selection
        A_list = list(A)
        R_list = list(remaining)
        
        while time.time() < deadline - 2:
            if not A:
                # If A becomes empty, restart greedy or pick random
                # For simplicity, pick random from 1..N
                x = random.randint(1, N)
                if is_valid_add(x, set()):
                    A.add(x)
                    remaining.discard(x)
                    current_score = count_3ap_gain(x, A)
                    A_list = list(A)
                    R_list = list(remaining)
                    continue

            move_type = random.random()
            
            if move_type < 0.4:
                # Swap: remove a, add b
                if not A_list or not R_list:
                    continue
                a = random.choice(A_list)
                b = random.choice(R_list)
                
                # Check validity of b in A - {a}
                A.remove(a)
                if is_valid_add(b, A):
                    gain_b = count_3ap_gain(b, A)
                    loss_a = compute_loss(a, A | {b}) # Wait, loss of a is based on A before adding b?
                    # Actually, we need the change in score.
                    # Score(A) = S
                    # Score(A - {a} + {b}) = S - Loss(a, A) + Gain(b, A - {a})
                    # Loss(a, A) is the number of 3-APs in A that contain a.
                    # Gain(b, A - {a}) is the number of 3-APs in (A - {a}) that contain b.
                    
                    # Let's recompute carefully.
                    # current_score is Score(A).
                    # loss_a = compute_loss(a, A) # 3-APs in A involving a
                    # gain_b = count_3ap_gain(b, A - {a}) # 3-APs in A-a involving b
                    # new_score = current_score - loss_a + gain_b
                    
                    # I removed a from A. So A is now A - {a}.
                    gain_b = count_3ap_gain(b, A)
                    # loss_a was computed on the original A.
                    # I need loss_a from the state BEFORE removing a.
                    # Let's store it.
                    # This is getting messy with state mutation. 
                    # Let's revert and do it cleaner.
                    A.add(a)
                    
                    loss_a = compute_loss(a, A)
                    A.remove(a)
                    gain_b = count_3ap_gain(b, A)
                    A.add(a) # Restore for now, or keep removed?
                    
                    new_score = current_score - loss_a + gain_b
                    delta = new_score - current_score
                    
                    if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                        # Accept
                        A.remove(a)
                        A.add(b)
                        remaining.remove(b)
                        remaining.add(a)
                        current_score = new_score
                        A_list.remove(a)
                        A_list.append(b)
                        R_list.remove(b)
                        R_list.append(a)
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_A = set(A)
                else:
                    A.add(a) # Restore a
            else:
                # Add or Remove
                if move_type < 0.7:
                    # Add
                    if R_list:
                        b = random.choice(R_list)
                        if is_valid_add(b, A):
                            gain_b = count_3ap_gain(b, A)
                            new_score = current_score + gain_b
                            delta = gain_b
                            
                            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                                A.add(b)
                                remaining.remove(b)
                                current_score = new_score
                                A_list.append(b)
                                R_list.remove(b)
                                
                                if current_score > best_score:
                                    best_score = current_score
                                    best_A = set(A)
                else:
                    # Remove
                    if A_list:
                        a = random.choice(A_list)
                        loss_a = compute_loss(a, A)
                        new_score = current_score - loss_a
                        delta = -loss_a
                        
                        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                            A.remove(a)
                            remaining.add(a)
                            current_score = new_score
                            A_list.remove(a)
                            R_list.append(a)
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_A = set(A)
            
            T *= cooling_rate
            if T < min_T:
                T = min_T

    # Final polish: try 1-swap local search on best_A? 
    # Or just return best_A.
    # The SA might have found a better set.
    
    # Ensure best_A is valid (it should be, as we only accept valid moves)
    # And return sorted list
    return sorted(best_A)
