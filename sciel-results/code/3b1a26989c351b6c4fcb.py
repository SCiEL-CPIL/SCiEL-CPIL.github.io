import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    N = 256
    
    # Precompute all 4-APs and 3-APs as bitmasks for fast checking
    # 4-AP: (a, a+d, a+2d, a+3d)
    four_aps = []  # list of (mask, elements)
    three_aps = []  # list of (mask, elements)
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                three_aps.append((a, b, c))
    
    # Precompute for each element x, which 4-APs it participates in
    # and which 3-APs it participates in
    elem_in_4ap = {x: [] for x in range(1, N + 1)}
    for ap in four_aps:
        for x in ap:
            elem_in_4ap[x].append(ap)
    
    elem_in_3ap = {x: [] for x in range(1, N + 1)}
    for ap in three_aps:
        for x in ap:
            elem_in_3ap[x].append(ap)
    
    def is_valid_add(x, A):
        """Check if adding x to A creates a 4-AP."""
        for ap in elem_in_4ap[x]:
            others = [y for y in ap if y != x]
            if all(y in A for y in others):
                return False
        return True
    
    def count_3ap_gain(x, A):
        """Count 3-APs that would be created by adding x to A."""
        gain = 0
        for ap in elem_in_3ap[x]:
            others = [y for y in ap if y != x]
            if all(y in A for y in others):
                gain += 1
        return gain
    
    def count_total_3ap(A):
        """Count all 3-APs in A."""
        count = 0
        for ap in three_aps:
            if all(x in A for x in ap):
                count += 1
        return count
    
    def compute_loss(x, A):
        """Count 3-APs in A that contain x."""
        loss = 0
        for ap in elem_in_3ap[x]:
            others = [y for y in ap if y != x]
            if all(y in A for y in others):
                loss += 1
        return loss
    
    def greedy_build(A, remaining):
        """Greedy phase: pick element with max 3-AP gain that doesn't create 4-AP."""
        while remaining:
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
    
    def local_search(A, remaining, time_limit):
        """Aggressive 1-swap and add/remove local search."""
        A_list = list(A)
        R_list = list(remaining)
        current_score = count_total_3ap(A)
        best_A_local = set(A)
        best_score_local = current_score
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                break
            
            improved = False
            move_type = random.random()
            
            if move_type < 0.5 and A_list and R_list:
                # Swap: remove a, add b
                a = random.choice(A_list)
                b = random.choice(R_list)
                
                A.remove(a)
                if is_valid_add(b, A):
                    gain_b = count_3ap_gain(b, A)
                    loss_a = compute_loss(a, A | {b})
                    new_score = current_score - loss_a + gain_b
                    delta = new_score - current_score
                    
                    if delta >= 0:
                        A.add(b)
                        remaining.remove(b)
                        remaining.add(a)
                        current_score = new_score
                        A_list.remove(a)
                        A_list.append(b)
                        R_list.remove(b)
                        R_list.append(a)
                        no_improve_count = 0
                        improved = True
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_A_local = set(A)
                    else:
                        A.add(a)
                        no_improve_count += 1
                else:
                    A.add(a)
                    no_improve_count += 1
            elif move_type < 0.75 and R_list:
                # Add
                b = random.choice(R_list)
                if is_valid_add(b, A):
                    gain_b = count_3ap_gain(b, A)
                    new_score = current_score + gain_b
                    if new_score >= current_score:
                        A.add(b)
                        remaining.remove(b)
                        current_score = new_score
                        A_list.append(b)
                        R_list.remove(b)
                        no_improve_count = 0
                        improved = True
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_A_local = set(A)
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Remove
                if A_list:
                    a = random.choice(A_list)
                    loss_a = compute_loss(a, A)
                    new_score = current_score - loss_a
                    # Only remove if it allows future improvements (accept with some probability)
                    if random.random() < 0.3 or new_score > current_score:
                        A.remove(a)
                        remaining.add(a)
                        current_score = new_score
                        A_list.remove(a)
                        R_list.append(a)
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
        
        return best_A_local, best_score_local
    
    # Multi-restart strategy
    best_A = set()
    best_score = -1
    
    num_restarts = 8
    per_restart_time = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + 5.0 + (restart + 1) * per_restart_time
        if time.time() > restart_deadline - 2:
            break
        
        A = set()
        remaining = set(range(1, N + 1))
        
        # Greedy phase
        greedy_build(A, remaining)
        
        current_score = count_total_3ap(A)
        if current_score > best_score:
            best_score = current_score
            best_A = set(A)
        
        # Local search phase
        if time.time() < restart_deadline - 2:
            ls_A, ls_score = local_search(A, remaining, restart_deadline - 2)
            if ls_score > best_score:
                best_score = ls_score
                best_A = ls_A
    
    # Final polish on best_A
    if time.time() < deadline - 2 and best_A:
        remaining_final = set(range(1, N + 1)) - best_A
        A_final = set(best_A)
        final_A, final_score = local_search(A_final, remaining_final, deadline - 2)
        if final_score > best_score:
            best_score = final_score
            best_A = final_A
    
    return sorted(best_A)
