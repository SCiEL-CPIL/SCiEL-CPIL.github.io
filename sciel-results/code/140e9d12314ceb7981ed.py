import time
import random
import sys

N = 256
TIME_LIMIT = 55.0  # Safety margin from 60s

def _precompute():
    # Precompute 3-AP and 4-AP masks
    m3 = []
    for d in range(1, N):
        for a in range(1, N - 2*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1))
            m3.append(mask)
            
    m4 = []
    for d in range(1, N):
        for a in range(1, N - 3*d + 1):
            mask = (1 << (a-1)) | (1 << (a+d-1)) | (1 << (a+2*d-1)) | (1 << (a+3*d-1))
            m4.append(mask)
            
    # For each element x (0-indexed), compute lists of "other bits" masks
    x_4ap_others = [[] for _ in range(N)]
    for m in m4:
        bits = []
        for i in range(N):
            if m & (1 << i):
                bits.append(i)
        for b in bits:
            others = m ^ (1 << b)
            x_4ap_others[b].append(others)
            
    x_3ap_others = [[] for _ in range(N)]
    for m in m3:
        bits = []
        for i in range(N):
            if m & (1 << i):
                bits.append(i)
        for b in bits:
            others = m ^ (1 << b)
            x_3ap_others[b].append(others)
            
    return x_4ap_others, x_3ap_others, m3, m4

X_4AP_OTHERS, X_3AP_OTHERS, M3_LIST, M4_LIST = _precompute()

def _is_valid_addition(x, current_mask):
    """Check if adding bit x to current_mask creates a 4-AP."""
    if current_mask & (1 << x):
        return True
    for others_mask in X_4AP_OTHERS[x]:
        if (current_mask & others_mask) == others_mask:
            return False
    return True

def _count_new_3aps(x, current_mask):
    """Count how many 3-APs are formed by adding bit x to current_mask."""
    count = 0
    for others_mask in X_3AP_OTHERS[x]:
        if (current_mask & others_mask) == others_mask:
            count += 1
    return count

def _count_3aps(mask):
    """Count total 3-APs in mask."""
    total = 0
    for x in range(N):
        if mask & (1 << x):
            total += _count_new_3aps(x, mask)
    return total // 3

def _greedy_construct():
    """Construct a 4-AP-free set using greedy method."""
    mask = 0
    remaining = set(range(N))
    while remaining:
        best_x = None
        best_score = -1
        for x in remaining:
            if _is_valid_addition(x, mask):
                score = _count_new_3aps(x, mask)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x is None:
            break
        mask |= (1 << best_x)
        remaining.remove(best_x)
    return mask

def _tabu_search(mask, time_limit):
    """
    Tabu Search for maximizing 3-APs with no 4-APs.
    """
    current_mask = mask
    current_score = _count_3aps(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    # Tabu list: stores (bit, expiration_iteration)
    tabu_list = []
    TABU_TENURE = 20
    
    start_time = time.time()
    no_improve_count = 0
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Clean up expired tabu entries
        tabu_list = [(b, exp) for b, exp in tabu_list if exp > iteration]
        
        # Generate candidate moves: Add, Remove, Swap
        # We'll collect the best move that is not tabu, or if all are tabu, the best overall.
        
        best_move = None
        best_delta = -10**9
        best_is_tabu = False
        
        # 1. Add moves
        for x in range(N):
            if current_mask & (1 << x):
                continue
            if _is_valid_addition(x, current_mask):
                delta = _count_new_3aps(x, current_mask)
                is_tabu = any(b == x for b, exp in tabu_list)
                if not is_tabu:
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('add', x)
                        best_is_tabu = False
                else:
                    if delta > best_delta or best_is_tabu:
                        # Aspiration: if it improves global best, allow tabu move
                        if current_score + delta > best_score:
                            best_delta = delta
                            best_move = ('add', x)
                            best_is_tabu = False
                        elif best_is_tabu and delta > best_delta:
                            best_delta = delta
                            best_move = ('add', x)
                            best_is_tabu = True
        
        # 2. Swap moves (Remove x, Add y)
        # Precompute loss for all x in mask
        loss_cache = {}
        for x in range(N):
            if current_mask & (1 << x):
                loss_cache[x] = _count_new_3aps(x, current_mask)
        
        for x in range(N):
            if not (current_mask & (1 << x)):
                continue
            
            x_tabu = any(b == x for b, exp in tabu_list)
            temp_mask = current_mask ^ (1 << x)
            loss = loss_cache[x]
            
            for y in range(N):
                if temp_mask & (1 << y):
                    continue
                
                y_tabu = any(b == y for b, exp in tabu_list)
                move_tabu = x_tabu or y_tabu
                
                if _is_valid_addition(y, temp_mask):
                    gain = _count_new_3aps(y, temp_mask)
                    delta = gain - loss
                    
                    if not move_tabu:
                        if delta > best_delta:
                            best_delta = delta
                            best_move = ('swap', x, y)
                            best_is_tabu = False
                    else:
                        if current_score + delta > best_score:
                            if delta > best_delta:
                                best_delta = delta
                                best_move = ('swap', x, y)
                                best_is_tabu = False
                        elif best_is_tabu and delta > best_delta:
                            best_delta = delta
                            best_move = ('swap', x, y)
                            best_is_tabu = True
        
        # 3. Remove moves (only if stuck or to escape local opt)
        # Usually not beneficial for score, but can enable future adds.
        # We'll only consider remove if no add/swap found or if it leads to a better state later?
        # For simplicity, skip pure removes in tabu unless we add a "2-swap" or "remove then add" logic.
        # The swap logic above covers remove+add. Pure remove reduces score.
        # However, sometimes removing a low-value element allows adding two high-value ones?
        # That's a 2-move. Tabu search typically handles 1-move.
        # Let's stick to Add and Swap.
        
        if best_move is None:
            # No moves? Shouldn't happen unless mask is full or stuck.
            # If stuck, perturb.
            break
            
        # Apply move
        if best_move[0] == 'add':
            x = best_move[1]
            current_mask |= (1 << x)
            current_score += best_delta
            tabu_list.append((x, iteration + TABU_TENURE))
            
        elif best_move[0] == 'swap':
            x, y = best_move[1], best_move[2]
            current_mask = (current_mask ^ (1 << x)) | (1 << y)
            current_score += best_delta
            tabu_list.append((x, iteration + TABU_TENURE))
            tabu_list.append((y, iteration + TABU_TENURE))
            
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Escape strategy: if stuck for too long, perturb
        if no_improve_count > 30:
            # Randomly flip 2-5 bits and repair
            num_flips = random.randint(2, 5)
            bits_in_mask = [i for i in range(N) if current_mask & (1 << i)]
            bits_not_in_mask = [i for i in range(N) if not (current_mask & (1 << i))]
            
            for _ in range(num_flips):
                if random.random() < 0.5 and bits_in_mask:
                    bit = random.choice(bits_in_mask)
                    current_mask &= ~(1 << bit)
                elif bits_not_in_mask:
                    bit = random.choice(bits_not_in_mask)
                    if _is_valid_addition(bit, current_mask):
                        current_mask |= (1 << bit)
            
            # Repair 4-APs
            changed = True
            while changed:
                changed = False
                for m in M4_LIST:
                    if (current_mask & m) == m:
                        bits_in_ap = [b for b in range(N) if m & (1 << b)]
                        b_to_remove = random.choice(bits_in_ap)
                        current_mask &= ~(1 << b_to_remove)
                        changed = True
                        break
            
            current_score = _count_3aps(current_mask)
            no_improve_count = 0
            # Clear tabu list after perturbation
            tabu_list = []

    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_mask = 0
    best_score = -1
    
    # Multi-start
    num_restarts = 10
    time_per_restart = (TIME_LIMIT - 5.0) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate initial mask
        if i == 0:
            mask = _greedy_construct()
        else:
            # Perturb best found
            mask = best_mask
            num_flips = random.randint(5, 15)
            for _ in range(num_flips):
                bit = random.randint(0, N-1)
                mask ^= (1 << bit)
            
            # Repair: ensure no 4-APs
            changed = True
            while changed:
                changed = False
                for m in M4_LIST:
                    if (mask & m) == m:
                        bits_in_ap = [b for b in range(N) if m & (1 << b)]
                        b_to_remove = random.choice(bits_in_ap)
                        mask &= ~(1 << b_to_remove)
                        changed = True
                        break
        
        score = _count_3aps(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Run tabu search
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break
            
        ls_mask, ls_score = _tabu_search(mask, remaining_time - 1.0)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    print(f"Fitness: {_count_3aps(mask)}")
    
    # Validate
    valid = True
    for m in M4_LIST:
        if (mask & m) == m:
            valid = False
            break
    print(f"Valid: {valid}")
