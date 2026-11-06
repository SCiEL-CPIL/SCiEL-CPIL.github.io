import random
import math

def is_valid_set(S):
    """
    Check if the sorted set S has distinct block products.
    """
    seq = sorted(S)
    n = len(seq)
    if n == 0:
        return True
    
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in prods:
                return False
            prods.add(prod)
    return True

def construct():
    all_nums = set(range(2, 129))
    
    # Initial state: Greedy
    # Try to build a sequence greedily
    S = set()
    for x in range(2, 129):
        if is_valid_set(S | {x}):
            S.add(x)
            
    best_S = S.copy()
    best_len = len(S)
    
    # Simulated Annealing / Random Walk
    # We will try to improve the length.
    # Since adding an element increases length by 1, we prioritize adds.
    
    current_S = S.copy()
    current_len = len(S)
    
    # Parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    # Number of iterations
    # 60s limit.
    # is_valid_set is O(L^2). L ~ 50-100.
    # 10000 iterations should be safe.
    
    for iter in range(20000):
        if T < min_T:
            break
            
        # Generate a neighbor
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_S = current_S.copy()
        
        if move_type == 'add':
            candidates = list(all_nums - current_S)
            if candidates:
                x = random.choice(candidates)
                new_S.add(x)
        elif move_type == 'remove':
            if current_S:
                x = random.choice(list(current_S))
                new_S.remove(x)
        elif move_type == 'swap':
            if current_S and (all_nums - current_S):
                x = random.choice(list(current_S))
                y = random.choice(list(all_nums - current_S))
                new_S.remove(x)
                new_S.add(y)
        
        if new_S == current_S:
            continue
            
        # Check validity
        if not is_valid_set(new_S):
            continue
            
        new_len = len(new_S)
        
        # Acceptance criterion
        if new_len > current_len:
            # Always accept improvement
            current_S = new_S
            current_len = new_len
            if current_len > best_len:
                best_len = current_len
                best_S = current_S.copy()
                # Reset temperature? Or just continue
        elif new_len == current_len:
            # Accept sideways move with some probability
            if random.random() < 0.5:
                current_S = new_S
                # current_len stays same
        else:
            # new_len < current_len
            # Accept with probability exp(-1/T) ?
            # Since we want max length, we rarely want to go down.
            # But it might help escape local minima.
            if random.random() < math.exp(-1.0 / T):
                current_S = new_S
                current_len = new_len
                
        T *= cooling
        
    return list(best_S)
