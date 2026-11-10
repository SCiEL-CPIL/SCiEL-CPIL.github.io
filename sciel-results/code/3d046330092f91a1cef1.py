import time
import random
from math import gcd

# Precompute prime factors and masks
MAX_VAL = 512
PRIMES = []
IS_PRIME = [True] * MAX_VAL
IS_PRIME[0] = IS_PRIME[1] = False
for i in range(2, MAX_VAL):
    if IS_PRIME[i]:
        PRIMES.append(i)
        for j in range(i*i, MAX_VAL, i):
            IS_PRIME[j] = False

# mask[n] is a bitmask of indices of primes dividing n
MASKS = [0] * MAX_VAL
for n in range(2, MAX_VAL):
    m = n
    mask = 0
    for i, p in enumerate(PRIMES):
        if p * p > m:
            break
        if m % p == 0:
            mask |= (1 << i)
            while m % p == 0:
                m //= p
    if m > 1:
        # m is a prime
        # find its index
        # since m < MAX_VAL, it must be in PRIMES
        idx = PRIMES.index(m)
        mask |= (1 << idx)
    MASKS[n] = mask

# Value of each number
VALUES = [0.0] * MAX_VAL
for a in range(1, MAX_VAL):
    VALUES[a] = 1.0 / (512 - a)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_set = None
    best_score = 0.0
    
    # Helper to check compatibility
    def is_compatible(n, prime_counts):
        mask = MASKS[n]
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if prime_counts[idx] > 0:
                return False
            m ^= lsb
        return True

    def get_score(set_list):
        return sum(VALUES[a] for a in set_list)

    def greedy_init():
        set_list = []
        prime_counts = [0] * len(PRIMES)
        # Descending order
        for a in range(511, 0, -1):
            if is_compatible(a, prime_counts):
                set_list.append(a)
                mask = MASKS[a]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] += 1
                    m ^= lsb
        return set_list, prime_counts

    def local_search(set_list, prime_counts, time_limit):
        current_score = get_score(set_list)
        in_set = [False] * MAX_VAL
        for a in set_list:
            in_set[a] = True
        
        not_in_set = [a for a in range(1, MAX_VAL) if not in_set[a]]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1-Swap Hill Climbing
            best_gain = 0
            best_swap = None
            best_remove = None
            best_add = None
            
            # Try swaps
            for a in set_list:
                # Temporarily remove a
                mask_a = MASKS[a]
                m = mask_a
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] -= 1
                    m ^= lsb
                
                for b in not_in_set:
                    if is_compatible(b, prime_counts):
                        gain = VALUES[b] - VALUES[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (a, b)
                
                # Restore a
                m = mask_a
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] += 1
                    m ^= lsb
            
            # Try adds
            for b in not_in_set:
                if is_compatible(b, prime_counts):
                    gain = VALUES[b]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = b
            
            # Try removes (only if it allows future adds? 
            # For now, remove is only good if it unblocks high value adds.
            # But simple hill climbing might not capture that.
            # We'll rely on SA for that.
            
            if best_swap:
                a, b = best_swap
                set_list.remove(a)
                set_list.append(b)
                in_set[a] = False
                in_set[b] = True
                not_in_set.remove(a) # a is now out
                not_in_set.remove(b) # b is now in
                # Update prime counts
                mask_a = MASKS[a]
                m = mask_a
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] -= 1
                    m ^= lsb
                mask_b = MASKS[b]
                m = mask_b
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] += 1
                    m ^= lsb
                current_score += best_gain
                improved = True
            elif best_add is not None:
                b = best_add
                set_list.append(b)
                in_set[b] = True
                not_in_set.remove(b)
                mask_b = MASKS[b]
                m = mask_b
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    prime_counts[idx] += 1
                    m ^= lsb
                current_score += best_gain
                improved = True
        
        # Simulated Annealing phase
        T = 0.1
        T_min = 0.001
        cooling = 0.99
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < time_limit:
            if stuck_count > max_stuck:
                break
            
            move_type = random.choice(['swap', 'add', 'remove'])
            new_set_list = set_list.copy()
            new_prime_counts = prime_counts.copy()
            new_in_set = in_set.copy()
            new_not_in_set = not_in_set.copy()
            
            delta = 0.0
            valid_move = False
            
            if move_type == 'swap' and set_list:
                a = random.choice(set_list)
                # Find compatible b
                compatible_bs = []
                mask_a = MASKS[a]
                m = mask_a
                # Temp remove a
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    new_prime_counts[idx] -= 1
                    m ^= lsb
                
                for b in new_not_in_set:
                    if is_compatible(b, new_prime_counts):
                        compatible_bs.append(b)
                        
                if compatible_bs:
                    b = random.choice(compatible_bs)
                    # Apply swap
                    new_set_list.remove(a)
                    new_set_list.append(b)
                    new_in_set[a] = False
                    new_in_set[b] = True
                    new_not_in_set.remove(a)
                    new_not_in_set.remove(b)
                    
                    # Update counts for b
                    mask_b = MASKS[b]
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        new_prime_counts[idx] += 1
                        m ^= lsb
                        
                    delta = VALUES[b] - VALUES[a]
                    valid_move = True
            elif move_type == 'add' and new_not_in_set:
                b = random.choice(new_not_in_set)
                if is_compatible(b, new_prime_counts):
                    new_set_list.append(b)
                    new_in_set[b] = True
                    new_not_in_set.remove(b)
                    mask_b = MASKS[b]
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        new_prime_counts[idx] += 1
                        m ^= lsb
                    delta = VALUES[b]
                    valid_move = True
            elif move_type == 'remove' and set_list:
                a = random.choice(set_list)
                new_set_list.remove(a)
                new_in_set[a] = False
                new_not_in_set.append(a)
                mask_a = MASKS[a]
                m = mask_a
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    new_prime_counts[idx] -= 1
                    m ^= lsb
                delta = -VALUES[a]
                valid_move = True
            
            if valid_move:
                import math
                if delta < 0 or random.random() < math.exp(delta / T):
                    set_list = new_set_list
                    prime_counts = new_prime_counts
                    in_set = new_in_set
                    not_in_set = new_not_in_set
                    current_score += delta
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    stuck_count = max_stuck // 2 # Restart cooling somewhat?
        
        return set_list, current_score

    # Multi-start
    for i in range(10):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if i == 0:
            init_set, init_counts = greedy_init()
        else:
            # Randomized greedy
            init_set = []
            init_counts = [0] * len(PRIMES)
            candidates = list(range(511, 0, -1))
            random.shuffle(candidates)
            for a in candidates:
                if is_compatible(a, init_counts):
                    init_set.append(a)
                    mask = MASKS[a]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        init_counts[idx] += 1
                        m ^= lsb
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
            
        # Run local search with a fraction of the time
        ls_time = time.time() + min(10, time_remaining * 0.5)
        res_set, res_score = local_search(init_set, init_counts, ls_time)
        
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    if best_set is None:
        # Fallback
        init_set, _ = greedy_init()
        best_set = init_set
        
    return best_set

