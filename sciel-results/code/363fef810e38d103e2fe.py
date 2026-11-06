import time
import random
from math import gcd

# Precomputation
N = 512
MAX_A = N - 1

def get_prime_factors(a):
    factors = set()
    d = 2
    while d * d <= a:
        while a % d == 0:
            factors.add(d)
            a //= d
        d += 1
    if a > 1:
        factors.add(a)
    return factors

# Map primes to indices for bitmask
primes = []
for i in range(2, MAX_A + 1):
    is_prime = True
    for p in primes:
        if p * p > i:
            break
        if i % p == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(i)

prime_to_idx = {p: i for i, p in enumerate(primes)}

masks = [0] * (MAX_A + 1)
for a in range(1, MAX_A + 1):
    factors = get_prime_factors(a)
    m = 0
    for p in factors:
        if p in prime_to_idx:
            m |= (1 << prime_to_idx[p])
    masks[a] = m

weights = [0.0] * (MAX_A + 1)
for a in range(1, MAX_A + 1):
    weights[a] = 1.0 / (N - a)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Greedy initialization
    chosen = []
    used_mask = 0
    for a in range(MAX_A, 0, -1):
        if (used_mask & masks[a]) == 0:
            chosen.append(a)
            used_mask |= masks[a]
    
    best_set = set(chosen)
    best_score = sum(weights[a] for a in best_set)
    current_set = set(chosen)
    current_score = best_score
    current_mask = used_mask
    
    # Simulated Annealing
    T = 0.5
    cooling = 0.9999
    min_T = 0.01
    
    # Candidate pool for additions/swaps: top 200 unchosen numbers
    # We will update this pool occasionally
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a number
            # Pick random from unchosen
            # To be efficient, pick from a random subset of unchosen
            # We can just pick a random number 1..511 and check if it's in current_set
            # But most are in current_set? No, current_set is small (~100 max).
            # So random pick is likely unchosen.
            b = random.randint(1, MAX_A)
            if b not in current_set:
                if (current_mask & masks[b]) == 0:
                    new_score = current_score + weights[b]
                    if new_score > current_score:
                        current_set.add(b)
                        current_mask |= masks[b]
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                    else:
                        # Accept with prob exp(-delta/T)
                        delta = current_score - new_score
                        if random.random() < pow(2.71828, -delta / T):
                            current_set.add(b)
                            current_mask |= masks[b]
                            current_score = new_score
        elif move_type < 0.8:
            # Try to swap
            if current_set:
                a = random.choice(list(current_set))
                b = random.randint(1, MAX_A)
                if b not in current_set:
                    # Check if b is coprime to current_set \ {a}
                    # mask_without_a = current_mask & ~masks[a]
                    mask_without_a = current_mask & ~masks[a]
                    if (mask_without_a & masks[b]) == 0:
                        new_score = current_score - weights[a] + weights[b]
                        if new_score > current_score:
                            current_set.remove(a)
                            current_set.add(b)
                            current_mask = (current_mask & ~masks[a]) | masks[b]
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
                        else:
                            delta = current_score - new_score
                            if random.random() < pow(2.71828, -delta / T):
                                current_set.remove(a)
                                current_set.add(b)
                                current_mask = (current_mask & ~masks[a]) | masks[b]
                                current_score = new_score
        else:
            # Try to remove a number (rarely helps, but helps escape local minima)
            if current_set:
                a = random.choice(list(current_set))
                new_score = current_score - weights[a]
                if new_score > current_score:
                    pass
                else:
                    delta = current_score - new_score
                    if random.random() < pow(2.71828, -delta / T):
                        current_set.remove(a)
                        current_mask &= ~masks[a]
                        current_score = new_score
        
        T *= cooling
        
    return list(best_set)
