import time
import random
from math import gcd

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute prime factorization exponents for each number
# For each number n, store a tuple of (prime, exponent) for primes <= 256
# Actually, we can store a dictionary or just the exponents for primes up to 256.
# Let's use a fixed list of primes.
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
# Actually, we only need primes up to 256.
# Let's just factorize on the fly or precompute a matrix of exponents.
# exp[n][p_idx] = exponent of p in n.

P_count = len(primes)
exp_matrix = [[0] * P_count for _ in range(NMAX + 1)]
for n in range(2, NMAX + 1):
    temp = n
    for p_idx, p in enumerate(primes):
        if p * p > temp:
            break
        while temp % p == 0:
            exp_matrix[n][p_idx] += 1
            temp //= p
    if temp > 1:
        # temp is a prime
        if temp in val_to_idx: # It's in our range, but we need its index in primes list
             # Find index of temp in primes
             for p_idx, p in enumerate(primes):
                 if p == temp:
                     exp_matrix[n][p_idx] += 1
                     break

# Precompute valid_c_mask
# valid_c_mask[a_idx][b_idx] = bitmask of c indices such that a | b*c
valid_c_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        g = gcd(a, b)
        req = a // g
        mask = 0
        # c must be a multiple of req
        # Iterate through multiples of req in [2, 256]
        for c_val in range(2, NMAX + 1):
            if c_val % req == 0:
                mask |= (1 << val_to_idx[c_val])
        valid_c_mask[i][j] = mask

def get_degrees(S_mask):
    degrees = [0] * N
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    for i in a_list:
        deg = 0
        for j in a_list:
            if j == i:
                continue
            valid_mask = valid_c_mask[i][j]
            count_mask = S_mask & valid_mask & ~(1 << i) & ~(1 << j)
            deg += count_mask.bit_count()
        degrees[i] = deg
    return degrees

def can_add(x, S_set, S_mask):
    x_idx = val_to_idx[x]
    # Check if x can be added to S
    # Conditions for all distinct a,b,c in S U {x}:
    # If x is the 'a' in a|bc: x | b*c for some b,c in S.
    # If x is the 'b' in a|xc: a | x*c for some a,c in S.
    # If x is the 'c' in a|xb: a | x*b for some a,b in S.
    
    # Check 1: x | b*c
    # This is equivalent to: exists b,c in S, b!=c, such that x | b*c.
    # We can check this by iterating b in S, and checking if there exists c in S, c!=b, such that (x/gcd(x,b)) | c.
    S_list = list(S_set)
    for b in S_list:
        g = gcd(x, b)
        req = x // g
        # Check if any c in S (c != b) is a multiple of req
        for c in S_list:
            if c == b:
                continue
            if c % req == 0:
                return False
    
    # Check 2: a | x*c
    # This is equivalent to: exists a,c in S, a!=c, such that a | x*c.
    # Iterate a in S, check if exists c in S, c!=a, such that (a/gcd(a,x)) | c.
    for a in S_list:
        g = gcd(a, x)
        req = a // g
        for c in S_list:
            if c == a:
                continue
            if c % req == 0:
                return False
    
    # Check 3: a | x*b
    # This is the same as Check 2 but with b and c swapped.
    # Since we iterate all a and all c in S, and the condition is symmetric in the "other" two variables relative to the divisor?
    # Wait.
    # Condition: for all distinct a,b,c, a does not divide b*c.
    # If we add x, the new triples are those containing x.
    # Case 1: x is the divisor. x | b*c. (Covered by Check 1)
    # Case 2: x is the first non-divisor. a | x*c. (Covered by Check 2)
    # Case 3: x is the second non-divisor. a | b*x. (Covered by Check 2, just swap names of b and c in the logic, but the set S is the same, so the existence of such a pair is the same).
    # Yes, Check 2 covers both Case 2 and Case 3 because it checks if there exists ANY pair (u, v) in S such that u | x*v.
    
    return True

def greedy_removal(deadline, tie_breaker='largest'):
    S_mask = (1 << N) - 1
    degrees = get_degrees(S_mask)
    
    while time.time() < deadline:
        max_deg = -1
        best_idx = -1
        for i in range(N):
            if not (S_mask >> i) & 1:
                continue
            deg = degrees[i]
            if deg > max_deg:
                max_deg = deg
                best_idx = i
            elif deg == max_deg:
                if best_idx == -1:
                    best_idx = i
                else:
                    val_i = idx_to_val[i]
                    val_best = idx_to_val[best_idx]
                    if tie_breaker == 'largest':
                        if val_i > val_best:
                            best_idx = i
                    elif tie_breaker == 'smallest':
                        if val_i < val_best:
                            best_idx = i
                    else:
                        if random.random() < 0.5:
                            best_idx = i
        
        if best_idx == -1:
            break
            
        if max_deg == 0:
            break
            
        S_mask &= ~(1 << best_idx)
        degrees = get_degrees(S_mask)
        
    res = []
    for i in range(N):
        if (S_mask >> i) & 1:
            res.append(idx_to_val[i])
    return res

def local_search(S, deadline):
    S_set = set(S)
    S_mask = 0
    for v in S_set:
        S_mask |= (1 << val_to_idx[v])
    
    best_S = list(S_set)
    best_size = len(S_set)
    
    all_elements = set(RANGE)
    removed = list(all_elements - S_set)
    
    degrees = get_degrees(S_mask)
    
    # We will try to add elements from 'removed' to S.
    # If we can add x, we do it.
    # If not, we try to swap x with y in S (where y is "least useful" or random).
    
    while time.time() < deadline:
        improved = False
        random.shuffle(removed)
        
        for x in removed:
            if can_add(x, S_set, S_mask):
                S_set.add(x)
                S_mask |= (1 << val_to_idx[x])
                removed.remove(x)
                if len(S_set) > best_size:
                    best_S = list(S_set)
                    best_size = len(S_set)
                improved = True
                degrees = get_degrees(S_mask)
                break # Restart loop with new S
                
        if improved:
            continue
            
        # If no direct addition, try swaps
        # Try swapping x with y
        # To make this faster, we can limit the number of swaps tried per pass.
        swaps_tried = 0
        max_swaps_per_pass = 100
        
        for x in removed:
            if swaps_tried >= max_swaps_per_pass:
                break
                
            # Find a candidate y in S to swap with x.
            # Heuristic: pick y with lowest degree? Or random?
            # Let's pick a few random y's.
            S_list = list(S_set)
            random.shuffle(S_list)
            
            for y in S_list[:5]: # Try 5 random y's for this x
                # Check if x can be added to S - {y}
                S_temp_set = S_set - {y}
                if can_add(x, S_temp_set, 0): # Pass 0 for mask as can_add uses S_set
                    # Perform swap
                    S_set.remove(y)
                    S_set.add(x)
                    S_mask &= ~(1 << val_to_idx[y])
                    S_mask |= (1 << val_to_idx[x])
                    removed.remove(x)
                    removed.append(y)
                    if len(S_set) > best_size:
                        best_S = list(S_set)
                        best_size = len(S_set)
                    improved = True
                    degrees = get_degrees(S_mask)
                    swaps_tried += 1
                    break
            if improved:
                break
                
        if not improved:
            # If we couldn't add or swap, we are stuck.
            # We could try a more aggressive restart or just break.
            # For now, break to avoid infinite loop if no improvements.
            # But maybe we should try different tie breakers in greedy first?
            # The local search is meant to polish. If it fails, we stop.
            break
            
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Run greedy removal with different tie breakers
    for tb in ['largest', 'smallest', 'random']:
        if time.time() > deadline - 15:
            break
        c = greedy_removal(deadline - 15, tb)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Local search on best
    if time.time() < deadline - 5:
        c = local_search(best, deadline - 5)
        if len(c) > best_size:
            best = c
            best_size = len(c)
            
    # Final verification
    res = list(best)
    n = len(res)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(n):
                if i == k or j == k: continue
                if (res[j] * res[k]) % res[i] == 0:
                    return [256]
    return res
