"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute conflict pairs for each x
    # conflicts[x] contains pairs (a, b) such that {a, b, x} is a 3-AP
    # We assume a < b in the pair
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        pairs = []
        # 1) x is middle: a + b = 2x => b = 2x - a
        # a < x < b
        for a in range(1, x):
            b = 2 * x - a
            if x < b <= N:
                pairs.append((a, b))
        # 2) x is largest: a + x = 2b => b = (a + x) / 2
        # a < b < x
        for a in range(1, x):
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x:
                    pairs.append((a, b))
        # 3) x is smallest: x + b = 2a => a = (x + b) / 2
        # x < a < b
        for b in range(x + 1, N + 1):
            if (x + b) % 2 == 0:
                a = (x + b) // 2
                if x < a < b:
                    pairs.append((a, b))
        # Remove duplicates and sort
        pairs = list(set(pairs))
        pairs.sort()
        conflicts[x] = pairs

    # Reciprocals
    recips = [Fraction(0, 1)] + [Fraction(1, i) for i in range(1, N + 1)]
    
    def is_valid_add(mask, x):
        """Check if adding x to mask is valid."""
        for a, b in conflicts[x]:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                return False
        return True

    def calc_sum(mask):
        s = Fraction(0, 1)
        for i in range(1, N + 1):
            if (mask >> (i - 1)) & 1:
                s += recips[i]
        return s

    # Generate Initial Candidates
    
    # Candidate 1: Greedy (smallest first)
    mask_greedy = 0
    for x in range(1, N + 1):
        if is_valid_add(mask_greedy, x):
            mask_greedy |= (1 << (x - 1))
            
    # Candidate 2: Cantor Set (no digit 2 in base 3)
    mask_cantor = 0
    for x in range(1, N + 1):
        # Check base 3 digits
        n = x
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            mask_cantor |= (1 << (x - 1))
            
    # Candidate 3: Greedy on Cantor? Or just Cantor.
    # Let's also try a greedy that prioritizes small numbers but allows backtracking? 
    # No, let's just run local search on both.
    
    candidates = [mask_greedy, mask_cantor]
    
    best_mask = 0
    best_sum = Fraction(0, 1)
    
    for init_mask in candidates:
        current_mask = init_mask
        current_sum = calc_sum(current_mask)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_mask = current_mask
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try single additions
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if not ((current_mask >> (x - 1)) & 1):
                    if is_valid_add(current_mask, x):
                        new_sum = current_sum + recips[x]
                        if new_sum > current_sum:
                            current_mask |= (1 << (x - 1))
                            current_sum = new_sum
                            improved = True
            
            # 2. Try 1-for-1 swaps
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if (current_mask >> (x - 1)) & 1:
                    trial_mask = current_mask & ~(1 << (x - 1))
                    for y in range(1, N + 1):
                        if time.time() > deadline:
                            break
                        if y == x:
                            continue
                        if not ((trial_mask >> (y - 1)) & 1):
                            if is_valid_add(trial_mask, y):
                                new_sum = current_sum - recips[x] + recips[y]
                                if new_sum > current_sum:
                                    current_mask = trial_mask | (1 << (y - 1))
                                    current_sum = new_sum
                                    improved = True
                                    break # Restart scan after a swap
            
            # 3. Try 1-for-2 swaps (Remove x, Add y, z)
            # This is expensive, so we only do it if time permits and we haven't improved in other ways recently?
            # Or just do it every iteration if fast enough.
            # Optimization: Only try if x is large and y, z are small.
            for x in range(N, 0, -1): # Remove larger x first
                if time.time() > deadline:
                    break
                if (current_mask >> (x - 1)) & 1:
                    trial_mask = current_mask & ~(1 << (x - 1))
                    base_sum = current_sum - recips[x]
                    
                    # Try to find two elements y, z to add
                    # y should be small
                    found_pair = False
                    for y in range(1, x): # y < x to ensure potential gain 1/y + 1/z > 1/x
                        if time.time() > deadline:
                            break
                        if not ((trial_mask >> (y - 1)) & 1):
                            if is_valid_add(trial_mask, y):
                                mask_with_y = trial_mask | (1 << (y - 1))
                                # Now find z
                                for z in range(1, x): # z < x
                                    if time.time() > deadline:
                                        break
                                    if z == y:
                                        continue
                                    if not ((mask_with_y >> (z - 1)) & 1):
                                        if is_valid_add(mask_with_y, z):
                                            new_sum = base_sum + recips[y] + recips[z]
                                            if new_sum > current_sum:
                                                current_mask = mask_with_y | (1 << (z - 1))
                                                current_sum = new_sum
                                                improved = True
                                                found_pair = True
                                                break
                                if found_pair:
                                    break
                    if found_pair:
                        break # Restart scan after a 1-for-2 swap
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = current_mask

    # Extract final list
    final_list = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            final_list.append(x)
    
    return final_list
# EVOLVE-BLOCK-END
