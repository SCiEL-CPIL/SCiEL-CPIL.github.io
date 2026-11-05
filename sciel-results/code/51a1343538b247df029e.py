import time
import random
import sys

N = 256
DEADLINE_MARGIN = 5  # seconds before the 90s limit

def bit(x):
    return 1 << (x - 1)

def in_mask(mask, x):
    return (mask >> (x - 1)) & 1

# Precomputation
# 1. 4-AP Constraints
# For each x, store the other 3 elements of any 4-AP containing x.
# four_ap_partners[x] = list of tuples (a, b, c) such that {x, a, b, c} is a 4-AP.
four_ap_partners = [[] for _ in range(N + 1)]

# Precompute all 4-APs to fill this structure
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            # The 4-AP is (a, b, c, e)
            for x in (a, b, c, e):
                others = tuple(y for y in (a, b, c, e) if y != x)
                four_ap_partners[x].append(others)

# 2. 3-AP Witnesses
# For each x, store pairs (u, v) such that if u and v are in S, adding x creates a 3-AP.
# This includes x as middle, left, or right.
three_ap_witnesses = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x is middle: a + c = 2x, a < x < c
    for a in range(1, x):
        c = 2 * x - a
        if c <= N and c > x:
            three_ap_witnesses[x].append((a, c))
    
    # x is left: x, b, c -> b = x+d, c = x+2d
    for d in range(1, (N - x) // 2 + 1):
        b = x + d
        c = x + 2 * d
        if c <= N:
            three_ap_witnesses[x].append((b, c))
            
    # x is right: a, b, x -> b = x-d, a = x-2d
    for d in range(1, (x - 1) // 2 + 1):
        b = x - d
        a = x - 2 * d
        if a >= 1:
            three_ap_witnesses[x].append((a, b))

def count_3aps_from_mask(mask):
    """Calculate total 3-APs in the set represented by mask."""
    count = 0
    for b in range(1, N + 1):
        if in_mask(mask, b):
            # Only count if b is the middle to avoid double counting?
            # No, a 3-AP (a, b, c) is unique by its middle b.
            # So we sum over all b in S the number of pairs (a, c) in S with a+c=2b.
            # This is exactly what three_ap_witnesses[b] provides, BUT
            # three_ap_witnesses[b] also contains cases where b is left or right.
            # If we sum over all b in S using ALL witnesses, we will count each 3-AP 3 times
            # (once for each element being the 'center' of the witness list).
            # Let's verify:
            # 3-AP (1, 2, 3).
            # Witnesses for 1: (2,3) [1 is left]
            # Witnesses for 2: (1,3) [2 is middle]
            # Witnesses for 3: (1,2) [3 is right]
            # If 1,2,3 are in S:
            # Sum over b=1: sees (2,3) in S -> +1
            # Sum over b=2: sees (1,3) in S -> +1
            # Sum over b=3: sees (1,2) in S -> +1
            # Total = 3. But there is only 1 3-AP.
            # So we must divide by 3? Or only count middle witnesses?
            # It is cleaner to precompute ONLY middle witnesses for counting.
            pass
    return count

# Let's re-verify the counting strategy.
# It is much faster to just use the middle witness logic for counting.
# Let's create a separate list for counting.
middle_witnesses = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for a in range(1, x):
        c = 2 * x - a
        if c <= N and c > x:
            middle_witnesses[x].append((a, c))

def count_3aps(mask):
    count = 0
    for b in range(1, N + 1):
        if in_mask(mask, b):
            for a, c in middle_witnesses[b]:
                if in_mask(mask, a) and in_mask(mask, c):
                    count += 1
    return count

def get_delta_3aps(x, mask):
    """Calculate change in 3-AP count if x is added to mask."""
    delta = 0
    for a, c in three_ap_witnesses[x]:
        if in_mask(mask, a) and in_mask(mask, c):
            delta += 1
    return delta

def can_add(x, mask):
    """Check if adding x to mask creates a 4-AP."""
    for others in four_ap_partners[x]:
        a, b, c = others
        if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c):
            return False
    return True

def greedy_reverse_construct(seed_val):
    """
    Start with all elements. Remove elements to break 4-APs.
    To maximize 3-APs, remove the element that is part of the fewest 3-APs?
    Or just random?
    Let's try random removal from the first found 4-AP.
    """
    random.seed(seed_val)
    mask = (1 << N) - 1  # All bits set
    
    # Precompute list of all 4-APs for fast checking?
    # Or just iterate.
    # Since we are removing elements, we can just check validity.
    
    # Optimization: Maintain a list of "active" 4-APs?
    # For simplicity, let's just loop until valid.
    # N=256, 4-APs ~ 16k.
    # Each check is O(1) per 4-AP if we use bitmask?
    # No, checking if a 4-AP (a,b,c,d) is in mask is 4 bit checks.
    
    # Let's precompute the list of all 4-APs as tuples of bits?
    # No, just indices.
    
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
                
    # We will repeatedly find a 4-AP in the current mask and remove one element.
    # To be efficient, we can shuffle the 4-APs and check them.
    
    while True:
        # Check if current mask is valid
        valid = True
        for ap in all_4aps:
            a, b, c, e = ap
            if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c) and in_mask(mask, e):
                # Found a 4-AP
                valid = False
                # Choose an element to remove
                # Strategy: Remove the one that minimizes loss of 3-APs?
                # Or random?
                # Let's try removing the one with the lowest "degree" in 3-APs?
                # Calculating degree is expensive.
                # Let's just pick randomly from the 4.
                # Or, to be smarter, pick the one that is "least useful".
                # Let's try random first.
                elem_to_remove = random.choice([a, b, c, e])
                mask &= ~bit(elem_to_remove)
                break
        
        if valid:
            break
            
    return mask

def simulated_annealing(start_mask, time_limit):
    """
    Perform simulated annealing on the mask.
    """
    start_time = time.time()
    current_mask = start_mask
    current_score = count_3aps(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Precompute lists for random selection
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit and T > min_T:
        # Pick a random move type
        # 0: Add, 1: Remove, 2: Swap
        move_type = random.randint(0, 2)
        
        new_mask = current_mask
        delta = 0
        
        if move_type == 0:
            # Add
            # Pick random x not in mask
            # To avoid infinite loop if mask is full, check size
            if bin(current_mask).count('1') >= N:
                continue
            # Find a random element not in mask
            # This is O(N) if we do it naively.
            # Better: keep a list of available elements?
            # Or just pick random x in 1..N and check if in mask.
            # Expected trials ~ N / (N - |S|). If |S| is close to N, this is slow.
            # But |S| is usually < N/2.
            x = random.choice(all_elements)
            if in_mask(current_mask, x):
                continue
            if not can_add(x, current_mask):
                continue
            delta = get_delta_3aps(x, current_mask)
            new_mask = current_mask | bit(x)
            
        elif move_type == 1:
            # Remove
            if current_mask == 0:
                continue
            # Pick random x in mask
            # Again, naive approach: pick random x, check if in mask.
            x = random.choice(all_elements)
            if not in_mask(current_mask, x):
                continue
            # Removing x always valid for 4-APs
            # Delta is negative
            # get_delta_3aps(x, mask_without_x) gives the loss?
            # No, get_delta_3aps(x, M) is gain if added to M.
            # If we remove x from S, the loss is the number of 3-APs that x was part of.
            # This is get_delta_3aps(x, S \ {x}).
            mask_without_x = current_mask & ~bit(x)
            loss = get_delta_3aps(x, mask_without_x)
            delta = -loss
            new_mask = mask_without_x
            
        else:
            # Swap
            # Pick x not in mask, y in mask
            # We need to try a few times to find valid x, y
            swapped = False
            for _ in range(5):
                x = random.choice(all_elements)
                y = random.choice(all_elements)
                if in_mask(current_mask, x) or not in_mask(current_mask, y):
                    continue
                # Check if adding x to (mask - y) is valid
                temp_mask = current_mask & ~bit(y)
                if not can_add(x, temp_mask):
                    continue
                
                # Calculate delta
                # New score = Score(mask - y) + Gain(x, mask - y)
                # Score(mask) = Score(mask - y) + Loss(y, mask - y)
                # Delta = Gain(x, mask - y) - Loss(y, mask - y)
                mask_without_y = temp_mask
                gain = get_delta_3aps(x, mask_without_y)
                loss = get_delta_3aps(y, mask_without_y)
                delta = gain - loss
                new_mask = mask_without_y | bit(x)
                swapped = True
                break
            
            if not swapped:
                continue
        
        # Acceptance criterion
        if delta > 0:
            current_mask = new_mask
            current_score += delta
        else:
            # Boltzmann probability
            # P = exp(delta / T)
            # delta is negative
            try:
                prob = 2.718281828 ** (delta / T)
            except OverflowError:
                prob = 0.0
            if random.random() < prob:
                current_mask = new_mask
                current_score += delta
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        T *= cooling_rate
        
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    best_mask = 0
    best_score = -1
    
    # Strategy 1: Reverse Greedy + Simulated Annealing
    # Run multiple times with different seeds
    
    # Estimate time per SA run
    # Let's run as many as possible
    
    trial = 0
    while time.time() - start_time < DEADLINE_MARGIN:
        trial += 1
        # Get initial mask
        init_mask = greedy_reverse_construct(trial)
        
        # Run SA
        # Allocate remaining time minus a small buffer for next trial
        # We don't know how many trials we can do, so let's just run SA with a fixed small time
        # or dynamic time.
        # Let's run SA for 2 seconds per trial.
        time_limit = min(2.0, DEADLINE_MARGIN - (time.time() - start_time))
        if time_limit <= 0:
            break
            
        sa_mask, sa_score = simulated_annealing(init_mask, time_limit)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
            
    # Fallback
    if best_mask == 0:
        # Simple valid set: all odd numbers?
        # 1, 3, 5, ...
        # 4-AP in odds?
        # a, a+d, a+2d, a+3d. If a is odd, d must be even for all to be odd?
        # If d is even, a, a+2, a+4, a+6 are all odd.
        # So odds contain 4-APs.
        # Just pick {1}.
        best_mask = bit(1)
        
    # Convert mask to list
    result = []
    for x in range(1, N + 1):
        if in_mask(best_mask, x):
            result.append(x)
            
    return result

def validate(result):
    mask = 0
    for x in result:
        mask |= bit(x)
    # Check for 4-APs
    for ap in all_4aps:
        a, b, c, e = ap
        if in_mask(mask, a) and in_mask(mask, b) and in_mask(mask, c) and in_mask(mask, e):
            return False
    return True

# Precompute all_4aps for validate
all_4aps = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            all_4aps.append((a, b, c, e))

if __name__ == "__main__":
    result = construct()
    m = 0
    for x in result:
        m |= bit(x)
    real_score = count_3aps(m)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {real_score}")
    print(f"Valid (no 4-AP): {valid}")
