import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256

    # Precompute 3-AP and 4-AP structures using bitmasks for speed
    # A 3-AP is (a, b, c) where a < b < c and b-a = c-b, i.e., a+c = 2b
    # A 4-AP is (a, b, c, d) where a < b < c < d and b-a = c-b = d-c

    # For each element x, precompute which pairs of other elements form a 3-AP with x
    three_ap_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if z <= N:
                three_ap_pairs[x].append((y, z))
        # x as second: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))
        # x as third: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))

    # For each element x, precompute which triples of other elements form a 4-AP with x
    four_ap_triples = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as 1st
        for d in range(1, (N - x) // 3 + 1):
            y = x + d
            z = x + 2 * d
            w = x + 3 * d
            if w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            y = x - d
            z = x + d
            w = x + 2 * d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            y = x - 2 * d
            z = x - d
            w = x + d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 4th
        for d in range(1, (x - 1) // 3 + 1):
            y = x - 3 * d
            z = x - 2 * d
            w = x - d
            if y >= 1:
                four_ap_triples[x].append((y, z, w))

    def count_3aps_bitmask(mask):
        count = 0
        for x in range(1, N + 1):
            if not (mask >> x) & 1:
                continue
            for (y, z) in three_ap_pairs[x]:
                if (mask >> y) & 1 and (mask >> z) & 1:
                    count += 1
        return count

    def has_4ap_bitmask(mask):
        for x in range(1, N + 1):
            if not (mask >> x) & 1:
                continue
            for (y, z, w) in four_ap_triples[x]:
                if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                    return True
        return False

    def creates_4ap_bitmask(x, mask):
        for (y, z, w) in four_ap_triples[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return True
        return False

    def gain_3ap_bitmask(x, mask):
        gain = 0
        for (y, z) in three_ap_pairs[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                gain += 1
        return gain

    def loss_3ap_bitmask(x, mask):
        """Calculate loss of 3-APs if x is removed from mask."""
        loss = 0
        for (y, z) in three_ap_pairs[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                loss += 1
        return loss

    def simulated_annealing(initial_mask, max_time=80.0):
        current_mask = initial_mask
        current_score = count_3aps_bitmask(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        # Temperature schedule
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        # Build a list of elements currently in the set for faster removal/swap
        in_set = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
        out_set = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
        
        while time.time() < deadline - 5.0 and T > min_T:
            move_type = random.random()
            new_mask = current_mask
            delta = 0
            
            if move_type < 0.4 and out_set:
                # Add
                x = random.choice(out_set)
                if not creates_4ap_bitmask(x, current_mask):
                    new_mask = current_mask | (1 << x)
                    delta = gain_3ap_bitmask(x, current_mask)
                    # Update sets
                    in_set.append(x)
                    out_set.remove(x)
                else:
                    continue
            elif move_type < 0.7 and in_set:
                # Remove
                x = random.choice(in_set)
                new_mask = current_mask & ~(1 << x)
                delta = -loss_3ap_bitmask(x, current_mask)
                # Update sets
                in_set.remove(x)
                out_set.append(x)
            elif move_type < 1.0 and in_set and out_set:
                # Swap
                x = random.choice(in_set)
                y = random.choice(out_set)
                # Check if removing x and adding y is valid
                mask_temp = (current_mask & ~(1 << x)) | (1 << y)
                # We need to check if y creates a 4-AP in the context of (current_mask - x)
                if not creates_4ap_bitmask(y, current_mask & ~(1 << x)):
                    # Calculate delta
                    # Loss of x, Gain of y
                    # Note: gain/loss functions depend on the *other* elements in the set.
                    # gain_3ap(y, mask - x) and loss_3ap(x, mask)
                    # But removing x might break some 3-APs that y completes? 
                    # Actually, delta = score(new) - score(old)
                    # score(new) = score(old) - loss(x) + gain(y, mask-x)
                    # However, gain(y, mask-x) counts 3-APs formed by y and pairs in (mask-x).
                    # loss(x, mask) counts 3-APs formed by x and pairs in mask.
                    # Are there interactions? 
                    # A 3-AP is a set of 3. 
                    # If a 3-AP contains x and y, it is counted in loss(x) (if the third is in mask) 
                    # and in gain(y) (if the third is in mask-x). 
                    # So we must be careful.
                    # Let's just compute the score of the new mask directly if the sets are small? 
                    # No, count_3aps is O(N^2) approx. 
                    # Let's use the incremental approach carefully.
                    # Delta = (Gain of y in mask-x) - (Loss of x in mask)
                    # This is correct because the 3-APs are disjoint in terms of "new" element? 
                    # No. 
                    # Let's just call count_3aps_bitmask for swaps to be safe, as swaps are less frequent.
                    new_score = count_3aps_bitmask(new_mask)
                    delta = new_score - current_score
                    # Update sets
                    in_set.remove(x)
                    in_set.append(y)
                    out_set.remove(y)
                    out_set.append(x)
                else:
                    continue
            else:
                continue

            if delta > 0 or random.random() < math.exp(-delta / T):
                current_mask = new_mask
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            
            T *= cooling
        
        return best_mask, best_score

    # Initial greedy to get a good starting point
    def greedy_init():
        mask = 0
        remaining = set(range(1, N + 1))
        random.shuffle(list(remaining)) # Shuffle for diversity
        # Actually, let's do a standard greedy
        while remaining and time.time() < deadline - 10:
            best_x = None
            best_gain = -1
            for x in remaining:
                if creates_4ap_bitmask(x, mask):
                    continue
                g = gain_3ap_bitmask(x, mask)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            mask |= (1 << best_x)
            remaining.remove(best_x)
        return mask

    best_mask = 0
    best_score = 0
    
    # Run multiple SA instances with different random seeds
    for i in range(10):
        if time.time() > deadline - 15:
            break
        
        random.seed(123 + i)
        init_mask = greedy_init()
        if init_mask == 0:
            init_mask = 1 << 1 # Fallback
        
        sa_mask, sa_score = simulated_annealing(init_mask)
        
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask

    # Final Polish: Hill Climbing on the best result
    # Try single swaps/adds/removes to improve
    current_mask = best_mask
    current_score = best_score
    in_set = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
    out_set = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try adding
        for x in out_set[:]:
            if creates_4ap_bitmask(x, current_mask):
                continue
            g = gain_3ap_bitmask(x, current_mask)
            if g > 0:
                new_mask = current_mask | (1 << x)
                new_score = current_score + g
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    in_set.append(x)
                    out_set.remove(x)
                    improved = True
                    break
        if improved:
            continue
        # Try removing (only if it allows better swaps later? No, just removal is rarely beneficial unless it enables a swap)
        # Try swapping
        for x in in_set[:]:
            for y in out_set[:]:
                if creates_4ap_bitmask(y, current_mask & ~(1 << x)):
                    continue
                new_mask = (current_mask & ~(1 << x)) | (1 << y)
                new_score = count_3aps_bitmask(new_mask)
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    in_set.remove(x)
                    in_set.append(y)
                    out_set.remove(y)
                    out_set.append(x)
                    improved = True
                    break
            if improved:
                break

    if current_score > best_score:
        best_mask = current_mask
        best_score = current_score

    # Verify
    if has_4ap_bitmask(best_mask):
        # Fallback: remove elements causing 4-APs
        for x in range(1, N + 1):
            if (best_mask >> x) & 1:
                if has_4ap_bitmask(best_mask):
                    best_mask &= ~(1 << x)

    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    if not result:
        result = [1]

    return result
