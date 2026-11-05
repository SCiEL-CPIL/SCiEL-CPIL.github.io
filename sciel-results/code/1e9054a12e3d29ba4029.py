import time
import random
import sys

N = 256

def _precompute():
    """
    Precompute data structures for fast validity and score updates.
    """
    # 1. Find all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We need to know, for each element x, which other 3 elements form a 4-AP with x.
    # Let's store for each x, a list of bitmasks of the other 3 elements.
    
    four_ap_triples = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a from 1 to N
    # d from 1 to (N-a)//3
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                # The 4-AP is (a, b, c, e)
                # For a: others are b, c, e
                mask_a = (1 << b) | (1 << c) | (1 << e)
                four_ap_triples[a].append(mask_a)
                
                # For b: others are a, c, e
                mask_b = (1 << a) | (1 << c) | (1 << e)
                four_ap_triples[b].append(mask_b)
                
                # For c: others are a, b, e
                mask_c = (1 << a) | (1 << b) | (1 << e)
                four_ap_triples[c].append(mask_c)
                
                # For e: others are a, b, c
                mask_e = (1 << a) | (1 << b) | (1 << c)
                four_ap_triples[e].append(mask_e)

    # 2. Find all 3-APs
    # A 3-AP is (a, a+d, a+2d). Let the elements be x, y, z with x+z=2y.
    # For each element x, which pairs (u, v) form a 3-AP with x?
    # If x is the first term: x, x+d, x+2d. Others are x+d, x+2d.
    # If x is the middle term: x-d, x, x+d. Others are x-d, x+d.
    # If x is the last term: x-2d, x-d, x. Others are x-2d, x-d.
    
    three_ap_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = b + d
            if c <= N:
                # 3-AP is (a, b, c)
                
                # For a: others are b, c
                mask_a = (1 << b) | (1 << c)
                three_ap_pairs[a].append(mask_a)
                
                # For b: others are a, c
                mask_b = (1 << a) | (1 << c)
                three_ap_pairs[b].append(mask_b)
                
                # For c: others are a, b
                mask_c = (1 << a) | (1 << b)
                three_ap_pairs[c].append(mask_c)

    return four_ap_triples, three_ap_pairs

def _is_valid_add(mask, x, four_ap_triples):
    """
    Check if adding x to the set represented by mask creates a 4-AP.
    """
    for triple_mask in four_ap_triples[x]:
        if (mask & triple_mask) == triple_mask:
            return False
    return True

def _delta_add(mask, x, three_ap_pairs):
    """
    Calculate the increase in the number of 3-APs if x is added to the set.
    """
    count = 0
    for pair_mask in three_ap_pairs[x]:
        if (mask & pair_mask) == pair_mask:
            count += 1
    return count

def _delta_remove(mask, x, three_ap_pairs):
    """
    Calculate the decrease in the number of 3-APs if x is removed from the set.
    Note: This is the number of 3-APs that *contained* x.
    """
    count = 0
    for pair_mask in three_ap_pairs[x]:
        if (mask & pair_mask) == pair_mask:
            count += 1
    return count

def _count_3aps(mask, three_ap_pairs):
    """
    Calculate the total number of 3-APs in the set.
    """
    count = 0
    # Iterate through all set bits
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        x = lsb.bit_length() - 1
        count += _delta_add(0, x, three_ap_pairs) # This is wrong, _delta_add expects the *rest* of the set
        # Actually, simpler to just sum contributions.
        # But _delta_add(mask, x) counts pairs in mask.
        # If we iterate x, we should count pairs in (mask & ~(1<<x)).
        # Let's just use the direct method for initialization.
        m &= m - 1
        
    # Re-implementing simple count for correctness
    count = 0
    m = mask
    while m:
        lsb = m & -m
        x = lsb.bit_length() - 1
        # Count 3-APs where x is the first element? No, just count all unique 3-APs.
        # A 3-AP is uniquely identified by its middle element and difference, or just the set of 3.
        # Let's sum over all x in set, count pairs in set that form 3-AP with x, then divide by 3?
        # No, each 3-AP {a,b,c} is counted once for a, once for b, once for c.
        # So total 3-APs = (Sum over x in A of #pairs in A forming 3-AP with x) / 3.
        
        # Let's just use the precomputed pairs.
        for pair_mask in three_ap_pairs[x]:
            if (mask & pair_mask) == pair_mask:
                count += 1
        m &= m - 1
        
    return count // 3

def _greedy_init(four_ap_triples, three_ap_pairs):
    """
    Construct a valid set using a greedy strategy.
    """
    mask = 0
    # Try adding elements in increasing order
    for x in range(1, N + 1):
        if _is_valid_add(mask, x, four_ap_triples):
            mask |= (1 << x)
    return mask

def _simulated_annealing(four_ap_triples, three_ap_pairs, deadline, seed=42):
    random.seed(seed)
    
    # Initialize with greedy
    mask = _greedy_init(four_ap_triples, three_ap_pairs)
    current_score = _count_3aps(mask, three_ap_pairs)
    
    best_mask = mask
    best_score = current_score
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    
    # List of elements for random selection
    all_elements = list(range(1, N + 1))
    
    while T > T_min and time.time() < deadline:
        # Get current set and unset elements
        # Efficient way to pick random x in mask and y not in mask
        # Since N=256 is small, we can just pick random integers and check
        
        # Choose move type
        r = random.random()
        
        if r < 0.3:
            # Add an element
            x = random.randint(1, N)
            if mask & (1 << x):
                continue # Already in set
            
            if _is_valid_add(mask, x, four_ap_triples):
                new_mask = mask | (1 << x)
                delta = _delta_add(mask, x, three_ap_pairs)
                new_score = current_score + delta
                
                # Accept
                if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
        elif r < 0.6:
            # Remove an element
            x = random.randint(1, N)
            if not (mask & (1 << x)):
                continue # Not in set
            
            new_mask = mask & ~(1 << x)
            delta = -_delta_remove(mask, x, three_ap_pairs)
            new_score = current_score + delta
            
            # Accept
            if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
        else:
            # Swap an element
            x = random.randint(1, N)
            y = random.randint(1, N)
            
            if x == y:
                continue
            
            # x must be in mask, y must not be in mask
            if (mask & (1 << x)) and not (mask & (1 << y)):
                # Check validity of adding y to mask without x
                temp_mask = mask & ~(1 << x)
                if _is_valid_add(temp_mask, y, four_ap_triples):
                    new_mask = temp_mask | (1 << y)
                    
                    # Score change:
                    # Remove x: -_delta_remove(mask, x)
                    # Add y to temp_mask: +_delta_add(temp_mask, y)
                    delta = -_delta_remove(mask, x, three_ap_pairs) + _delta_add(temp_mask, y, three_ap_pairs)
                    new_score = current_score + delta
                    
                    # Accept
                    if new_score >= current_score or random.random() < pow(2.71828, (new_score - current_score) / T):
                        mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                            
        T *= cooling_rate
        
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute
    four_ap_triples, three_ap_pairs = _precompute()
    
    # Run Simulated Annealing
    best_mask, best_score = _simulated_annealing(four_ap_triples, three_ap_pairs, deadline)
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    # Quick validation
    # Check no 4-AP
    s = set(res)
    valid = True
    for a in s:
        for d in range(1, (256-a)//3 + 1):
            if a+d in s and a+2*d in s and a+3*d in s:
                valid = False
                break
        if not valid:
            break
    if not valid:
        print("ERROR: Invalid set (contains 4-AP)")
    else:
        print("Valid set")
    
    # Count 3-APs
    count = 0
    for a in s:
        for d in range(1, (256-a)//2 + 1):
            if a+d in s and a+2*d in s:
                count += 1
    print(f"Number of 3-APs: {count}")
