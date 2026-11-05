"""
4-term progression-free set construction for [1, 512].
Uses bitmask-based Simulated Annealing with efficient 4-AP conflict detection.
"""
import time
import random

N = 512
K = 4  # length of arithmetic progression

# Precompute all 4-APs in [1, 512]
# For each element x, store the list of 4-APs that contain x
# A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1

def precompute_4ap_conflicts():
    """
    For each element x in [1, N], compute the set of other elements that 
    form a 4-AP with x and two other elements.
    Specifically, for each x, conflict[x] is a list of (y, z) pairs such that
    {x, y, z, w} forms a 4-AP for some w, and we need to track which triples
    would complete a 4-AP if x were added.
    
    Actually, let's use a different approach: for each x, precompute which 
    subsets of size 2 from the remaining elements would complete a 4-AP with x.
    
    Better: For each x, precompute a bitmask of "forbidden pairs" - i.e., for each
    pair (y, z), if adding x would create a 4-AP containing x, y, z, and some w,
    then mark that. But this is complex.
    
    Simplest efficient approach: For each x, precompute the list of 4-APs that 
    contain x. Then is_valid_addition(S, x) checks if any 4-AP containing x has 
    all other 3 elements in S.
    """
    # ap_containing[x] = list of (other1, other2, other3) tuples
    # where {x, other1, other2, other3} forms a 4-AP
    ap_containing = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # The 4-AP is (a, b, c, e)
            # For a: the other three are b, c, e
            ap_containing[a].add((b, c, e))
            # For b: the other three are a, c, e
            ap_containing[b].add((a, c, e))
            # For c: the other three are a, b, e
            ap_containing[c].add((a, b, e))
            # For e: the other three are a, b, c
            ap_containing[e].add((a, b, c))
    
    return ap_containing

# Precompute once
AP_CONTAINING = precompute_4ap_conflicts()

def is_valid_addition_bitmask(mask, x):
    """
    Check if adding x (1-indexed) to the set represented by bitmask keeps it 4-AP-free.
    mask is an integer where bit i is set if element i is in the set.
    """
    # Check all 4-APs containing x
    for (y, z, w) in AP_CONTAINING[x]:
        # Check if y, z, w are all in the current set
        if (mask >> (y - 1)) & 1 and (mask >> (z - 1)) & 1 and (mask >> (w - 1)) & 1:
            return False
    return True

def greedy_construct_bitmask(order=None):
    """Greedy: add elements in given order if valid, using bitmask."""
    if order is None:
        order = list(range(1, N + 1))
    mask = 0
    for x in order:
        if is_valid_addition_bitmask(mask, x):
            mask |= (1 << (x - 1))
    return mask

def mask_to_list(mask):
    """Convert bitmask to sorted list of elements."""
    result = []
    for i in range(N):
        if (mask >> i) & 1:
            result.append(i + 1)
    return result

def simulated_annealing(initial_mask, time_limit=50.0, rng=None):
    """
    Simulated Annealing to maximize the size of a 4-AP-free set.
    
    Moves:
    1. Add a random element not in the set (if valid)
    2. Remove a random element from the set
    3. Swap: remove one, add another
    
    Accept worse solutions with probability exp(-delta / T).
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    mask = initial_mask
    best_mask = mask
    best_size = bin(mask).count('1')
    
    # Temperature schedule
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Generate a neighbor
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [i + 1 for i in range(N) if not ((mask >> i) & 1)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if is_valid_addition_bitmask(mask, x):
                new_mask = mask | (1 << (x - 1))
                new_size = bin(new_mask).count('1')
                delta = new_size - best_size  # We want to maximize
                # Always accept if better
                if new_size >= bin(mask).count('1'):
                    mask = new_mask
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                else:
                    # Shouldn't happen since we only add valid elements
                    pass
            else:
                continue
                
        elif move_type < 0.7:
            # Try to remove a random element (to escape local optima)
            elements = [i + 1 for i in range(N) if (mask >> i) & 1]
            if not elements:
                continue
            x = rng.choice(elements)
            new_mask = mask & ~(1 << (x - 1))
            new_size = bin(new_mask).count('1')
            old_size = bin(mask).count('1')
            delta = new_size - old_size  # Negative
            
            # Accept with probability exp(delta / T)
            if rng.random() < (2.718281828 ** (delta / T)):
                mask = new_mask
        else:
            # Swap: remove one, try to add another
            elements = [i + 1 for i in range(N) if (mask >> i) & 1]
            if not elements:
                continue
            x_remove = rng.choice(elements)
            candidates = [i + 1 for i in range(N) if not ((mask >> i) & 1)]
            if not candidates:
                continue
            x_add = rng.choice(candidates)
            
            # Temporarily remove x_remove
            temp_mask = mask & ~(1 << (x_remove - 1))
            if is_valid_addition_bitmask(temp_mask, x_add):
                new_mask = temp_mask | (1 << (x_add - 1))
                new_size = bin(new_mask).count('1')
                old_size = bin(mask).count('1')
                delta = new_size - old_size
                
                if delta >= 0:
                    mask = new_mask
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                else:
                    if rng.random() < (2.718281828 ** (delta / T)):
                        mask = new_mask
        
        # Cool down
        T *= cooling_rate
    
    # After SA, do a final greedy add-back pass
    for x in range(1, N + 1):
        if not ((best_mask >> (x - 1)) & 1) and is_valid_addition_bitmask(best_mask, x):
            best_mask |= (1 << (x - 1))
    
    return best_mask

def construct():
    """Construct a large 4-AP-free subset of [1, 512]."""
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Start with greedy in natural order
    best_mask = greedy_construct_bitmask()
    best_size = bin(best_mask).count('1')
    
    # Try greedy in reverse order
    mask_rev = greedy_construct_bitmask(list(range(N, 0, -1)))
    if bin(mask_rev).count('1') > best_size:
        best_mask = mask_rev
        best_size = bin(best_mask).count('1')
    
    # Try several random orderings with greedy
    rng = random.Random(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        mask = greedy_construct_bitmask(order)
        if bin(mask).count('1') > best_size:
            best_mask = mask
            best_size = bin(best_mask).count('1')
    
    # Run Simulated Annealing on the best found so far
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_mask = simulated_annealing(best_mask, time_limit=remaining_time - 3.0, rng=rng)
        if bin(sa_mask).count('1') > best_size:
            best_mask = sa_mask
            best_size = bin(best_mask).count('1')
    
    # If we have time, run SA on a few other starting points
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for trial in range(5):
            if time.time() > deadline:
                break
            order = list(range(1, N + 1))
            rng.shuffle(order)
            mask = greedy_construct_bitmask(order)
            sa_mask = simulated_annealing(mask, time_limit=min(8.0, deadline - time.time() - 2.0), rng=rng)
            if bin(sa_mask).count('1') > best_size:
                best_mask = sa_mask
                best_size = bin(best_mask).count('1')
    
    return mask_to_list(best_mask)
