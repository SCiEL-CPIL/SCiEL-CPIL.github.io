import time
import random

N = 256
DEADLINE = 85.0  # 90s limit, keep 5s safety margin

def _precompute():
    """
    Precompute 3-APs and 4-APs with bitmask representations for fast operations.
    Returns:
    aps3: list of (a, b, c)
    aps4: list of (a, b, c, e)
    elem_to_aps3: list of lists, elem_to_aps3[z] contains mask of other 2 elements for each 3-AP containing z
    elem_to_aps4: list of lists, elem_to_aps4[z] contains mask of other 3 elements for each 4-AP containing z
    """
    aps3 = []
    aps4 = []
    
    # Initialize lists for each element (1 to N)
    # We use index 1..N, so list size N+1
    elem_to_aps3 = [[] for _ in range(N + 1)]
    elem_to_aps4 = [[] for _ in range(N + 1)]
    
    # Generate 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            
            # For each element in the AP, store the mask of the other two
            # a: others are b, c
            elem_to_aps3[a].append((1 << b) | (1 << c))
            # b: others are a, c
            elem_to_aps3[b].append((1 << a) | (1 << c))
            # c: others are a, b
            elem_to_aps3[c].append((1 << a) | (1 << b))
            
    # Generate 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            
            # For each element in the AP, store the mask of the other three
            # a: others are b, c, e
            elem_to_aps4[a].append((1 << b) | (1 << c) | (1 << e))
            # b: others are a, c, e
            elem_to_aps4[b].append((1 << a) | (1 << c) | (1 << e))
            # c: others are a, b, e
            elem_to_aps4[c].append((1 << a) | (1 << b) | (1 << e))
            # e: others are a, b, c
            elem_to_aps4[e].append((1 << a) | (1 << b) | (1 << c))
            
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_mask():
    """
    Greedy construction using bitmask checks.
    Iterates 1..N, adds x if it doesn't complete a 4-AP with existing elements.
    """
    # Re-fetch precomputed data or pass it? 
    # To avoid global state issues in concurrent evals, we should pass or re-compute.
    # Precomputation is fast, but passing is cleaner. 
    # However, construct() is the entry point. We can precompute inside construct or use a global cache.
    # Let's use a simple global cache to avoid re-computation if called multiple times (though usually called once).
    global _cache_aps3, _cache_aps4, _cache_elem_to_aps3, _cache_elem_to_aps4
    if not hasattr(_initial_greedy_mask, 'initialized'):
        _cache_aps3, _cache_aps4, _cache_elem_to_aps3, _cache_elem_to_aps4 = _precompute()
        _initial_greedy_mask.initialized = True
        
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _cache_aps3, _cache_aps4, _cache_elem_to_aps3, _cache_elem_to_aps4
    
    A_mask = 0
    for x in range(1, N + 1):
        valid = True
        # Check if adding x creates a 4-AP
        # A 4-AP is formed if there is a 4-AP Q containing x such that Q\{x} is subset of A
        for mask_other3 in elem_to_aps4[x]:
            if (A_mask & mask_other3) == mask_other3:
                valid = False
                break
        if valid:
            A_mask |= (1 << x)
            
    return A_mask

def construct():
    start_time = time.time()
    
    # Ensure precomputation is done
    if not hasattr(construct, 'initialized'):
        _cache_aps3, _cache_aps4, _cache_elem_to_aps3, _cache_elem_to_aps4 = _precompute()
        construct.initialized = True
        
    aps3, aps4, elem_to_aps3, elem_to_aps4 = construct.initialized and (_cache_aps3, _cache_aps4, _cache_elem_to_aps3, _cache_elem_to_aps4) or _precompute()
    
    # Initial greedy set
    A_mask = _initial_greedy_mask()
    
    # Calculate initial score
    def calc_score(mask):
        score = 0
        # Iterate through all 3-APs
        for a, b, c in aps3:
            if (mask & (1 << a)) and (mask & (1 << b)) and (mask & (1 << c)):
                score += 1
        return score

    current_score = calc_score(A_mask)
    best_score = current_score
    best_mask = A_mask
    
    # Helper to check validity of swap (x, y)
    # x is in A, y is not in A
    # Returns True if A \ {x} U {y} is 4-AP free
    def is_valid_swap(x, y, mask):
        # Check 4-APs containing y
        # A 4-AP Q containing y is formed if Q\{y} subset of A \ {x}
        # This means:
        # 1. Q\{y} subset of A
        # 2. x is NOT in Q\{y} (so removing x doesn't break the subset condition)
        for mask_other3 in elem_to_aps4[y]:
            # Check if Q\{y} is subset of A
            if (mask & mask_other3) == mask_other3:
                # Check if x is in Q\{y}
                # If x is in Q\{y}, then x is in mask_other3
                if not (mask_other3 & (1 << x)):
                    return False
        return True

    # Helper to calculate score delta for swap (x, y)
    def calc_delta(x, y, mask):
        loss = 0
        gain = 0
        
        # Loss: 3-APs containing x that are in A
        for mask_other2 in elem_to_aps3[x]:
            if (mask & mask_other2) == mask_other2:
                loss += 1
                
        # Gain: 3-APs containing y that are in A \ {x}
        # A 3-AP P containing y is formed if P\{y} subset of A \ {x}
        # This means:
        # 1. P\{y} subset of A
        # 2. x is NOT in P\{y}
        for mask_other2 in elem_to_aps3[y]:
            if (mask & mask_other2) == mask_other2:
                if not (mask_other2 & (1 << x)):
                    gain += 1
                    
        return gain - loss

    # Local Search
    # We will perform hill climbing. 
    # To escape local optima, we will allow "kicks" (random valid swaps even if score drops) 
    # if we stall for a certain number of passes.
    
    max_passes_no_improve = 5
    passes_no_improve = 0
    kick_counter = 0
    
    while (time.time() - start_time) < DEADLINE - 2:
        improved = False
        # Get lists of elements in A and not in A
        # Optimization: Precompute bit lists? 
        # N=256 is small, list comprehension is fine.
        x_in_A = [i for i in range(1, N + 1) if (A_mask >> i) & 1]
        y_not_in_A = [i for i in range(1, N + 1) if not ((A_mask >> i) & 1)]
        
        random.shuffle(x_in_A)
        random.shuffle(y_not_in_A)
        
        # Try to find an improving swap
        found_improvement = False
        for x in x_in_A:
            if (time.time() - start_time) > DEADLINE - 2:
                break
            for y in y_not_in_A:
                if not is_valid_swap(x, y, A_mask):
                    continue
                
                delta = calc_delta(x, y, A_mask)
                
                if delta > 0:
                    # Accept improvement
                    A_mask &= ~(1 << x)
                    A_mask |= (1 << y)
                    current_score += delta
                    improved = True
                    found_improvement = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask
                    break # Restart x loop with new state
            if found_improvement:
                break
        
        if improved:
            passes_no_improve = 0
        else:
            passes_no_improve += 1
            
            # If stalled, perform a kick
            if passes_no_improve >= max_passes_no_improve:
                # Find a random valid swap to kick
                # We might not find one quickly if the space is tight, but usually there are many valid swaps
                kicked = False
                # Try a few random swaps
                for _ in range(100):
                    if not x_in_A or not y_not_in_A:
                        break
                    x = random.choice(x_in_A)
                    y = random.choice(y_not_in_A)
                    if is_valid_swap(x, y, A_mask):
                        delta = calc_delta(x, y, A_mask)
                        # Accept even if delta <= 0 to escape local optimum
                        A_mask &= ~(1 << x)
                        A_mask |= (1 << y)
                        current_score += delta
                        kicked = True
                        break
                
                if kicked:
                    passes_no_improve = 0
                    # Update best if somehow it improved (unlikely for a kick, but possible if delta > 0)
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask
                else:
                    # No valid kick found, try a different strategy or stop
                    # For now, break if we can't kick
                    break

    return sorted([i for i in range(1, N + 1) if (best_mask >> i) & 1])

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the entry point is exactly 'construct'
# The code above defines it.
# EVOLVE-BLOCK-END
