"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
# EVOLVE-BLOCK-START
import random
import time

N = 512

def _precompute_triplets():
    """
    Precompute for each x in [1, N] a list of bitmasks.
    Each bitmask represents a set of 3 other elements {a, b, c} such that 
    {a, b, c, x} forms a 4-term arithmetic progression.
    """
    triplets = [[] for _ in range(N + 1)]
    
    # Generate all 4-term APs in [1, N]
    # AP: p, p+d, p+2d, p+3d
    for p in range(1, N + 1):
        for d in range(1, (N - p) // 3 + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            if t4 > N:
                continue
            
            # For each element in this AP, the other 3 form a "forbidden triplet"
            # If we add t1, we must not have {t2, t3, t4} all in S.
            # Mask for {t2, t3, t4}
            mask = (1 << t2) | (1 << t3) | (1 << t4)
            triplets[t1].append(mask)
            
            mask = (1 << t1) | (1 << t3) | (1 << t4)
            triplets[t2].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t4)
            triplets[t3].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t3)
            triplets[t4].append(mask)
            
    return triplets

def construct():
    random.seed(42)
    triplets = _precompute_triplets()
    
    # S is a bitmask of the current set
    S = 0
    # List of elements in S for fast random access
    S_list = []
    
    def can_add(x, S_mask):
        """Check if x can be added to S_mask without forming a 4-AP."""
        for mask in triplets[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    # 1. Greedy Initialization
    # We try to add elements in a random order.
    # To improve density, we might try multiple random orders or a specific heuristic.
    # A common heuristic is to pick elements that "block" fewer future additions, 
    # but simple random greedy is often decent for SA starting point.
    # Let's do a few random greedy passes and keep the best.
    
    best_S = 0
    best_size = 0
    best_list = []
    
    for _ in range(10):
        current_S = 0
        current_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]
            
    S = best_S
    S_list = best_list
    best_S = S
    best_size = best_size
    best_list = S_list[:]

    # 2. Simulated Annealing
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    iterations = 150000  # Adjust based on time limit
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    
    current_T = T_start
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, N + 1))
    
    for i in range(iterations):
        if current_T < T_end:
            break
            
        # Generate a move
        # Move Type 1: Add a random element (if possible)
        # Move Type 2: Remove a random element, then try to add a different one (Swap)
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random element
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    # Accept improvement
                    S |= (1 << x)
                    S_list.append(x)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = S
                        best_list = S_list[:]
                # If invalid, we don't accept worsening moves for simple adds 
                # because it would make the set invalid. 
                # Wait, SA usually operates on valid states. 
                # If we only keep valid states, we can't "accept worsening moves" 
                # in the sense of having an invalid set. 
                # We must accept moves that keep the set valid but might be "worse" 
                # in some metric? But the objective is to maximize size. 
                # Any valid set of size K is "worse" than size K+1. 
                # So we should only accept if size increases? 
                # No, that's just greedy. 
                # To escape local optima, we need to allow temporary size decreases?
                # But if we remove an element, size decreases. 
                # Then we try to add two elements? Or add one element that was previously blocked?
                
                # Standard approach for Max Independent Set:
                # State must be valid (independent set).
                # Moves:
                # 1. Add vertex v (if valid). Size +1. Always accept.
                # 2. Remove vertex u, Add vertex v (if valid). Size 0 change. 
                #    Accept with prob exp(-delta/T) where delta is 0? 
                #    This is just random walk on valid states of same size.
                # 3. Remove vertex u, Add vertex v, Add vertex w? Complex.
                
                # Better Move:
                # Pick a random valid state neighbor.
                # Neighbor definition: 
                # A. Add x (if valid) -> Size +1
                # B. Remove x -> Size -1 (Always valid)
                # C. Swap x, y: Remove x, Add y (if valid) -> Size 0
                # D. 2-Swap: Remove x, y; Add z, w (if valid) -> Size 0
                
                # If we only accept Size +1, we get stuck.
                # We must allow Size -1 or Size 0 moves to rearrange.
                
                # Let's implement:
                # 1. Try Add. If valid, accept (Size +1).
                # 2. Try Remove. Always valid (Size -1). Accept with prob 1? 
                #    No, we want to maximize size. Removing is bad. 
                #    But it might allow adding 2 others later?
                #    Usually, we do:
                #    - If we can Add, we do it.
                #    - If we cannot Add, we try a Swap (Remove one, Add one).
                #    - If Swap fails, we might try Remove one (to unblock) and then Add two?
                
                # Let's stick to:
                # - Add (if valid)
                # - Swap (Remove x, Add y, if valid)
                
                # We need to track size.
                pass # Handled below in unified logic

        elif move_type < 0.8:
            # Swap: Remove x from S, Add y to S
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    # Try swap
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        # New state is valid
                        # Size remains same
                        # Acceptance: Since size is same, we accept with prob 1 
                        # to allow random walk? Or use temperature?
                        # If we accept all size-preserving moves, we explore the 
                        # manifold of size-K sets.
                        # This is good for escaping local optima where no single 
                        # add is possible.
                        S = S_new | (1 << y)
                        # Update list
                        # Remove x, add y
                        # Efficient list update:
                        S_list[S_list.index(x)] = y
                        # Check if this leads to a better state later? 
                        # No immediate size gain, but might enable future adds.
                        
        else:
            # Remove x (Size -1)
            # We only do this if we are stuck? 
            # Or use it to "reset" part of the set.
            # Let's try: Remove x, then try to add TWO elements?
            # For now, let's just do Remove to allow recovery from bad swaps?
            # Actually, if we accept all Swaps, we might wander.
            # Let's add a "Remove" move that is accepted with low probability 
            # or only if it allows an immediate Add of 2?
            
            # Simpler: If we are stuck (no adds, no beneficial swaps found recently), 
            # we might need to shrink.
            # But for now, let's rely on Swaps to explore.
            
            # Let's add a "2-Add" attempt: 
            # Pick x, y not in S. 
            # If can_add(x) and can_add(y) and can_add(x, S|y) etc.
            # This is expensive.

            # Let's just do a Remove move to allow "backtracking".
            if S_list:
                x = random.choice(S_list)
                # Accept with probability exp(-1/T) ? 
                # Delta in energy = -1 (size decreases by 1).
                # Energy E = -Size. Delta E = 1.
                # Prob = exp(-Delta E / T) = exp(-1/T)
                if random.random() < (2.71828 ** (-1.0 / current_T)):
                    S &= ~(1 << x)
                    S_list.remove(x)
                    # Note: Size decreased. We don't update best.

        current_T *= cooling_rate

    return sorted(best_list)

# EVOLVE-BLOCK-END
