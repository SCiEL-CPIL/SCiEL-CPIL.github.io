import time
import random
import math

def construct():
    N = 512
    # Bit i represents element i+1
    
    # Precompute conflict masks
    # conflicts[x_idx] is a list of masks. 
    # A mask M in conflicts[x_idx] represents a set of 3 elements {a, b, c} 
    # such that {x, a, b, c} forms a 4-AP.
    # If (S & M) == M, then adding x to S creates a 4-AP.
    
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # 1 <= a, a+3d <= N
    # d >= 1
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-indexed bits
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            # For each element in the AP, the other 3 form a conflict mask
            # Element a (index a-1)
            mask_bce = b_bit | c_bit | e_bit
            conflicts[a - 1].append(mask_bce)
            
            # Element b (index b-1)
            mask_ace = a_bit | c_bit | e_bit
            conflicts[b - 1].append(mask_ace)
            
            # Element c (index c-1)
            mask_abe = a_bit | b_bit | e_bit
            conflicts[c - 1].append(mask_abe)
            
            # Element e (index e-1)
            mask_abc = a_bit | b_bit | c_bit
            conflicts[e - 1].append(mask_abc)

    def can_add(x_idx, S):
        """Check if element x_idx can be added to set S without creating a 4-AP."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def greedy_fill(S, order):
        """Add elements from order to S if they don't create a 4-AP."""
        for x_idx in order:
            if not (S & (1 << x_idx)):
                if can_add(x_idx, S):
                    S |= (1 << x_idx)
        return S

    def count_bits(S):
        return bin(S).count('1')

    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer for 60s limit

    best_S = 0
    best_size = 0
    current_S = 0
    current_size = 0

    # 1. Initial Greedy Constructions
    # Try a few random orderings to get a good starting point
    base_order = list(range(N))
    
    # Always try natural order first (often good for AP-free sets? Maybe not, but cheap)
    S_try = greedy_fill(0, base_order)
    size_try = count_bits(S_try)
    if size_try > best_size:
        best_size = size_try
        best_S = S_try
        current_S = S_try
        current_size = size_try

    # Try random orderings
    for _ in range(20):
        random.shuffle(base_order)
        S_try = greedy_fill(0, base_order)
        size_try = count_bits(S_try)
        if size_try > best_size:
            best_size = size_try
            best_S = S_try
            current_S = S_try
            current_size = size_try

    # 2. Local Search with Perturbation and Greedy Fill
    # Strategy: Remove a small number of random elements, then run greedy fill to recover.
    # This allows escaping local optima where no single addition is possible, but a 
    # rearrangement of a few elements allows more total elements.
    
    # We will use a "kick" strategy.
    # Occasionally, we also try simple swaps.
    
    # To speed up greedy_fill, we can maintain a list of "available" elements?
    # No, N=512 is small. Iterating 512 times is very fast.
    
    # We need to be careful with the time.
    # Each greedy_fill is O(N * K) where K is avg conflicts per element.
    # K is roughly (N/4) * 3 * 2? No, number of APs containing x is approx N/4.
    # So one fill is ~512 * 100 = 50k ops. Very fast in Python?
    # Bitwise ops are fast. The loop is the bottleneck.
    # 50k iterations in Python takes ~2-5ms.
    # So we can do ~1000 fills in 5 seconds.
    # We have 55 seconds. So ~10,000 fills.
    
    # Let's structure the loop:
    # While time < deadline:
    #    1. Perturb: Remove k random elements from current_S
    #    2. Fill: Run greedy_fill on the perturbed set
    #    3. If size > best_size, update best
    #    4. Set current_S to the filled set (or keep the perturbed one if we want to explore more? 
    #       Usually, moving to the local optimum (filled set) is better for stability, 
    #       but we might get stuck. 
    #       A mix: With prob 0.5, keep the filled set. With prob 0.5, revert to previous best 
    #       and try a different perturbation? 
    #       Actually, standard practice is to accept the new state if it's valid (it is) 
    #       and track the best. We should probably move to the new local optimum 
    #       to avoid repeating the same perturbations on the same basin.)
        
    # To diversify, we can also try "Swaps":
    # Pick u in S, v not in S.
    # If can_add(v, S \ {u}), then S = S \ {u} \ {v} + v? No, S' = S - u + v.
    # This is a 1-swap.
    
    # Let's combine:
    # Most of the time, do "Remove k and Fill".
    # Sometimes, do "Swap".
    
    # Optimization: Precompute list of indices for current set?
    # We can just iterate 0..N-1 to find elements in S.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Choose move type
        # 80% Remove-and-Fill, 20% Swap
        if random.random() < 0.8:
            # Remove k elements
            # k depends on stagnation? Let's keep it simple: random 1-3
            k = random.randint(1, 3)
            
            # Get list of elements in current_S
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            
            if len(elements_in_set) < k:
                k = len(elements_in_set)
                if k == 0:
                    continue
            
            if k > 0:
                removed = random.sample(elements_in_set, k)
                S_temp = current_S
                for idx in removed:
                    S_temp &= ~(1 << idx)
                
                # Greedy fill
                # Use a random order to break symmetry? 
                # Or natural order? 
                # Random order might find different solutions.
                # Let's use a random shuffle of the full range.
                # To save time, we can just use the global base_order shuffled once per iteration?
                # Shuffling 512 items is cheap.
                order = list(range(N))
                random.shuffle(order)
                
                S_filled = greedy_fill(S_temp, order)
                size_filled = count_bits(S_filled)
                
                if size_filled > best_size:
                    best_size = size_filled
                    best_S = S_filled
                
                # Move to the new local optimum
                current_S = S_filled
                current_size = size_filled
            else:
                # If set was empty, just fill from scratch
                order = list(range(N))
                random.shuffle(order)
                S_filled = greedy_fill(0, order)
                size_filled = count_bits(S_filled)
                if size_filled > best_size:
                    best_size = size_filled
                    best_S = S_filled
                current_S = S_filled
                current_size = size_filled
                
        else:
            # Swap move
            # Pick u in S
            elements_in_set = [i for i in range(N) if (current_S & (1 << i))]
            if not elements_in_set:
                continue
            u = random.choice(elements_in_set)
            
            # Pick v not in S
            # To make it faster, we can just pick a random index and check
            # Or maintain a list of not-in-set?
            # Let's just pick random v from 0..N-1 and check if it's in S.
            # If it is, pick another.
            # Since S is dense, this might take a few tries.
            # Or we can construct the list of not-in-set if size is small?
            # Size is ~100-200. N=512.
            # List comprehension for not-in-set is O(N).
            
            # Let's try a few random v's
            found_swap = False
            for _ in range(10): # Try up to 10 random v's
                v = random.randint(0, N - 1)
                if current_S & (1 << v):
                    continue # v is in S, try another
                
                # Check if v can be added to S \ {u}
                S_temp = current_S & ~(1 << u)
                if can_add(v, S_temp):
                    S_new = S_temp | (1 << v)
                    # Size remains same, but structure changes.
                    # We don't track "best" based on size alone for swaps if size doesn't increase.
                    # But it might lead to a better basin for future removals.
                    # So we accept the swap if it's valid.
                    current_S = S_new
                    # current_size is unchanged
                    found_swap = True
                    break
            
            # If no swap found, we might want to do a small removal to unblock?
            # The 80% branch handles that.

    # Return the best set found
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
