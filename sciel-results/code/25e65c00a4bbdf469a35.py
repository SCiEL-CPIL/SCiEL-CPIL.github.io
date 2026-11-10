import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # 1. Precompute all 4-APs and conflict masks
    # masks[x] is a list of integers (bitsets) representing the other 3 elements
    # in each 4-AP containing x.
    masks = [[] for _ in range(N + 1)]
    
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            
            # Create bitmask for the AP
            m = (1 << x1) | (1 << x2) | (1 << x3) | (1 << x4)
            
            # Add mask for each element
            masks[x1].append(m ^ (1 << x1))
            masks[x2].append(m ^ (1 << x2))
            masks[x3].append(m ^ (1 << x3))
            masks[x4].append(m ^ (1 << x4))

    def can_add(x, S):
        """Check if x can be added to set S (represented as int bitset)"""
        for m in masks[x]:
            if (m & S).bit_count() >= 3:
                return False
        return True

    def greedy_add(S, candidates):
        """Greedily add elements from candidates to S"""
        added = False
        for v in candidates:
            if not (S >> v) & 1: # if v not in S
                if can_add(v, S):
                    S |= (1 << v)
                    added = True
        return S, added

    # 2. Initial Heuristics to find a good starting set
    best_set = 0
    best_size = 0
    
    # Heuristic 1: Base-4 (Salem-Spencer)
    # Numbers with base-4 digits only 0,1
    S_base4 = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            S_base4 |= (1 << i)
    
    if S_base4.bit_count() > best_size:
        best_set = S_base4
        best_size = S_base4.bit_count()

    # Heuristic 2: Random Greedy
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = 0
        for x in order:
            if can_add(x, S):
                S |= (1 << x)
        
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # 3. Local Search (Iterated Local Search)
    # We will perform random perturbations and repairs.
    
    while time.time() < deadline - 2:
        # Perturb current best
        S = best_set
        s_size = S.bit_count()
        
        # Remove k random elements
        k = random.randint(2, 5)
        elements = [i for i in range(1, N + 1) if (S >> i) & 1]
        if len(elements) > k:
            to_remove = random.sample(elements, k)
            for u in to_remove:
                S &= ~(1 << u)
        
        # Repair: Greedily add elements
        # Prioritize adding elements that were removed? Or just random order.
        # Let's try random order first, then maybe a second pass with sorted order?
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S, _ = greedy_add(S, candidates)
        
        # Local Optimize: 1-Swap
        # Try to find a swap (u in S, v not in S) that allows adding more elements
        # or just improves the set structure.
        # Since we maximize size, a swap that keeps size same is only useful
        # if it enables a subsequent addition.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try a limited number of random swaps
            for _ in range(200):
                if time.time() > deadline - 2:
                    break
                    
                s_elements = [i for i in range(1, N + 1) if (S >> i) & 1]
                c_elements = [i for i in range(1, N + 1) if not ((S >> i) & 1)]
                
                if not s_elements or not c_elements:
                    break
                    
                u = random.choice(s_elements)
                v = random.choice(c_elements)
                
                # Check if swapping u for v is valid
                S_temp = S & ~(1 << u)
                if can_add(v, S_temp):
                    S_new = S_temp | (1 << v)
                    
                    # Try to add more elements after the swap
                    S_final, added = greedy_add(S_new, c_elements)
                    
                    if S_final.bit_count() > S.bit_count():
                        S = S_final
                        improved = True
                        # If we improved, we can break out of swap loop and restart
                        # local optimization or just continue
                        break
                    elif S_final.bit_count() == S.bit_count():
                        # Accept swap even if size doesn't increase immediately?
                        # This allows escaping local optima.
                        # But we must ensure we don't cycle.
                        # Let's accept it with some probability or always accept
                        # and rely on the repair phase to find additions.
                        # However, if we accept a non-improving swap, we must
                        # be careful not to degrade the set.
                        # Let's only accept if it leads to an addition in the
                        # greedy_add phase.
                        # If no addition, reject.
                        pass
                        # Actually, let's try:
                        # If S_final == S_new (no addition), reject.
                        # If S_final > S_new, accept.
                        # We already checked S_final > S.bit_count() for improvement.
                        # If S_final == S.bit_count(), we can accept it to change structure.
                        # Let's accept it 50% of the time to allow exploration.
                        if random.random() < 0.5:
                            S = S_final
                            # Don't set improved = True because size didn't increase
                            # But we did change the set.
                            # Let's just break and restart the local opt loop
                            # to see if this new structure allows additions.
                            break

        # Final check
        if S.bit_count() > best_size:
            best_set = S
            best_size = S.bit_count()

    # Convert bitset to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    return sorted(result)
