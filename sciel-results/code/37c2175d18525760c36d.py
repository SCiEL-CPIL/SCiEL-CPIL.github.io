import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    n = 128
    target_start = 128
    target_end = 256

    def evaluate(A_set):
        """
        Returns (max_mult, num_uncovered).
        Uses convolution for speed.
        """
        if not A_set:
            return 1000, 100
        
        # Create indicator vector
        v = np.zeros(129, dtype=np.int32)
        for x in A_set:
            if 0 <= x <= 128:
                v[x] = 1
            else:
                return 1000, 100
        
        # Convolution gives ordered sums
        conv = np.convolve(v, v, mode='full')
        
        max_mult = 0
        num_uncovered = 0
        
        for t in range(target_start, target_end + 1):
            ordered = conv[t]
            # Unordered multiplicity: (ordered + diag) / 2
            # diag is 1 if t/2 is in A and t is even, else 0
            if t % 2 == 0:
                mid = t // 2
                if 0 <= mid <= 128 and v[mid] == 1:
                    diag = 1
                else:
                    diag = 0
            else:
                diag = 0
            
            unordered = (ordered + diag) // 2
            
            if unordered == 0:
                num_uncovered += 1
            elif unordered > max_mult:
                max_mult = unordered
                
        return max_mult, num_uncovered

    def cost(A_set):
        m, u = evaluate(A_set)
        # Penalty for uncovered is huge to prioritize coverage first
        # Then minimize max multiplicity
        return u * 10000 + m

    # 1. Smart Initialization
    # We know 128 must be in A for 256.
    # A good starting point is a set that roughly covers the range.
    # Let's try a few different random initializations and pick the best.
    
    best_A = set()
    best_cost = float('inf')
    
    # Heuristic: A subset of [0, 128] of size k has k(k+1)/2 pairs.
    # We need to cover 129 values.
    # If k=20, ~210 pairs. Average multiplicity ~1.6. Max might be 3 or 4.
    # If k=30, ~465 pairs. Average ~3.6. Max might be 6-8.
    # We want small max mult.
    
    # Try different sizes
    for target_size in [20, 25, 30, 35, 40]:
        if time.time() > deadline - 10:
            break
            
        for _ in range(20): # 20 restarts per size
            A = set()
            A.add(128)
            # Add random elements
            while len(A) < target_size:
                A.add(random.randint(0, 128))
            
            c = cost(A)
            if c < best_cost:
                best_cost = c
                best_A = set(A)
                
            # Quick local improvement on this init
            # Hill Climbing: try removing/adding/swapping
            current_A = set(A)
            current_c = c
            
            for _ in range(100): # Quick polish
                if time.time() > deadline - 10:
                    break
                improved = False
                # Try removing
                if len(current_A) > 1:
                    x = random.choice(list(current_A))
                    test_A = current_A - {x}
                    c_test = cost(test_A)
                    if c_test < current_c:
                        current_A = test_A
                        current_c = c_test
                        improved = True
                # Try adding
                if not improved:
                    x = random.randint(0, 128)
                    if x not in current_A:
                        test_A = current_A | {x}
                        c_test = cost(test_A)
                        if c_test < current_c:
                            current_A = test_A
                            current_c = c_test
                            improved = True
                # Try swapping
                if not improved and len(current_A) > 1:
                    x = random.choice(list(current_A))
                    y = random.randint(0, 128)
                    if y != x:
                        test_A = (current_A - {x}) | {y}
                        c_test = cost(test_A)
                        if c_test < current_c:
                            current_A = test_A
                            current_c = c_test
                            improved = True
                            
            if current_c < best_cost:
                best_cost = current_c
                best_A = set(current_A)

    # 2. Simulated Annealing / Global Polish
    # Now we have a good candidate. Let's run a longer SA to try to reduce max multiplicity.
    # We prioritize coverage (u=0) and then minimize m.
    
    current_A = set(best_A)
    current_c = cost(current_A)
    best_global_A = set(current_A)
    best_global_c = current_c
    
    T = 5.0
    T_min = 0.01
    cooling = 0.99
    
    steps = 0
    while time.time() < deadline - 2:
        steps += 1
        move_type = random.random()
        new_A = set(current_A)
        
        if move_type < 0.3 and len(current_A) > 1:
            # Remove
            x = random.choice(list(current_A))
            new_A.remove(x)
        elif move_type < 0.6:
            # Add
            x = random.randint(0, 128)
            if x in current_A:
                continue
            new_A.add(x)
        else:
            # Swap
            if len(current_A) < 2:
                continue
            x = random.choice(list(current_A))
            y = random.randint(0, 128)
            if y == x:
                continue
            new_A.remove(x)
            new_A.add(y)
            
        if not new_A:
            continue
            
        new_c = cost(new_A)
        delta = new_c - current_c
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_A = new_A
            current_c = new_c
            
            if current_c < best_global_c:
                best_global_c = current_c
                best_global_A = set(current_A)
                
        T = max(T * cooling, T_min)
        
        # If we have a perfect coverage (u=0), we can focus more on reducing m
        # The cost function handles this.

    # 3. Final Deterministic Polish
    # Try to remove any redundant elements that don't increase max mult or break coverage
    # Iterate until no improvement
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        m_best, u_best = evaluate(best_global_A)
        
        for x in sorted(best_global_A):
            if time.time() > deadline - 1:
                break
            if x == 128 and len(best_global_A) == 1:
                continue # Can't remove last element, and 128 is critical for 256
            
            test_A = best_global_A - {x}
            if not test_A:
                continue
            m_test, u_test = evaluate(test_A)
            
            # Accept if coverage is maintained and max mult is not increased
            if u_test == 0 and u_best == 0:
                if m_test <= m_best:
                    best_global_A = test_A
                    improved = True
                    # Re-evaluate best
                    m_best, u_best = evaluate(best_global_A)
                    # Break loop to restart with new set size? Or just continue?
                    # Restarting is safer to ensure we check all elements against new state
                    break
            elif u_best > 0:
                # If we currently have uncovered, accept if it improves coverage or keeps same and reduces max
                if u_test < u_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break
                elif u_test == u_best and m_test < m_best:
                    best_global_A = test_A
                    improved = True
                    m_best, u_best = evaluate(best_global_A)
                    break

    # Ensure 128 is in the set (it should be, but just in case)
    if 128 not in best_global_A:
        best_global_A.add(128)

    return sorted(best_global_A)
