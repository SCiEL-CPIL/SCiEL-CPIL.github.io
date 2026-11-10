import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    best_size = 0
    best_set = []
    
    def greedy_build(order):
        S = []
        counts = [0] * (2 * N + 1)
        for x in order:
            valid = True
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts[s] >= 2:
                    valid = False
            
            if valid:
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
                S.append(x)
        return S

    def local_search(S, max_rounds=5):
        """Try to improve set by removing and re-adding elements."""
        if not S:
            return S
        
        current_set = list(S)
        current_size = len(current_set)
        in_set = set(current_set)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing each element and see if we can add more
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline - 5.0:
                    break
                
                elem = current_set[i]
                # Temporarily remove elem
                temp_set = current_set[:i] + current_set[i+1:]
                temp_in_set = in_set - {elem}
                
                # Rebuild counts for temp_set
                counts = [0] * (2 * N + 1)
                for x in temp_set:
                    for y in temp_set:
                        if x <= y:
                            counts[x + y] += 1
                
                # Try to add back elements not in temp_set
                added = 0
                for x in range(1, N + 1):
                    if x in temp_in_set:
                        continue
                    
                    valid = True
                    # Check sums with existing elements
                    for y in temp_set:
                        s = x + y
                        if counts[s] >= 2:
                            valid = False
                            break
                    if valid:
                        s = 2 * x
                        if counts[s] >= 2:
                            valid = False
                    
                    if valid:
                        for y in temp_set:
                            counts[x + y] += 1
                        counts[2 * x] += 1
                        temp_set.append(x)
                        temp_in_set.add(x)
                        added += 1
                
                # If we added more than we removed, keep the improvement
                if added > 1:
                    current_set = temp_set
                    current_size = len(current_set)
                    in_set = temp_in_set
                    improved = True
            
            if not improved:
                break
        
        return current_set

    def try_order(order):
        nonlocal best_size, best_set
        S = greedy_build(order)
        
        # Apply local search
        S_improved = local_search(S)
        
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved

    # 1. Natural order
    try_order(list(range(1, N + 1)))

    # 2. Reverse order
    if time.time() < deadline:
        try_order(list(range(N, 0, -1)))

    # 3. Primes first
    if time.time() < deadline:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [x for x in range(1, N+1) if is_prime(x)]
        composites = [x for x in range(1, N+1) if not is_prime(x)]
        try_order(primes + composites)

    # 4. Random shuffles with local search
    rng = random.Random(42)
    no_improve_count = 0
    prev_best = best_size
    
    while time.time() < deadline - 10:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S = greedy_build(order)
        S_improved = local_search(S)
        
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > 20:
            break

    return best_set
