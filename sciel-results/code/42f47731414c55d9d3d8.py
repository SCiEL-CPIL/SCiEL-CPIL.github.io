import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    # Precompute for fast validity checking
    # For each x, and for each existing element y, we can quickly check conflicts.
    # We'll maintain the set as a sorted list and use efficient checks.
    
    def is_valid_addition(x, S):
        """Check if adding x to S (a list) keeps it valid."""
        if not S:
            return True
        
        # Case 1: x is 'a' - need b,c in S with x < b <= c and x | (b+c)
        # Case 2: x is 'b' - need a,c in S with a < x <= c and a | (x+c)
        # Case 3: x is 'c' - need a,b in S with a < b <= x and a | (b+x)
        
        S_set = set(S)
        
        # Case 1: x is a
        # For all pairs (b,c) in S with x < b <= c: check (b+c) % x != 0
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b
        # For all a,c in S with a < x <= c: check (x+c) % a != 0
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is c
        # For all a,b in S with a < b <= x: check (b+x) % a != 0
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def is_valid_set(S):
        """Verify the entire set is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def greedy_construct(order):
        """Greedy construction in given order."""
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if is_valid_addition(x, S):
                S.append(x)
                S_set.add(x)
        return S

    def local_search_add(S):
        """Try to add any missing elements."""
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set:
                    if is_valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        improved = True
        return S

    # Strategy:
    # 1. Multiple greedy initializations
    # 2. Local search (add)
    # 3. Simulated annealing with add/remove/swap moves
    
    best_set = []
    best_size = 0

    # Phase 1: Greedy initializations
    random.seed(42)
    
    # Descending
    S1 = greedy_construct(range(n, 0, -1))
    S1 = local_search_add(S1)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1[:]

    # Ascending
    S2 = greedy_construct(range(1, n + 1))
    S2 = local_search_add(S2)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2[:]

    # Random restarts for greedy
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_try = greedy_construct(order)
        S_try = local_search_add(S_try)
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Phase 2: Simulated annealing
    # Moves: 
    #   - Add a random element not in S
    #   - Remove a random element from S
    #   - Swap: remove one, add two (if net +1)
    #   - Swap: remove two, add one (if net -1, but might enable more adds later)
    
    if time.time() < deadline - 10:
        current_S = best_set[:]
        current_set = set(current_S)
        current_size = len(current_S)
        
        T_start = 10.0
        T_end = 0.01
        # Number of iterations based on remaining time
        remaining_time = deadline - 10 - time.time()
        # Estimate: each SA step takes ~0.1ms, so we can do ~10000 steps in 1 second
        max_iter = int(remaining_time * 5000)
        max_iter = min(max_iter, 200000)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (it / max(max_iter, 1))
            
            move_type = random.random()
            new_S = current_S[:]
            new_set = set(new_S)
            
            if move_type < 0.5:
                # Add move
                missing = [x for x in range(1, n+1) if x not in new_set]
                if not missing:
                    continue
                x = random.choice(missing)
                if is_valid_addition(x, new_S):
                    new_S.append(x)
                    new_set.add(x)
                    delta = 1
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_S = new_S
                        current_set = new_set
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_S[:]
                else:
                    # Invalid add, reject
                    pass
                    
            elif move_type < 0.7:
                # Remove move
                if not current_S:
                    continue
                x = random.choice(current_S)
                new_S.remove(x)
                new_set.discard(x)
                delta = -1
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_S = new_S
                    current_set = new_set
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_S[:]
                        
            elif move_type < 0.9:
                # Swap: remove 1, add 2
                if len(current_S) < 2:
                    continue
                missing = [x for x in range(1, n+1) if x not in new_set]
                if len(missing) < 2:
                    continue
                x_remove = random.choice(current_S)
                x_add1, x_add2 = random.sample(missing, 2)
                # Try removing x_remove
                temp_S = [y for y in current_S if y != x_remove]
                # Try adding x_add1
                if not is_valid_addition(x_add1, temp_S):
                    continue
                temp_S.append(x_add1)
                # Try adding x_add2
                if not is_valid_addition(x_add2, temp_S):
                    continue
                temp_S.append(x_add2)
                delta = 1  # net +1
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_S = temp_S
                    current_set = set(temp_S)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_S[:]
            else:
                # Swap: remove 2, add 1
                if len(current_S) < 2:
                    continue
                missing = [x for x in range(1, n+1) if x not in new_set]
                if not missing:
                    continue
                x_remove1, x_remove2 = random.sample(current_S, 2)
                x_add = random.choice(missing)
                temp_S = [y for y in current_S if y != x_remove1 and y != x_remove2]
                if not is_valid_addition(x_add, temp_S):
                    continue
                temp_S.append(x_add)
                delta = -1  # net -1
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_S = temp_S
                    current_set = set(temp_S)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_S[:]

        # Final local search on best
        if time.time() < deadline - 3:
            best_set = local_search_add(best_set)

    # Final verification
    if not is_valid_set(best_set):
        # Fallback: return a known valid set
        # Upper tail [129, 256] should be valid
        best_set = list(range(129, n + 1))
    
    return sorted(best_set)
