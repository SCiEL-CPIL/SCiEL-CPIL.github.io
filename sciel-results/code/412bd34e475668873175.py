import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s
    
    N = 256
    
    # Start with all odd numbers (128 elements) - known good base
    S = set(range(1, N + 1, 2))
    
    # Precompute conflict matrix: conflict[a][b] = True if (a*b)%(a+b)==0
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a][b] = True
                conflict[b][a] = True
    
    # Helper: count how many elements in S conflict with x
    def conflict_count(x, S):
        cnt = 0
        for y in S:
            if conflict[x][y]:
                cnt += 1
        return cnt
    
    # Helper: check if x can be added to set S
    def can_add(x, S):
        for y in S:
            if conflict[x][y]:
                return False
        return True
    
    # Helper: check validity of entire set
    def is_valid(S):
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if conflict[s_list[i]][s_list[j]]:
                    return False
        return True
    
    # Greedy add elements in given order
    def greedy_add(order, S_init):
        S = set(S_init)
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    # Local search: try removing one element and greedily adding others
    def local_search(S, all_candidates, max_rounds=20):
        current_S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 3:
            improved = False
            rounds += 1
            # Try removing each element
            for y in list(current_S):
                if time.time() > deadline - 3:
                    break
                test_S = current_S - {y}
                # Try to add all candidates not in test_S
                added = 0
                for x in all_candidates:
                    if x in test_S:
                        continue
                    if can_add(x, test_S):
                        test_S.add(x)
                        added += 1
                if added > 1:  # net gain
                    current_S = test_S
                    improved = True
                    break
        return current_S
    
    # Simulated annealing
    def anneal(S_init, all_candidates, steps=10000, T_start=20.0, T_end=0.01):
        S = set(S_init)
        best_S = set(S)
        best_size = len(S)
        
        # Precompute log temperature for smooth cooling
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            T = T_start * (T_end / T_start) ** (step / max(1, steps))
            T = max(T, 1e-10)
            
            # Choose a random move
            move_type = random.random()
            new_S = set(S)
            
            if move_type < 0.35:
                # Add a random element not in S
                not_in_S = [x for x in all_candidates if x not in S]
                if not not_in_S:
                    continue
                x = random.choice(not_in_S)
                new_S.add(x)
            elif move_type < 0.65:
                # Remove a random element from S
                if not S:
                    continue
                x = random.choice(list(S))
                new_S.discard(x)
            else:
                # Swap: remove one, add another
                if not S:
                    continue
                x = random.choice(list(S))
                not_in_S = [y for y in all_candidates if y not in S]
                if not not_in_S:
                    continue
                y = random.choice(not_in_S)
                new_S.discard(x)
                new_S.add(y)
            
            # Check validity
            if not is_valid(new_S):
                continue
            
            # Accept based on Metropolis criterion
            delta = len(new_S) - len(S)
            if delta >= 0 or random.random() < math.exp(delta / T):
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        return best_S
    
    # Multiple starting points and greedy orders
    best_S = set(S)
    best_size = len(S)
    
    evens = list(range(2, N + 1, 2))
    all_candidates = list(range(1, N + 1))
    
    # Order 1: natural order
    if time.time() < deadline - 10:
        S1 = greedy_add(evens, S)
        if len(S1) > best_size:
            best_S = S1
            best_size = len(S1)
    
    # Order 2: reversed order
    if time.time() < deadline - 10:
        S2 = greedy_add(reversed(evens), S)
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
    
    # Order 3: by increasing conflict count (prefer elements that conflict with fewer)
    if time.time() < deadline - 10:
        evens_by_conflict = sorted(evens, key=lambda x: conflict_count(x, S))
        S3 = greedy_add(evens_by_conflict, S)
        if len(S3) > best_size:
            best_S = S3
            best_size = len(S3)
    
    # Order 4-11: random shuffles
    for trial in range(8):
        if time.time() > deadline - 10:
            break
        evens_shuffled = evens[:]
        random.shuffle(evens_shuffled)
        S4 = greedy_add(evens_shuffled, S)
        if len(S4) > best_size:
            best_S = S4
            best_size = len(S4)
    
    # Local search on best
    if time.time() < deadline - 8:
        best_S = local_search(best_S, all_candidates, max_rounds=20)
        best_size = len(best_S)
    
    # Simulated annealing refinement
    if time.time() < deadline - 10:
        annealed = anneal(best_S, all_candidates, steps=12000, T_start=25.0, T_end=0.005)
        if len(annealed) > best_size:
            best_S = annealed
            best_size = len(best_S)
    
    # Another local search pass
    if time.time() < deadline - 5:
        best_S = local_search(best_S, all_candidates, max_rounds=25)
        best_size = len(best_S)
    
    # Second round: try different starting points if time permits
    if time.time() < deadline - 15:
        # Try starting with just odd numbers > 100
        high_odd = set(range(101, N + 1, 2))
        
        for order in [evens, evens[::-1], sorted(evens, key=lambda x: conflict_count(x, high_odd))]:
            if time.time() > deadline - 10:
                break
            S_alt = greedy_add(order, high_odd)
            if len(S_alt) > best_size:
                best_S = S_alt
                best_size = len(S_alt)
        
        if time.time() < deadline - 8:
            best_S = local_search(best_S, all_candidates, max_rounds=20)
            best_size = len(best_S)
        
        if time.time() < deadline - 10:
            annealed = anneal(best_S, all_candidates, steps=15000, T_start=30.0, T_end=0.003)
            if len(annealed) > best_size:
                best_S = annealed
                best_size = len(annealed)
    
    # Final local search and polish
    if time.time() < deadline - 3:
        best_S = local_search(best_S, all_candidates, max_rounds=30)
        best_size = len(best_S)
    
    # Final verification and return
    if not is_valid(best_S):
        # Fallback: return a known valid set
        best_S = set(range(1, N + 1, 2))
    
    return list(best_S)
