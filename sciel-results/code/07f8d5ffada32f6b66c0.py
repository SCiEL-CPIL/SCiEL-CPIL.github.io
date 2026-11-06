import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    MAX_SUM = 2 * N

    def is_valid_add(S_bits, counts, Tight_bits, x):
        if counts[2 * x] > 1:
            return False
        if (S_bits << x) & Tight_bits:
            return False
        return True

    def add_element(S, S_bits, counts, Tight_bits, x):
        changed_sums = [2*x]
        for y in S:
            counts[x + y] += 1
            changed_sums.append(x + y)
        counts[2*x] += 1
        
        for k in changed_sums:
            if counts[k] > 1:
                Tight_bits |= (1 << k)
            else:
                Tight_bits &= ~(1 << k)
                
        S.add(x)
        S_bits |= (1 << x)
        return S, S_bits, counts, Tight_bits

    def remove_element(S, S_bits, counts, Tight_bits, x):
        changed_sums = [2*x]
        for y in S:
            if y != x:
                counts[x + y] -= 1
                changed_sums.append(x + y)
        counts[2*x] -= 1
        
        for k in changed_sums:
            if counts[k] > 1:
                Tight_bits |= (1 << k)
            else:
                Tight_bits &= ~(1 << k)
                
        S.discard(x)
        S_bits &= ~(1 << x)
        return S, S_bits, counts, Tight_bits

    def greedy_init(order):
        S = set()
        S_bits = 0
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        Tight_bits = 0
        for x in order:
            if is_valid_add(S_bits, counts, Tight_bits, x):
                S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
        return S, S_bits, counts, Tight_bits

    def local_search(S, S_bits, counts, Tight_bits, max_iters, rng):
        best_S = S.copy()
        best_S_bits = S_bits
        best_counts = counts.copy()
        best_Tight_bits = Tight_bits
        best_size = len(S)
        
        S_list = list(S)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            if len(S) < 2:
                break
            
            # Try pure add first
            if rng.random() < 0.3:
                # Try to add a random valid element by sampling
                added = False
                for attempt in range(50):
                    x = rng.randint(1, N)
                    if x not in S and is_valid_add(S_bits, counts, Tight_bits, x):
                        S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
                        added = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.copy()
                            best_S_bits = S_bits
                            best_counts = counts.copy()
                            best_Tight_bits = Tight_bits
                        break
                if added:
                    continue
            
            # Remove-then-add (swap) move
            old_x = S_list[rng.randrange(len(S_list))]
            S, S_bits, counts, Tight_bits = remove_element(S, S_bits, counts, Tight_bits, old_x)
            S_list.remove(old_x)
            
            found = False
            for attempt in range(50):
                x = rng.randint(1, N)
                if x not in S and x != old_x and is_valid_add(S_bits, counts, Tight_bits, x):
                    S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
                    S_list.append(x)
                    found = True
                    break
                    
            if not found:
                S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, old_x)
                S_list.append(old_x)
            else:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_S_bits = S_bits
                    best_counts = counts.copy()
                    best_Tight_bits = Tight_bits
        
        return best_S, best_S_bits, best_counts, best_Tight_bits

    def final_polish(S, S_bits, counts, Tight_bits):
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in range(1, N + 1):
                if x not in S and is_valid_add(S_bits, counts, Tight_bits, x):
                    S, S_bits, counts, Tight_bits = add_element(S, S_bits, counts, Tight_bits, x)
                    added = True
                    break
        return S, S_bits, counts, Tight_bits

    # Phase 1: Multiple greedy initializations
    rng = random.Random(42)
    best_size = 0
    best_S = set()
    best_S_bits = 0
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    best_Tight_bits = 0
    
    # Try several different orderings
    for trial in range(8):
        if time.time() > deadline - 25:
            break
        
        if trial == 0:
            order = list(range(1, N + 1))
        elif trial == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        
        S, S_bits, counts, Tight_bits = greedy_init(order)
        size = len(S)
        
        if size > best_size:
            best_size = size
            best_S = S.copy()
            best_S_bits = S_bits
            best_counts = counts.copy()
            best_Tight_bits = Tight_bits
    
    # Phase 2: Local search from best initialization
    S, S_bits, counts, Tight_bits = best_S, best_S_bits, best_counts.copy(), best_Tight_bits
    best_S, best_S_bits, best_counts, best_Tight_bits = local_search(
        S, S_bits, counts, Tight_bits, 5000, rng
    )
    
    # Phase 2b: Second local search pass with different seed
    if time.time() < deadline - 15:
        rng2 = random.Random(123)
        S2, S2_bits, counts2, Tight_bits2 = best_S.copy(), best_S_bits, best_counts.copy(), best_Tight_bits
        best_S2, best_S2_bits, best_counts2, best_Tight_bits2 = local_search(
            S2, S2_bits, counts2, Tight_bits2, 3000, rng2
        )
        if len(best_S2) > len(best_S):
            best_S = best_S2
            best_S_bits = best_S2_bits
            best_counts = best_counts2
            best_Tight_bits = best_Tight_bits2
    
    # Phase 3: Final polish
    best_S, best_S_bits, best_counts, best_Tight_bits = final_polish(
        best_S, best_S_bits, best_counts.copy(), best_Tight_bits
    )

    return list(best_S)
