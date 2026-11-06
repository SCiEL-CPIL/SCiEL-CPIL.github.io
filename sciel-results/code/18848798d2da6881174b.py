import time
import random

def construct():
    deadline = time.time() + 110.0
    
    def score_set(S):
        # S is a set of 32 ints
        # Convert to list for iteration
        S_list = list(S)
        n = len(S_list)
        diffs = set()
        # To avoid O(N log N) sort, we can just iterate pairs
        # But we need to check membership in S, which is O(1) for set
        for i in range(n):
            si = S_list[i]
            for j in range(i + 1, n):
                sj = S_list[j]
                d = sj - si
                if d > 0: # Always true if j>i and distinct
                    if (sj + d) in S:
                        diffs.add(d)
        return len(diffs)

    best_set = None
    best_score = -1
    
    # We will run SA for a while
    # To ensure we explore different basins, we do restarts
    while time.time() < deadline:
        current_set = set(random.sample(range(1, 513), 32))
        current_score = score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        # Estimate max steps based on time? No, just loop and check time.
        # 5000 steps is fast. Let's do 20000 steps per restart max.
        max_steps = 20000
        
        for _ in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Pick random element to remove
            # Using list for random access
            # We can maintain a list version of current_set to pick from
            # But set is needed for membership.
            # Let's just pick random from set? No, set iteration is unordered.
            # Convert to list for picking? That's O(N).
            # Or keep a parallel list?
            # Let's keep a list `current_list` and set `current_set` in sync.
            
            # Actually, picking random from set is not directly supported efficiently.
            # random.sample(list(S), 1) is O(N). N=32. Fine.
            rem_elem = random.choice(list(current_set))
            add_elem = random.randint(1, 512)
            
            if add_elem in current_set:
                continue
                
            # New set
            # Instead of creating a new set every time, we can modify in place?
            # No, we need to revert if rejected.
            # Creating a new set is O(N).
            new_set = set(current_set)
            new_set.remove(rem_elem)
            new_set.add(add_elem)
            
            new_score = score_set(new_set)
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                if T > T_min:
                    # Accept with probability exp(delta / T)
                    # delta is negative
                    prob = pow(2.71828, delta / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_score = new_score
            
            T *= cooling_rate
            if T < T_min:
                break

    if best_set is None:
        return list(range(1, 33))
        
    return sorted(best_set)
