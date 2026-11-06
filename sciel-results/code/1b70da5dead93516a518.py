import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85  # 5s margin
    
    # Precompute 3-APs and 4-APs
    # 3-APs: (a, a+d, a+2d)
    taps = []
    taps_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                ap = (a, b, c)
                taps.append(ap)
                for x in ap:
                    taps_by_elem[x].append(ap)
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    faps = []
    faps_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ap = (a, b, c, e)
                faps.append(ap)
                for x in ap:
                    faps_by_elem[x].append(ap)

    # Helper to count 3-APs in a set
    def count_3aps(A_set):
        count = 0
        for ap in taps:
            if ap[0] in A_set and ap[1] in A_set and ap[2] in A_set:
                count += 1
        return count

    # Helper to check if adding x to A_set creates a 4-AP
    def can_add(x, A_set):
        for ap in faps_by_elem[x]:
            others = [y for y in ap if y != x]
            if all(y in A_set for y in others):
                return False
        return True

    # Helper to check if A_set is 4-AP free (for initialization/verification)
    def is_4ap_free(A_set):
        for ap in faps:
            if all(x in A_set for x in ap):
                return False
        return True

    # Initial solution: Greedy seed
    A = set()
    for x in range(1, N + 1):
        if can_add(x, A):
            A.add(x)
    
    current_score = count_3aps(A)
    best_A = set(A)
    best_score = current_score

    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # We will use moves: Add, Remove, Swap
    # To speed up, we precompute the "impact" of adding/removing?
    # For now, direct calculation is fine for N=256.
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = 1.0
            # Restart or continue? Continue with high T to explore
            # Or just keep going with low T for polishing. 
            # Let's reset T occasionally to allow exploration.
            if random.random() < 0.1:
                T = 1.0
            else:
                break # If we hit min T, we are polishing. Let's just stop if no improvement for a while? 
                # Actually, just let it run.
        
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Pick random x not in A
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                continue
            x = random.choice(candidates)
            if can_add(x, A):
                A.add(x)
                new_score = current_score + sum(1 for ap in taps_by_elem[x] if ap[0] in A and ap[1] in A and ap[2] in A and x not in ap) 
                # Wait, the sum above is wrong. 
                # New 3-APs are those containing x where other 2 are in A.
                # Since x was not in A, no 3-AP containing x existed before.
                # So new_score = current_score + count_3aps_containing(x, A)
                
                # Let's recalculate properly to be safe, N is small
                # count_3aps is O(#taps) ~ 10k. 
                # #taps ~ 256*128/3 ~ 10000. 
                # 10000 operations per move. 1000 moves/sec -> 10M ops/sec. 
                # Python is slow. 10M ops might take 1-2 seconds. 
                # So we can do ~500-1000 moves per second.
                # 85 seconds -> ~50k moves.
                
                # Let's optimize the score update.
                # Add x:
                # Gain: 3-APs containing x with other 2 in A.
                gain = 0
                for ap in taps_by_elem[x]:
                    others = [y for y in ap if y != x]
                    if all(y in A for y in others):
                        gain += 1
                new_score = current_score + gain
                
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(A)
                current_score = new_score
            else:
                # Rejection
                pass
                
        elif move_type == 'remove':
            if not A:
                continue
            x = random.choice(list(A))
            # Loss: 3-APs containing x with other 2 in A.
            loss = 0
            for ap in taps_by_elem[x]:
                others = [y for y in ap if y != x]
                if all(y in A for y in others):
                    loss += 1
            new_score = current_score - loss
            
            A.remove(x)
            current_score = new_score
            # Note: Removing never creates a 4-AP, so always valid.
            
        else: # swap
            if not A:
                continue
            x = random.choice(list(A))
            candidates = [y for y in range(1, N + 1) if y not in A]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # Check if swapping x->y is 4-AP free
            # New set A' = (A - {x}) + {y}
            # 4-APs in A' must not exist.
            # 4-APs in A were free.
            # 4-APs involving x are gone.
            # 4-APs involving y might be created.
            # Check 4-APs containing y.
            valid = True
            for ap in faps_by_elem[y]:
                others = [z for z in ap if z != y]
                # All others must be in A'
                # A' contains A\{x} and y.
                # So others must be in A\{x}.
                if any(z == x for z in others):
                    continue # z is x, which is removed, so z not in A'
                if all(z in A for z in others):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate score change
            # Lose 3-APs containing x (others in A)
            loss = 0
            for ap in taps_by_elem[x]:
                others = [z for z in ap if z != x]
                if all(z in A for z in others):
                    loss += 1
                    
            # Gain 3-APs containing y (others in A')
            gain = 0
            for ap in taps_by_elem[y]:
                others = [z for z in ap if z != y]
                # others must be in A'
                # A' = A - {x} + {y}
                # So others must be in A, and none can be x.
                if any(z == x for z in others):
                    continue
                if all(z in A for z in others):
                    gain += 1
            
            new_score = current_score - loss + gain
            
            # Acceptance
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                A.remove(x)
                A.add(y)
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(A)
            
        T *= cooling_rate
        
    return sorted(best_A)

# Run once to check
if __name__ == "__main__":
    res = construct()
    print(len(res), res)
