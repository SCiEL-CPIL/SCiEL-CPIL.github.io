import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85  # 5s margin
    
    # 1. Precompute 3-APs and 4-APs structures
    # taps_by_elem[x] contains list of (u, v) such that {x, u, v} is a 3-AP
    taps_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                # AP is (a, b, c)
                # For a: partners (b, c)
                taps_by_elem[a].append((b, c))
                # For b: partners (a, c)
                taps_by_elem[b].append((a, c))
                # For c: partners (a, b)
                taps_by_elem[c].append((a, b))
                
    # faps_by_elem[x] contains list of (u, v, w) such that {x, u, v, w} is a 4-AP
    faps_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                # AP is (a, b, c, e)
                # For a: partners (b, c, e)
                faps_by_elem[a].append((b, c, e))
                # For b: partners (a, c, e)
                faps_by_elem[b].append((a, c, e))
                # For c: partners (a, b, e)
                faps_by_elem[c].append((a, b, e))
                # For e: partners (a, b, c)
                faps_by_elem[e].append((a, b, c))

    # Helper: Count 3-APs in bitmask A
    def count_3aps(A_mask):
        count = 0
        # Iterate through all 3-APs? Or iterate through elements?
        # Iterating through elements and checking pairs is O(N * deg_3)
        # Total 3-APs is ~ N^2/6. 
        # Let's just sum over all elements x in A, count pairs (u,v) in taps_by_elem[x] 
        # such that u,v in A, and divide by 3 (since each AP is counted 3 times, once for each element).
        # Wait, iterating through all x in A is fast if we iterate bits.
        # But checking pairs for each x involves bit shifts.
        
        # Alternative: Precompute list of all 3-APs as (a,b,c) and check mask.
        # Number of 3-APs in [1, 256] is approx sum_{d=1}^{127} (256 - 2d) ~ 256*128/2 ~ 16k.
        # Checking 16k APs is fast in Python if optimized.
        
        # Let's stick to the incremental update strategy in the main loop.
        # For initialization, we can just compute from scratch.
        
        total = 0
        # We need a list of all 3-APs to count them efficiently from scratch?
        # Or use the taps_by_elem.
        # Sum over x in 1..N:
        #   if x in A:
        #     for (u, v) in taps_by_elem[x]:
        #       if u in A and v in A:
        #         total += 1
        # Return total // 3
        
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                for u, v in taps_by_elem[x]:
                    if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1):
                        total += 1
        return total // 3

    # Helper: Check if adding x to A_mask is valid (no 4-AP created)
    def can_add(x, A_mask):
        for u, v, w in faps_by_elem[x]:
            if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1) and ((A_mask >> (w - 1)) & 1):
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added to A_mask
    def gain_add(x, A_mask):
        gain = 0
        for u, v in taps_by_elem[x]:
            if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1):
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed from A_mask
    def loss_remove(x, A_mask):
        loss = 0
        for u, v in taps_by_elem[x]:
            if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1):
                loss += 1
        return loss

    # 2. Initialization: Greedy Construction
    # Start with empty set
    A_mask = 0
    current_score = 0
    
    # To speed up greedy init, we can just iterate elements 1..N and add if valid and beneficial?
    # Better: Sort elements by some heuristic? 
    # Simple greedy: Iterate 1..N, add if valid.
    # This creates a 4-AP free set.
    for x in range(1, N + 1):
        if can_add(x, A_mask):
            A_mask |= (1 << (x - 1))
            current_score += gain_add(x, A_mask)
            
    best_mask = A_mask
    best_score = current_score

    # 3. Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.001
    
    # Precompute list of elements for random choice
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            # Restart temperature occasionally to escape deep local optima
            if random.random() < 0.1:
                T = 0.5
            else:
                T = min_T
        
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Find a random x not in A
            # Instead of building a list of candidates every time, pick random x and check
            # If N is 256, picking a random x in 1..N and checking if it's in A is O(1).
            # Probability of hitting a new element is (N - |A|)/N.
            # If |A| is large, this is inefficient.
            # Let's maintain a list of candidates? Or just try a few times.
            # For N=256, it's small.
            
            # Efficient way: pick random x. If in A, retry a few times.
            for _ in range(5):
                x = random.choice(all_elements)
                if not ((A_mask >> (x - 1)) & 1):
                    break
            else:
                continue # Couldn't find an element to add (A is full?)
                
            if can_add(x, A_mask):
                gain = gain_add(x, A_mask)
                new_score = current_score + gain
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    A_mask |= (1 << (x - 1))
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = A_mask

        elif move_type == 'remove':
            # Pick random x in A
            # Find bits set in A_mask
            # We can maintain a list of active elements? 
            # Or just pick random x and check if in A.
            for _ in range(5):
                x = random.choice(all_elements)
                if (A_mask >> (x - 1)) & 1:
                    break
            else:
                continue
                
            loss = loss_remove(x, A_mask)
            new_score = current_score - loss
            delta = new_score - current_score # delta is negative
            
            if random.random() < math.exp(delta / T):
                A_mask &= ~(1 << (x - 1))
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask

        else: # swap
            # Pick x in A, y not in A
            for _ in range(5):
                x = random.choice(all_elements)
                if (A_mask >> (x - 1)) & 1:
                    break
            else:
                continue
                
            for _ in range(5):
                y = random.choice(all_elements)
                if not ((A_mask >> (y - 1)) & 1):
                    break
            else:
                continue
                
            if x == y: continue
            
            # Check if y can be added to A \ {x}
            # A' = A - x + y
            # 4-AP constraint for y:
            # For any 4-AP (y, u, v, w), u,v,w must NOT all be in A'.
            # A' contains A\{x}.
            # So u,v,w must not all be in A, OR one of them is x (which is removed).
            
            valid = True
            for u, v, w in faps_by_elem[y]:
                # Check if u,v,w are all in A'
                # A' has x removed.
                # If any of u,v,w is x, it is not in A'. So condition fails -> valid.
                # If none are x, they must be in A.
                if u == x or v == x or w == x:
                    continue
                if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1) and ((A_mask >> (w - 1)) & 1):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Calculate score change
            # Loss from removing x: 3-APs containing x where other 2 are in A.
            loss = 0
            for u, v in taps_by_elem[x]:
                if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1):
                    loss += 1
                    
            # Gain from adding y: 3-APs containing y where other 2 are in A'.
            gain = 0
            for u, v in taps_by_elem[y]:
                # u, v must be in A'
                # A' = A - {x} + {y}
                # So u, v must be in A, and neither can be x.
                if u == x or v == x:
                    continue
                if ((A_mask >> (u - 1)) & 1) and ((A_mask >> (v - 1)) & 1):
                    gain += 1
            
            new_score = current_score - loss + gain
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                A_mask &= ~(1 << (x - 1))
                A_mask |= (1 << (y - 1))
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                    
        T *= cooling_rate
        
    # Convert best_mask to sorted list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res), res)
