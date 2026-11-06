import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    # Max possible d is (512-1)//2 = 255
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Helper to update bits from list
    def get_bits(lst):
        b = 0
        for x in lst:
            b |= (1 << x)
        return b

    # Seeds
    seeds = []
    # Seed 1: Interval [1..32]
    seeds.append(list(range(1, 33)))
    
    # Seed 2: Interval [1..31] + [33]
    seeds.append(list(range(1, 32)) + [33])
    
    # Seed 3: Random
    seeds.append(random.sample(range(1, MAXV + 1), N))
    
    # Seed 4: Two intervals
    seeds.append(list(range(1, 17)) + list(range(256, 272)))
    
    # Seed 5: Spread out
    seeds.append([i * 16 + 1 for i in range(32)]) # 1, 17, 33...
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_list = seed[:]
        current_list.sort()
        current_bits = get_bits(current_list)
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        local_deadline = start_time + min(20.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Pick x to remove
            x_idx = random.randint(0, N - 1)
            x = current_list[x_idx]
            
            # Generate candidates for y
            # 1. Random candidate
            # 2. Heuristic candidates from pairs
            candidates = set()
            
            # Heuristic: Generate candidates that complete 3-term progressions
            # For each pair (u, v) in current_list, find potential third terms
            # Note: We are removing x, so we should generate candidates based on current_list 
            # but maybe excluding x? Or including? 
            # If we remove x, the set is current_list \ {x}.
            # Candidates should be based on the remaining set to ensure we add something useful 
            # to the new set.
            
            # Let's use the current set for generation, it's close enough.
            # Or better, use current_list without x.
            # Since N is small, we can just iterate.
            
            # To save time, we might not want to generate ALL candidates for EVERY move 
            # if it's too slow, but 500 pairs is fast.
            
            # Let's collect candidates from pairs in current_list
            # We will pick one random candidate from this set.
            
            # Optimization: If we do this every step, it's 500 ops. 
            # Score is 256 ops. 
            # It's comparable.
            
            # Let's do it.
            
            # We need to be careful not to pick x or existing elements.
            # We will filter later.
            
            # To avoid heavy overhead, we can sample a few pairs.
            # But 32 elements is small. Let's do all pairs.
            
            # Actually, let's just use the full current_list for candidate generation.
            # The removal of x is a minor perturbation.
            
            # Generate candidates
            # We'll use a list to avoid set overhead if possible, but set is fine for dedup.
            
            # We will pick a random candidate from the generated set.
            # If the set is empty, fallback to random.
            
            # To keep it fast, we might limit the number of pairs we check if N was larger,
            # but N=32 is fine.
            
            # Let's implement the candidate generation.
            
            # We can precompute pairs? No, the set changes.
            
            # Let's just do it inline.
            
            # 10% chance of pure random
            if random.random() < 0.1:
                y = random.randint(1, MAXV)
                if y == x or ((current_bits >> y) & 1):
                    continue
            else:
                # Generate candidates
                # We'll use a list of candidates and pick one
                cand_list = []
                
                # Iterate over pairs in current_list
                # We can use indices to avoid creating a new list
                for i in range(N):
                    u = current_list[i]
                    for j in range(i+1, N):
                        v = current_list[j]
                        
                        # y = 2v - u
                        y1 = 2*v - u
                        if 1 <= y1 <= MAXV:
                            cand_list.append(y1)
                        
                        # y = 2u - v
                        y2 = 2*u - v
                        if 1 <= y2 <= MAXV:
                            cand_list.append(y2)
                        
                        # y = (u+v)/2
                        if (u+v) % 2 == 0:
                            y3 = (u+v)//2
                            if 1 <= y3 <= MAXV:
                                cand_list.append(y3)
                
                if not cand_list:
                    y = random.randint(1, MAXV)
                    if y == x or ((current_bits >> y) & 1):
                        continue
                else:
                    # Pick a random candidate
                    y = random.choice(cand_list)
                    if y == x or ((current_bits >> y) & 1):
                        # Retry or fallback?
                        # Just skip this move if invalid?
                        # Or pick another?
                        # Let's just skip to save time, as there are many candidates.
                        continue

            # Update bits
            new_bits = current_bits ^ (1 << x) ^ (1 << y)
            
            # Update list
            new_list = current_list[:x_idx] + current_list[x_idx+1:]
            ins_idx = bisect.bisect_left(new_list, y)
            new_list.insert(ins_idx, y)
            
            new_score = _score(new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
