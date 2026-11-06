import random
import time
import itertools

def construct():
    best_set = []
    start_time = time.time()
    
    # B_3[1] set: all unordered triples (with repetition) have unique sums
    # We use a greedy approach with randomized ordering, trying many times
    # to find the largest set possible.
    
    # Strategy: use a combination of greedy with random restarts
    # Also try to be smart about which elements to pick (spread out values)
    
    # Pre-compute: for a candidate x and existing set A, the new sums are:
    # 3x, 2x+y for y in A, x+y+z for y<=z in A
    # Total new sums: 1 + k + k*(k+1)/2 = (k+1)(k+2)/2 where k = len(A)
    
    for trial in range(3000):
        if time.time() - start_time > 55:
            break
        
        # Different randomization strategies
        if trial % 3 == 0:
            # Uniform random order
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        elif trial % 3 == 1:
            # Try to spread out: interleave low and high
            low = list(range(1, 2049))
            high = list(range(2049, 4097))
            random.shuffle(low)
            random.shuffle(high)
            candidates = []
            li, hi = 0, 0
            while li < len(low) or hi < len(high):
                if li < len(low):
                    candidates.append(low[li])
                    li += 1
                if hi < len(high):
                    candidates.append(high[hi])
                    hi += 1
        else:
            # Bias towards geometric spacing
            base = random.uniform(1.05, 1.3)
            candidates = []
            v = 1
            while v <= 4096:
                candidates.append(int(v))
                v *= base
            random.shuffle(candidates)
            # Fill in some random extras
            extra = [x for x in range(1, 4097) if x not in set(candidates)]
            random.shuffle(extra)
            candidates = candidates + extra
        
        A = []
        sum_set = set()
        
        for x in candidates:
            k = len(A)
            num_expected = (k + 1) * (k + 2) // 2
            
            # Quick check: generate new sums and check for collisions
            s_new = []
            s_new.append(3 * x)
            for y in A:
                s_new.append(2 * x + y)
            for i in range(k):
                y = A[i]
                for j in range(i, k):
                    z = A[j]
                    s_new.append(x + y + z)
            
            # Check internal collisions (duplicates within s_new)
            if len(s_new) != len(set(s_new)):
                continue
            
            s_new_set = set(s_new)
            
            # Check collisions with existing sums
            if s_new_set & sum_set:
                continue
            
            # Valid, add x
            A.append(x)
            sum_set.update(s_new_set)
            
            # If we've already exceeded our best, we can continue to try for more
            # But if we hit 80, we can stop this trial early
            if len(A) >= 80:
                break
        
        if len(A) > len(best_set):
            best_set = A
            if len(best_set) >= 80:
                break
    
    return best_set
