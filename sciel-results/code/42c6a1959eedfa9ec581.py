import time
import random
from fractions import Fraction

N = 256
TIME_LIMIT = 85.0  # Safety margin for 90s kill

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = [Fraction(1, i) for i in range(N + 1)]
    
    # Precompute for each x, the list of (a, b) such that x = 2b - a
    # This allows O(1) or O(k) checks for validity when adding/removing x
    # Specifically, if we add x, we must ensure no a, b in set such that 2b - a = x
    # This is equivalent to: for all a, b in set, x != 2b - a.
    # Or: for all a in set, if b = (a+x)/2 is integer and in set, invalid.
    # Or: for all b in set, if a = 2b-x is integer and in set, invalid.
    
    # Let's use a bitmask for the set S.
    # A state is valid if for all a, b in S with a < b, 2b - a is not in S.
    
    # Initial state: Greedy construction from seed program
    def greedy_init():
        mask = 0
        # We iterate 1..256
        for x in range(1, N + 1):
            # Check if adding x creates a 3-term AP
            # x can be the middle term: a, x, c -> c = 2x - a. Need a in S, c in S? No, c not in S yet.
            # x can be the last term: a, b, x -> b = (a+x)/2. Need a in S, b in S.
            # x can be the first term: x, b, c -> b = (x+c)/2. Need b in S, c in S.
            
            valid = True
            # Case 1: x is the 3rd term (largest). a, b, x. b = (a+x)/2.
            for a in range(1, x):
                if mask & (1 << (a - 1)):
                    b = (a + x) // 2
                    if (a + x) % 2 == 0 and 1 <= b <= N and (mask & (1 << (b - 1))):
                        valid = False
                        break
            if not valid:
                continue
                
            # Case 2: x is the 2nd term (middle). x, y, z. y = 2x - z? No.
            # If x is middle, we have a, x, c. c = 2x - a.
            # We need to check if there exists a in S such that c = 2x - a is in S?
            # Wait, if we are adding x, we haven't added c yet if c > x.
            # But if c < x, it might be in S.
            # Actually, if x is the middle term, the progression is a, x, c.
            # Since we are adding x, and a, c are already in S or not.
            # If a, c are in S, then a, x, c is a progression.
            # So we check: for all a in S, let c = 2x - a. If c in S, invalid.
            for a in range(1, N + 1):
                if mask & (1 << (a - 1)):
                    c = 2 * x - a
                    if 1 <= c <= N and (mask & (1 << (c - 1))):
                        valid = False
                        break
            if not valid:
                continue
            
            # Case 3: x is the 1st term (smallest). x, b, c. b = (x+c)/2.
            # We need to check if there exists c in S such that b = (x+c)/2 is in S.
            # Since x is smallest, b, c > x. They might be in S.
            for c in range(x + 1, N + 1):
                if mask & (1 << (c - 1)):
                    b = (x + c) // 2
                    if (x + c) % 2 == 0 and 1 <= b <= N and (mask & (1 << (b - 1))):
                        valid = False
                        break
            if not valid:
                continue
                
            mask |= (1 << (x - 1))
            
        return mask

    current_mask = greedy_init()
    best_mask = current_mask
    best_score = sum(recip[i] for i in range(1, N + 1) if (current_mask >> (i - 1)) & 1)
    
    # Helper to calculate score quickly?
    # Exact sum of reciprocals is required.
    # Updating the score incrementally is hard with Fractions.
    # But N=256, so summing 256 fractions is fast enough?
    # Let's estimate: 256 additions of Fractions.
    # Fractions can grow large. LCM of 1..256 is huge.
    # However, Python's Fraction handles this.
    # Let's check speed of summing 256 fractions.
    
    def calc_score(mask):
        s = Fraction(0, 1)
        for i in range(1, N + 1):
            if (mask >> (i - 1)) & 1:
                s += recip[i]
        return s

    current_score = best_score
    
    # Check validity of a mask
    def is_valid(mask):
        # Check all pairs a, b in mask with a < b
        # If 2b - a in mask, invalid.
        # Optimization: iterate set bits.
        # For N=256, max size of 3-free set is ~256/log(256) ~ 30-40?
        # Actually, largest 3-free subset of [1, N] is roughly N / log N.
        # For N=256, maybe ~30-50 elements.
        # Pairs ~ 2500. Checks ~ 2500.
        # This is very fast.
        
        # Extract indices
        indices = [i for i in range(1, N + 1) if (mask >> (i - 1)) & 1]
        n = len(indices)
        for i in range(n):
            a = indices[i]
            for j in range(i + 1, n):
                b = indices[j]
                c = 2 * b - a
                if 1 <= c <= N and (mask >> (c - 1)) & 1:
                    return False
        return True

    # Simulated Annealing
    # Move: flip a bit.
    # If adding a bit: check validity. If valid, accept.
    # If removing a bit: always valid.
    
    # We want to maximize score.
    # Removing a bit decreases score. Adding a bit increases score.
    # So we prefer adding.
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # To make it efficient, we can try to add bits first.
    # If stuck, remove bits to allow other additions.
    
    # Precompute neighbors? No, N=256, just pick random index.
    
    while time.time() - start_time < TIME_LIMIT:
        # Pick a random index to flip
        idx = random.randint(1, N)
        bit = 1 << (idx - 1)
        
        if current_mask & bit:
            # Remove bit
            new_mask = current_mask & ~bit
            # Always valid to remove
            new_score = current_score - recip[idx]
            # Since score decreases, we only accept if random chance (SA)
            # delta_score = new_score - current_score = -recip[idx]
            # This is negative.
            # Accept with probability exp(delta/T)
            # But delta is Fraction. exp needs float.
            # Use float approximation for decision, exact for tracking.
            delta_float = -float(recip[idx])
            if random.random() < pow(2.71828, delta_float / T):
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Add bit
            # Check validity
            # Only need to check if adding idx creates a progression.
            # A progression involving idx must have idx as one of the terms.
            # 1. idx is 3rd term: a, b, idx. b = (a+idx)/2.
            # 2. idx is 2nd term: a, idx, c. c = 2*idx - a.
            # 3. idx is 1st term: idx, b, c. b = (idx+c)/2.
            
            valid = True
            # Check 1: idx is 3rd
            for a in range(1, idx):
                if (current_mask >> (a - 1)) & 1:
                    b = (a + idx) // 2
                    if (a + idx) % 2 == 0 and 1 <= b <= N and ((current_mask >> (b - 1)) & 1):
                        valid = False
                        break
            if not valid:
                continue
                
            # Check 2: idx is 2nd
            for a in range(1, N + 1):
                if a == idx: continue
                if (current_mask >> (a - 1)) & 1:
                    c = 2 * idx - a
                    if 1 <= c <= N and ((current_mask >> (c - 1)) & 1):
                        valid = False
                        break
            if not valid:
                continue
            
            # Check 3: idx is 1st
            for c in range(idx + 1, N + 1):
                if (current_mask >> (c - 1)) & 1:
                    b = (idx + c) // 2
                    if (idx + c) % 2 == 0 and 1 <= b <= N and ((current_mask >> (b - 1)) & 1):
                        valid = False
                        break
            if not valid:
                continue
                
            # Valid to add
            new_mask = current_mask | bit
            new_score = current_score + recip[idx]
            # Score increases, so always accept
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Maybe reheat? Or just keep going with low T (greedy)
            # Since we have time, let's restart with higher T if stuck?
            # For now, just keep cooling to T_min.
            
    # Return the best mask found as a list
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    if not result:
        # Fallback to greedy
        return greedy_init_list()
    return result

def greedy_init_list():
    # Same as greedy_init but returns list
    selected = []
    for x in range(1, 257):
        if all(2*b-a!=x for a in selected for b in selected if a<b):
            selected.append(x)
    return selected

# Wait, the helper `greedy_init_list` is defined after `construct` uses it?
# No, `construct` uses `greedy_init` (mask version) inside.
# The fallback in `construct` calls `greedy_init_list`.
# I should define `greedy_init_list` before `construct` or ensure it's available.
# In Python, function definitions are executed at module load, but the body is executed when called.
# So `greedy_init_list` must be defined before `construct()` is called.
# The code block puts `construct` first. This is fine as long as `greedy_init_list` is defined
# before the evaluator calls `construct()`. Since the whole block is loaded, it is.

# Let's double check the validity check logic in SA.
# When adding `idx`:
# We check if `idx` forms an AP with existing elements.
# The APs are (a, b, idx), (a, idx, c), (idx, b, c).
# My checks:
# 1. (a, b, idx): a < b < idx. b = (a+idx)/2.
#    Loop a from 1 to idx-1. If a in set, check b.
#    This covers all a < idx.
# 2. (a, idx, c): a < idx < c. c = 2*idx - a.
#    Loop a from 1 to N. If a in set, check c.
#    Note: if a > idx, then c < idx. This is covered?
#    If a > idx, c = 2idx - a < idx.
#    The progression is (c, idx, a).
#    This is the same as (a, idx, c) with a and c swapped? No, order matters for AP.
#    A 3-term AP is defined by values {x, x+d, x+2d}.
#    If we have a, idx, c in the set, and they form an AP, it doesn't matter which is which.
#    So checking all a in set, c = 2idx - a, if c in set, is sufficient to detect
#    any AP where idx is the middle term.
#    Does it cover cases where idx is not the middle?
#    If idx is the smallest: (idx, b, c). b = (idx+c)/2.
#    If idx is the largest: (a, b, idx). b = (a+idx)/2.
#    
#    My Check 2 (idx is middle) covers: a, idx, c.
#    My Check 1 (idx is largest) covers: a, b, idx.
#    My Check 3 (idx is smallest) covers: idx, b, c.
#    
#    Are there overlaps or misses?
#    If a, b, c form an AP and idx is one of them.
#    If idx is middle, Check 2 finds it.
#    If idx is largest, Check 1 finds it.
#    If idx is smallest, Check 3 finds it.
#    
#    So yes, these three checks are exhaustive for "idx is part of an AP with two other existing elements".
#    
#    One detail: In Check 2, I loop `a` over all 1..N.
#    If `a` is in set, I check `c`.
#    If `a == idx`, skip.
#    If `a < idx`, `c > idx`.
#    If `a > idx`, `c < idx`.
#    This correctly identifies if `idx` is the middle term of an AP with two other elements in the set.
#    
#    In Check 1, I loop `a` from 1 to idx-1.
#    This assumes `a < idx`. Since `idx` is the largest term, `a` must be smaller.
#    `b` is between `a` and `idx`.
#    This correctly identifies if `idx` is the largest term.
#    
#    In Check 3, I loop `c` from idx+1 to N.
#    This assumes `c > idx`. Since `idx` is the smallest term, `c` must be larger.
#    `b` is between `idx` and `c`.
#    This correctly identifies if `idx` is the smallest term.
#    
#    Logic seems sound.

# One optimization: 
# In Check 2, looping 1..N is 256 iterations.
# In Check 1, looping 1..idx-1 is up to 255.
# In Check 3, looping idx+1..N is up to 255.
# Total ~768 checks per addition attempt.
# Plus bit operations.
# This is very fast.
# SA steps can be ~100k-1M in 85s?
# Python is slow. Maybe 10k-100k steps.
# Greedy gets a decent start. SA can improve it.

# Let's refine the code structure.

