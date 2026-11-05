"""Optimized construction for max 3-AP, no 4-AP subset of [1,256] using bitmasks."""
import time
import random
import sys
from collections import Counter

N = 256
DEADLINE_MARGIN = 10  # seconds before the 90s limit

# Precompute all 4-term APs and their bitmasks
FAP = []  # list of (a, b, c, d) tuples (1-indexed)
fap_masks = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            FAP.append((a, b, c, e))
            mask = 0
            mask |= 1 << (a - 1)
            mask |= 1 << (b - 1)
            mask |= 1 << (c - 1)
            mask |= 1 << (e - 1)
            fap_masks.append(mask)

# Precompute 3-AP counting masks
# For each element x (0-indexed in bit, 1-indexed in value), 
# we want to count pairs (a, c) such that a + c = 2*x and a, c in A.
# Let's define a mask M_x such that for any set A, popcount(A & M_x) 
# is NOT directly the count. 
# Instead, let's precompute for each x, the list of (a, c) pairs.
# But that's what we are trying to speed up.
# 
# Alternative: For each x, create a bitmask P_x where bit i is 1 if 2*x - i is in [1, N].
# Then the number of 3-APs centered at x is popcount(A & P_x & (A shifted? No)).
# Actually, if we have A, the number of pairs (a, c) with a+c=2x is 
# sum(1 for a in A if 2x-a in A and a < x). 
# 
# Let's use a different fast counting method.
# Total 3-APs = sum_{x in A} (number of a in A such that 2x-a in A and a < x)
# 
# We can precompute for each x (1..N) a bitmask CenterMask[x] where bit i is set 
# if i and 2*x-i are both valid indices in [1, N].
# Then for a given set A (bitmask), the number of 3-APs centered at x is 
# popcount(A & CenterMask[x]).
# Wait, this counts ordered pairs (a, c) and (c, a). 
# If a != c, it counts 2. If a == c (impossible for distinct terms in AP with d>0? 
# No, d>0 means a<x<c, so a!=c).
# So popcount(A & CenterMask[x]) counts the number of elements a in A such that 2x-a is also in A.
# Since a < x < c, for every valid AP (a,x,c), a is in A and c is in A.
# The term a contributes 1 to the count, and the term c contributes 1 to the count 
# when we consider the pair (c, a)? No, CenterMask[x] has bits for both a and c.
# So if a and c are in A, both bit a and bit c are set in CenterMask[x] AND A.
# So popcount(A & CenterMask[x]) = 2 * (number of 3-APs centered at x).
# 
# Let's verify. 
# x=5. AP (3,5,7). 
# CenterMask[5] has bits for 3 and 7 (and others like 1,9; 2,8; etc).
# If A={3,5,7}, A & CenterMask[5] has bits 3 and 7 set. Popcount = 2.
# Number of APs = 1. So count = popcount // 2.
# 
# This is very fast! O(N) per evaluation (iterate over x in A, do popcount).
# Or even faster: Precompute CenterMask for all x.
# Then count_3aps(A) = sum(popcount(A & CenterMask[x]) for x in A) // 2.

CenterMasks = [0] * (N + 1)
for x in range(1, N + 1):
    mask = 0
    for a in range(1, x):
        c = 2 * x - a
        if c <= N and c > x:
            mask |= 1 << (a - 1)
            mask |= 1 << (c - 1)
    CenterMasks[x] = mask

def count_3aps_fast(mask):
    """Count 3-APs in bitmask representation."""
    count = 0
    # Iterate over set bits in mask
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        x = lsb.bit_length()  # 1-indexed position
        # Count pairs centered at x
        count += (mask & CenterMasks[x]).bit_count()
        m -= lsb
    return count // 2

def has_4ap_fast(mask):
    """Check if bitmask has any 4-AP."""
    for fm in fap_masks:
        if (mask & fm) == fm:
            return True
    return False

def greedy_construct():
    """Greedy: add elements that maximize 3-AP contribution while avoiding 4-APs."""
    selected_mask = 0
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        candidate_mask = selected_mask | bit
        if not has_4ap_fast(candidate_mask):
            selected_mask = candidate_mask
    return selected_mask

def local_search(mask, time_limit=5.0):
    """Local search: try adding/removing/swapping elements to improve 3-AP count."""
    start = time.time()
    current_score = count_3aps_fast(mask)
    best_mask = mask
    best_score = current_score
    
    # Precompute list of all bits
    all_bits = [1 << (x - 1) for x in range(1, N + 1)]
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding elements
        for bit in all_bits:
            if mask & bit:
                continue
            candidate = mask | bit
            if has_4ap_fast(candidate):
                continue
            new_score = count_3aps_fast(candidate)
            if new_score > current_score:
                mask = candidate
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                break
        
        if not improved:
            # Try removing elements
            m = mask
            while m:
                bit = m & -m
                candidate = mask ^ bit
                new_score = count_3aps_fast(candidate)
                if new_score > current_score:
                    mask = candidate
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                    break
                m -= bit
        
        if not improved:
            # Try swaps
            # To speed up, we can limit swaps to a few random ones if time is tight,
            # but with fast evaluation, we can try more.
            # We'll do a limited number of random swaps to avoid O(N^2) per iteration
            # if we don't find an improvement quickly.
            swaps_tried = 0
            max_swaps_per_iter = 50
            while swaps_tried < max_swaps_per_iter:
                # Pick random bit to remove
                m = mask
                if not m: break
                # Random selection from set bits is tricky without list, 
                # so we'll just pick a random index and check if it's set
                # Or better, maintain a list of set bits? 
                # For now, let's just iterate through a random subset of candidates.
                # Actually, let's just do a full scan of swaps but break early if found.
                # With N=256, N^2 = 65k swaps. Each swap check is fast.
                # Let's try full scan.
                break # Disable full scan for now, rely on add/remove. 
                # If we need swaps, we can add them back later. 
                # Given the structure, add/remove might be sufficient for local optima.
                # Let's add a simple random swap mechanism.
                if random.random() < 0.1: # Only sometimes try swaps
                    for _ in range(10):
                        # Pick random x to add, y to remove
                        x = random.randint(1, N)
                        y = random.randint(1, N)
                        bit_x = 1 << (x - 1)
                        bit_y = 1 << (y - 1)
                        if mask & bit_x or not (mask & bit_y):
                            continue
                        candidate = (mask ^ bit_y) | bit_x
                        if has_4ap_fast(candidate):
                            continue
                        new_score = count_3aps_fast(candidate)
                        if new_score > current_score:
                            mask = candidate
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_mask = mask
                            break
                if improved:
                    break
                swaps_tried += 1
            
            if not improved:
                # If no improvement, break out of loop
                # Or continue with simulated annealing logic? 
                # For now, break.
                break
                
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    best_mask = 0
    best_score = -1
    
    # Strategy 1: Greedy forward
    if time.time() - start_time < DEADLINE_MARGIN:
        m1 = greedy_construct()
        s1 = count_3aps_fast(m1)
        if s1 > best_score:
            best_score = s1
            best_mask = m1
        if time.time() - start_time < DEADLINE_MARGIN:
            m1_ls, s1_ls = local_search(m1, time_limit=min(10.0, DEADLINE_MARGIN - (time.time() - start_time)))
            if s1_ls > best_score:
                best_score = s1_ls
                best_mask = m1_ls
    
    # Strategy 2: Randomized Greedy with multiple restarts
    if time.time() - start_time < DEADLINE_MARGIN:
        for trial in range(100):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            random.seed(trial)
            order = list(range(1, N + 1))
            random.shuffle(order)
            selected_mask = 0
            for x in order:
                bit = 1 << (x - 1)
                candidate = selected_mask | bit
                if not has_4ap_fast(candidate):
                    selected_mask = candidate
            s3 = count_3aps_fast(selected_mask)
            if s3 > best_score:
                best_score = s3
                best_mask = selected_mask
            # Quick local search on this random result
            if time.time() - start_time < DEADLINE_MARGIN - 5:
                m_ls, s_ls = local_search(selected_mask, time_limit=1.0)
                if s_ls > best_score:
                    best_score = s_ls
                    best_mask = m_ls
    
    # Final local search on best
    if best_mask and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 2.0:
            final_ls, final_s = local_search(best_mask, time_limit=min(20.0, remaining))
            if final_s > best_score:
                best_score = final_s
                best_mask = final_ls
    
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        x = lsb.bit_length()
        result.append(x)
        m -= lsb
    result.sort()
    
    if not result:
        result = [1]
        
    return result

def validate(result):
    sel = set(result)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps_fast(0) # dummy
    # Recalculate score for printing
    mask = 0
    for x in result:
        mask |= 1 << (x - 1)
    score = count_3aps_fast(mask)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
