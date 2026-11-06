import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    TARGET = 200
    MAX_ELE = 100
    MASK = (1 << (TARGET + 1)) - 1  # bits 0 through 200
    
    # Step 1: Identify the "Large" region [51, 100]
    # Any subset of [51, 100] with size >= 4 sums to > 200.
    # Pairs in [51, 100] sum to <= 199.
    # So only triples x+y+z=200 are the forbidden configurations in this region.
    
    large_vals = list(range(51, 101))
    val_to_idx = {v: i for i, v in enumerate(large_vals)}
    
    # Generate all bad triples (x, y, z) with 51 <= x < y < z <= 100 and x+y+z=200
    triples = []
    for z in range(100, 66, -1):
        target_xy = 200 - z
        # x < y => x < target_xy / 2
        # y < z => target_xy - x < z => x > target_xy - z
        # x >= 51
        min_x = max(51, target_xy - z + 1)
        max_x = (target_xy - 1) // 2
        
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        # No constraints in large region, take all
        result = list(large_vals)
        # Try to add small ones
        bitset = 1
        for x in result:
            bitset = bitset | ((bitset << x) & MASK)
            
        for x in range(50, 0, -1):
            if not (bitset & (1 << (TARGET - x))):
                result.append(x)
                bitset = bitset | ((bitset << x) & MASK)
        return result

    # Step 2: Find a Minimum (or near-minimum) Hitting Set H for these triples
    # H is a set of values to REMOVE from [51, 100]
    
    # Greedy Hitting Set
    H = set()
    # To make it deterministic but allow slight variation if needed, we use a fixed seed or just standard greedy
    # Standard Greedy: Pick element with max frequency in remaining triples
    
    # We represent triples as sets of values for easy checking
    active_triples = [set(t) for t in triples]
    
    while active_triples:
        # Count frequency of each value in active_triples
        freq = {}
        for t in active_triples:
            for v in t:
                freq[v] = freq.get(v, 0) + 1
                
        if not freq:
            break
            
        # Pick value with max frequency
        # Tie-breaking: pick largest value? Smallest? 
        # Usually doesn't matter much for this size, but let's pick the one with max freq, then max value
        best_v = max(freq.items(), key=lambda item: (item[1], item[0]))[0]
        
        H.add(best_v)
        
        # Remove all triples containing best_v
        active_triples = [t for t in active_triples if best_v not in t]
        
        if time.time() > deadline - 5.0:
            break

    # Step 3: Local Search / Reduction on H
    # Try to remove any element from H if it is not essential
    # An element v in H is essential if there is a triple T in `triples` such that T ∩ H == {v}
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for v in list(H):
            # Check if v is essential
            essential = False
            for t in triples:
                t_set = set(t)
                # Count how many elements of H are in t
                hits = sum(1 for h in H if h in t_set)
                if hits == 1 and v in t_set:
                    essential = True
                    break
            
            if not essential:
                H.remove(v)
                improved = True

    # Step 4: Construct the set A from [51, 100] \ H
    A = [v for v in large_vals if v not in H]
    
    # Step 5: Compute subset sums of A using bitset
    bitset = 1
    for x in A:
        bitset = bitset | ((bitset << x) & MASK)
        
    # Step 6: Try to add elements from [1, 50]
    # We can add x if (bitset has 200-x) is False
    # Note: Adding x updates bitset.
    
    # To maximize additions, we might want to try adding smaller numbers first?
    # Or larger? 
    # Smaller numbers create more "room" for other numbers in terms of sum, 
    # but they are more likely to form sums with existing large numbers.
    # Let's try adding from 1 to 50.
    
    for x in range(1, 51):
        if time.time() > deadline - 2.0:
            break
        # Check if adding x creates a sum of 200
        # New sums are old_sums and old_sums + x
        # 200 is in new_sums if 200 in old_sums (False) OR 200-x in old_sums
        if not (bitset & (1 << (TARGET - x))):
            A.append(x)
            bitset = bitset | ((bitset << x) & MASK)

    # Final Verification (Safety)
    # Check if any subset of A sums to 200
    # We can re-compute bitset from scratch to be sure, or trust the incremental update.
    # Trusting incremental is faster. But let's do a quick sanity check on size.
    
    if not A:
        return [100]
        
    return A

# Quick test to ensure no crashes
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    target = 200
    mask = (1 << (target + 1)) - 1
    bitset = 1
    for x in res:
        bitset = bitset | ((bitset << x) & mask)
    if bitset & (1 << target):
        print("INVALID: Sum 200 found")
    else:
        print("VALID")
