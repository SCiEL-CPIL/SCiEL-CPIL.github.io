"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Precompute conflict matrix: conflict[i][j] is True if lcm(i+2, j+2) <= 256
    # We only care about numbers in [2, 256]
    # Index 0 corresponds to number 2, Index 254 corresponds to number 256
    n = 255
    # Use a list of bitmasks for fast conflict checking? 
    # Or just a 2D boolean array. 255x255 is small.
    conflict = [[False] * n for _ in range(n)]
    
    for i in range(n):
        a = i + 2
        for j in range(i + 1, n):
            b = j + 2
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            if a * b // g <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    # Objective: Maximize sum(1/x) for x in selected.
    # Strategy: 
    # 1. Greedy "Large First": Iterate from 256 down to 2. 
    #    Why? If we pick a large number, it has a small reciprocal (low cost in sum, but we want to MAXIMIZE sum, so we want HIGH reciprocals? 
    #    Wait. Maximize sum(1/x). 
    #    Small x -> Large 1/x. 
    #    So we WANT small numbers. 
    #    BUT small numbers conflict with many others. 
    #    Large numbers have small 1/x, but conflict with fewer others.
    #    
    #    Let's re-evaluate. 
    #    If I pick 2, I gain 0.5. I cannot pick any even number. I cannot pick 4, 6, 8... 
    #    If I pick 3, I gain 0.333. I cannot pick 6, 9, 12...
    #    
    #    The "value" of a number is 1/x. 
    #    The "cost" is the set of numbers it excludes.
    #    
    #    Greedy by Value/Cost ratio? Or just Greedy by Value?
    #    If I pick 2 (value 0.5), I lose all evens. 
    #    If I pick 3 (value 0.33), I lose multiples of 3.
    #    
    #    Let's try Greedy by Value (Smallest numbers first).
    #    Try 2: Add 2.
    #    Try 3: Add 3.
    #    Try 4: lcm(4,2)=4 <= 256. Conflict.
    #    Try 5: Add 5.
    #    Try 6: lcm(6,2)=6, lcm(6,3)=6. Conflict.
    #    Try 7: Add 7.
    #    ...
    #    This is the current seed program's logic (iterating 2..256). 
    #    The seed program iterates 2..256 and adds if compatible. 
    #    This is "Greedy by Index (Small to Large)".
    #    Since Value = 1/x, this is "Greedy by Value (High to Low)".
    #    
    #    Is this optimal? 
    #    Consider swapping. 
    #    Maybe excluding 2 allows me to include 4, 6, 8... 
    #    If I exclude 2, I can include 4 (val 0.25), 6 (val 0.16), 8 (val 0.125)...
    #    Sum of reciprocals of evens > 256? No, evens are <= 256.
    #    If I drop 2, I gain 1/4 + 1/6 + 1/8 + ... 
    #    But 4 and 6 conflict with each other? lcm(4,6)=12 <= 256. 
    #    So I can't just take all evens.
    #    
    #    Let's look at the structure. 
    #    The condition lcm(a,b) > 256 is quite loose for large numbers.
    #    If a > 128 and b > 128, then a*b > 16384. 
    #    lcm(a,b) >= max(a,b) > 128. 
    #    Actually, if a=129, b=130, lcm is large. 
    #    The only way lcm(a,b) <= 256 is if a and b share factors or are small.
    #    Specifically, if a and b are both > 128, their lcm is at least min(a,b) > 128. 
    #    Wait, lcm(a,b) is a multiple of a and b. 
    #    If a=129, b=129 (not distinct). 
    #    If a=129, b=138. gcd(129,138)=3. lcm = 129*138/3 = 129*46 = 5934 > 256.
    #    It seems ALL pairs of distinct numbers > 128 have lcm > 256?
    #    Let's check: min(a,b) > 128. 
    #    lcm(a,b) >= a * b / gcd(a,b).
    #    Worst case for lcm is when gcd is large.
    #    If a = 129, b = 258 (not in range).
    #    If a = 135, b = 270 (not in range).
    #    Max number is 256.
    #    If a, b > 128, then a,b in [129, 256].
    #    Can lcm(a,b) <= 256?
    #    lcm(a,b) is a multiple of a. So lcm(a,b) >= a.
    #    If a=129, lcm >= 129. Next multiple is 258 > 256.
    #    So if a > 128, the only multiple of a in [1, 256] is a itself.
    #    Therefore, lcm(a,b) must be a multiple of a. 
    #    If a > 128, the only multiple of a <= 256 is a.
    #    So lcm(a,b) = a implies b divides a.
    #    But b is distinct from a. If b divides a and b < a, then b <= a/2.
    #    If a > 128, a/2 > 64.
    #    So if we pick two numbers a, b > 128, can one divide the other?
    #    If b divides a, then a = k*b. Since a,b > 128, k must be 1 (if k>=2, a >= 2*129 = 258 > 256).
    #    So no number > 128 divides another number > 128 (except itself).
    #    Thus, for any distinct a, b > 128, lcm(a,b) > 256.
    #    
    #    CONCLUSION: The set {129, 130, ..., 256} is a VALID subset.
    #    What is its score? Sum(1/x for x in 129..256).
    #    Approx integral 129 to 256 of 1/x dx = ln(256) - ln(129) = ln(256/129) ≈ ln(1.98) ≈ 0.68.
    #    
    #    Can we do better?
    #    The current greedy (small to large) picks 2, 3, 5, 7, 11, 13, 17...
    #    1/2 + 1/3 + 1/5 + 1/7 + ...
    #    0.5 + 0.33 + 0.2 + 0.14 + 0.09 + ...
    #    This sum grows like log(log(N)) for primes, but we include composites too if compatible.
    #    Actually, the greedy picks 2. Then 3. Then 5. 
    #    Does it pick 4? No (conflict with 2).
    #    Does it pick 6? No (conflict with 2,3).
    #    Does it pick 8? No (conflict with 2).
    #    It essentially picks numbers that are "relatively prime" to the set in a strong sense?
    #    No, just lcm > 256.
    #    
    #    Let's compare {129..256} vs Greedy Small.
    #    Greedy Small includes 2 (0.5). 
    #    {129..256} includes 129 (0.0077).
    #    The small numbers have MUCH higher individual value.
    #    However, small numbers block many others.
    #    
    #    Let's try a different strategy: 
    #    Start with the set of all numbers > 128. This is valid.
    #    Then, try to ADD smaller numbers if they are compatible with ALL current members.
    #    If we add 128, it must have lcm(128, x) > 256 for all x in 129..256.
    #    lcm(128, 129) = 128*129 > 256.
    #    lcm(128, 130) = 128*130/2 = 64*130 = 8320 > 256.
    #    Actually, 128 is a power of 2. 
    #    If x is odd, gcd(128,x)=1, lcm=128x > 256 (since x>=129).
    #    If x is even, say x=130. gcd(128,130)=2. lcm=128*130/2 = 8320.
    #    If x=128? Not distinct.
    #    What if x=64? 64 is not in the set (we started with >128).
    #    So 128 is compatible with all numbers in 129..256?
    #    Let's check x=256. gcd(128,256)=128. lcm=256. 
    #    Condition is STRICTLY > 256. 
    #    So lcm(128, 256) = 256 is NOT > 256. 
    #    So 128 conflicts with 256.
    #    
    #    So we cannot simply add 128 to {129..256} because of 256.
    #    We might need to remove 256 to add 128.
    #    Value(128) = 1/128 = 0.0078.
    #    Value(256) = 1/256 = 0.0039.
    #    Swap 256 for 128: Gain 0.0039. 
    #    But does 128 conflict with others?
    #    lcm(128, 192). 192 is in 129..256.
    #    gcd(128, 192) = 64. lcm = 128*192/64 = 2*192 = 384 > 256. OK.
    #    lcm(128, 224). gcd=32. lcm = 128*224/32 = 4*224 = 896. OK.
    #    
    #    This suggests a Local Search starting from the "Large" set is promising.
    #    1. Initialize S = {129, ..., 256}.
    #    2. Calculate Score.
    #    3. Try to add any number x in [2, 128] if compatible with all in S.
    #    4. Try to swap x in S with y not in S if y is compatible with S\{x} and Score(S - x + y) > Score(S).
    #    
    #    This is a different family: "Large-Set Local Search" vs "Small-Set Greedy".
    
    # Initialize with large numbers
    S = set(range(129, 257))
    
    # Helper to check if x can be added to set S
    def can_add(x, S):
        for y in S:
            if x == y:
                continue
            # Fast check: if x*y > 256 * gcd(x,y)
            # Or just compute lcm
            g = gcd(x, y)
            if x * y // g <= 256:
                return False
        return True

    # Phase 1: Try to add smaller numbers
    # Iterate from 2 to 128
    for x in range(2, 129):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    # Phase 2: Local Search / Swaps
    # We want to maximize sum(1/x).
    # Try swapping a large number in S with a smaller number not in S.
    # Or adding a number if we remove conflicting ones?
    # Simple swap: Remove y from S, Add x to S\{y} if valid.
    
    # Precompute values
    vals = {x: 1.0/x for x in range(2, 257)}
    
    # Current score
    current_score = sum(vals[x] for x in S)
    
    # List of candidates to add (smaller numbers not in S)
    candidates = [x for x in range(2, 129) if x not in S]
    # List of elements to remove (larger numbers in S)
    # We should prioritize removing large numbers (low value) to replace with small numbers (high value)
    # Sort S by value ascending (largest numbers first, lowest value)
    S_list = sorted(S, key=lambda x: vals[x])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each candidate x to add
        for x in candidates:
            if time.time() > deadline:
                break
            # Try to swap x with some y in S
            # We only swap if x is "better" than y, i.e., 1/x > 1/y => x < y
            # So x should be smaller than y.
            for y in S_list:
                if x >= y:
                    break # Since S_list is sorted by value asc, i.e., x_num desc. 
                          # Wait. vals[x] = 1/x. 
                          # Sorted by vals[x] ascending means 1/x ascending means x descending.
                          # So S_list is [256, 255, ...].
                          # We want to replace y with x.
                          # If x < y, then 1/x > 1/y. Potential gain.
                          # If x >= y, 1/x <= 1/y. No gain from swap (unless it allows other additions later, 
                          # but let's stick to direct gain for now).
                
                # Check if x is compatible with S \ {y}
                # S is a set. 
                valid = True
                for z in S:
                    if z == y:
                        continue
                    if z == x:
                        continue
                    g = gcd(x, z)
                    if x * z // g <= 256:
                        valid = False
                        break
                
                if valid:
                    # Calculate new score
                    # new_score = current_score - 1/y + 1/x
                    # Since x < y, this is an increase.
                    S.remove(y)
                    S.add(x)
                    current_score += vals[x] - vals[y]
                    improved = True
                    # Update S_list? 
                    # S_list is stale. 
                    # For efficiency, we can just re-sort or break and restart loop.
                    # Given small N, re-sorting is fine.
                    S_list = sorted(S, key=lambda x: vals[x])
                    # Remove x from candidates since it's in S now
                    if x in candidates:
                        candidates.remove(x)
                    break # Restart inner loop with new S
        
    # Final check for validity (safety)
    final_list = list(S)
    for i in range(len(final_list)):
        for j in range(i+1, len(final_list)):
            a = final_list[i]
            b = final_list[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                # Should not happen, but if it does, return empty or valid subset
                # To be safe, return the large set which is always valid
                return list(range(129, 257))
                
    return final_list

# EVOLVE-BLOCK-END
