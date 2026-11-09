import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = 60.0  # 60 seconds limit, keep margin
    
    # Initial seed: 1 to 22
    # lcm(a, b) <= 512 for all a, b <= 22 is true?
    # Max lcm is lcm(21, 22) = 462. Yes.
    S = set(range(1, 23))
    
    # Candidates to consider
    # We can try to add any number from 23 to 512
    # But we should prioritize numbers that are likely to fit.
    # Numbers with small prime factors are better.
    # Let's just iterate through all candidates.
    
    # Optimization: Precompute lcm for all pairs? 
    # 512 * 512 / 2 = ~130,000 pairs. 
    # Storing a boolean matrix or set of valid pairs is efficient.
    
    # Let's precompute valid pairs
    # valid_pairs[u] = set of v such that lcm(u, v) <= 512
    # This takes 512*512/2 = 131072 lcm calculations. Very fast.
    
    N = 512
    valid_pairs = [set() for _ in range(N + 1)]
    
    for u in range(1, N + 1):
        for v in range(u, N + 1):
            if lcm(u, v) <= N:
                valid_pairs[u].add(v)
                if u != v:
                    valid_pairs[v].add(u)
    
    # Now S is a set. 
    # Check if x can be added to S:
    # x must be in valid_pairs[y] for all y in S.
    # i.e., S must be a subset of valid_pairs[x].
    
    def can_add(x, S):
        for y in S:
            if x not in valid_pairs[y]:
                return False
        return True

    # Greedy addition
    # Sort candidates by some heuristic? 
    # Smaller numbers are generally more compatible.
    # But we already have 1..22.
    # Let's try adding from 23 upwards.
    
    for x in range(23, N + 1):
        if time.time() > deadline - 10:
            break
        if can_add(x, S):
            S.add(x)
            
    # Local Search: Swap
    # Try to swap y in S with x not in S if it leads to a valid set.
    # Since we want to maximize size, a swap keeps size constant.
    # But it might enable further additions.
    # So, if we swap, we should restart the greedy addition phase.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all swaps
        # To avoid O(N^2 * |S|) which might be slow, we can be smarter.
        # But N=512, |S|~50. 512*512*50 = 13M checks. 
        # Each check is O(|S|) if we re-verify, but we can use valid_pairs.
        # Check if S \ {y} U {x} is valid:
        # 1. x compatible with S \ {y}
        # 2. y compatible with S \ {y} (already true)
        # So we just need x compatible with S \ {y}.
        
        S_list = list(S)
        for x in range(1, N + 1):
            if x in S:
                continue
            # Check if x can replace any y in S
            # If x is compatible with all elements in S except potentially y, 
            # and y is compatible with all others (true), then swap is valid.
            # Actually, if x is compatible with S \ {y}, then S \ {y} U {x} is valid.
            # We don't need x to be incompatible with y.
            
            # Optimization: 
            # Find y in S such that x is NOT compatible with y.
            # If no such y exists, x could be added directly (handled in greedy).
            # So we look for y in S where x not in valid_pairs[y].
            # If there is exactly one such y, we can try swapping y with x.
            # If there are multiple, we can't swap just one to make it valid.
            
            incompatible_ys = [y for y in S_list if x not in valid_pairs[y]]
            
            if len(incompatible_ys) == 1:
                y = incompatible_ys[0]
                # Check if swapping y with x is valid.
                # It is valid if x is compatible with S \ {y}.
                # Which is true by definition of incompatible_ys having size 1.
                # So S' = S - {y} + {x} is valid.
                
                # Should we swap?
                # Greedy addition didn't add x because it was incompatible with y.
                # By removing y, we free up "space" (in terms of constraints).
                # Does this help add MORE numbers later?
                # Likely yes, if x is "better" than y or allows other numbers.
                # Heuristic: Swap if x > y? Or if x has more divisors?
                # Let's just swap if it leads to a state where we can add more.
                # Simple strategy: Swap and restart greedy.
                
                S.remove(y)
                S.add(x)
                improved = True
                # Restart greedy addition with new S
                for x2 in range(1, N + 1):
                    if time.time() > deadline - 2:
                        break
                    if x2 not in S and can_add(x2, S):
                        S.add(x2)
                
                # If we swapped, we should break out of the x loop and restart the while loop
                # to re-evaluate other potential swaps/additions with the new S.
                break 
                
            # If len(incompatible_ys) == 0, it means x is compatible with all S, 
            # which should have been added in the greedy phase. 
            # Unless S changed. But we are in the swap loop.
            
            # If len(incompatible_ys) > 1, we can't fix it with a single swap.
            
        # If we broke out of x loop due to a swap, the while loop continues.
        # If we finished x loop without swap, improved is False, loop ends.
        
    return list(S)
