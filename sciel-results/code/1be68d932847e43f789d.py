import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    def is_valid_set(S):
        # S is a list of ints
        # Check for all a in S, a does not divide sum of any 3 distinct others
        # O(n^4) naive, but we can optimize
        n = len(S)
        if n < 4:
            return True
        # Precompute sums of all pairs?
        # For each a in S:
        #   For each triple (b,c,d) in S\{a}:
        #      if (b+c+d) % a == 0: return False
        
        # Optimization:
        # Sort S?
        # Actually, just iterate.
        # To speed up, we can break early.
        
        for i, a in enumerate(S):
            # Get indices of other elements
            others = [S[j] for j in range(n) if j != i]
            # Check all combinations of 3 from others
            # If len(others) < 3, continue
            if len(others) < 3:
                continue
            
            # Optimization: If a is large, it's harder to divide a sum?
            # Sum of 3 max is 3*192 = 576.
            # If a > 576, it can't divide the sum (since sum < a).
            # But max a is 192, so this doesn't help much.
            
            # Check
            # We can precompute pair sums of 'others' to reduce to O(k^2) per a?
            # Sum of 3 is x+y+z.
            # (x+y+z) % a == 0  <=>  (x+y) % a == -z % a
            
            # Let's stick to simple triplets for now, it might be fast enough for n=48.
            # 48 choose 3 is ~17000.
            # 48 * 17000 = 800,000 ops per full check.
            # If we do 100 checks, 80M ops. Might be tight in Python.
            
            # Better: Incremental check when adding x.
            
        return True

    def can_add(S, x):
        # Check if x can be added to S
        # 1. x does not divide sum of any 3 in S
        # 2. No a in S divides sum of any 3 in S U {x} that includes x
        
        # Part 1: x | sum(3 from S)
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # Part 2: a | x + y + z for a in S, y,z in S
        # y,z distinct, and y,z != a.
        # So we need y,z from S \ {a}.
        for a in S:
            # We need to check if there exist y,z in S\{a} such that (x+y+z) % a == 0
            # (y+z) % a == (-x) % a
            # Let target = (-x) % a
            # We need a pair y,z from S\{a} such that (y+z) % a == target
            
            # Precompute pair sums mod a?
            # Since S is small, just iterate pairs.
            # But we do this for every a.
            
            found = False
            # Iterate y, z in S
            # y != a, z != a, y != z
            for i in range(len(S)):
                y = S[i]
                if y == a: continue
                for j in range(i+1, len(S)):
                    z = S[j]
                    if z == a: continue
                    if (x + y + z) % a == 0:
                        return False
        return True

    # Initial Greedy
    def greedy(order):
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        return S

    # Generate candidates
    best_S = []
    
    # 1. Descending
    S1 = greedy(range(192, 0, -1))
    if len(S1) > len(best_S): best_S = S1
    
    # 2. Ascending (likely poor, but try)
    S2 = greedy(range(1, 193))
    if len(S2) > len(best_S): best_S = S2
    
    # 3. Random Restarts
    random.seed(42)
    while time.time() - start_time < deadline:
        # Random order
        order = list(range(1, 193))
        random.shuffle(order)
        S_rand = greedy(order)
        
        # Local Search on S_rand
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            # Try to add
            candidates = [x for x in range(1, 193) if x not in S_rand]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S_rand, x):
                    S_rand.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try swap
                # Remove one, add one
                # To keep it simple, just try adding. Swaps are complex to implement correctly with can_add.
                # We can do:
                # for y in S_rand:
                #    S_temp = S_rand - {y}
                #    for x in candidates:
                #       if can_add(S_temp, x):
                #          S_rand = S_temp + {x}
                #          improved = True
                #          break
                # This is expensive.
                
                # Let's just do a few random swaps if stuck
                for _ in range(10):
                    if time.time() - start_time > deadline: break
                    if not S_rand: break
                    y = random.choice(S_rand)
                    S_temp = [v for v in S_rand if v != y]
                    candidates = [x for x in range(1, 193) if x not in S_temp]
                    random.shuffle(candidates)
                    for x in candidates:
                        if can_add(S_temp, x):
                            S_rand = S_temp + [x]
                            improved = True
                            break
                    if improved:
                        break
        
        if len(S_rand) > len(best_S):
            best_S = S_rand
            # If we hit 48, we can stop early?
            if len(best_S) == 48:
                break

    return best_S
