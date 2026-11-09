import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25  # 5s margin
    
    best_size = float('inf')
    best_set = []
    
    # Pre-compute nothing, just use sets
    
    for seed in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        A = []
        S = set()
        # Addable elements are those not in A and not blocked
        # Initially all are addable? No, we check dynamically.
        
        # We maintain a list of candidates
        # To speed up, we can just iterate 1..N each time
        # But N=256 is small.
        
        while True:
            # Find all addable elements
            addable = []
            for x in range(1, N + 1):
                if x in A:
                    continue
                # Check if x can be added
                # 1. 2x not in S
                if 2 * x in S:
                    continue
                # 2. x+a not in S for all a in A
                ok = True
                for a in A:
                    if x + a in S:
                        ok = False
                        break
                if ok:
                    addable.append(x)
            
            if not addable:
                # Maximal set found
                if len(A) < best_size:
                    best_size = len(A)
                    best_set = list(A)
                break
            
            # Choose x from addable to minimize |Add(A U {x})|
            best_x = None
            min_new_addable_count = float('inf')
            
            # To speed up, we can sample if addable is large?
            # But N is small. Let's check all.
            # Optimization: If len(addable) is large, checking all is O(N * |A| * N) ?
            # No, for each x in addable, we simulate.
            # Simulation cost: O(|A|) to check validity (already done) + O(N * |A|) to count new addable.
            # Total per step: O(|Add| * N * |A|).
            # 256 * 256 * 20 ~ 1.3M.
            # 1000 seeds * 20 steps * 1.3M = 26 Billion ops? Too slow in Python.
            
            # We need a faster heuristic or approximation.
            # Approximation: Instead of counting exact new addable,
            # estimate the number of new sums and how many they block.
            # Or, just pick x that creates the most "conflicts" for others?
            
            # Let's try a simpler greedy:
            # Pick x that maximizes the number of y in Addable such that y becomes blocked?
            # y becomes blocked if y+a' in S' or 2y in S'.
            # S' = S U {x+a, 2x}.
            # So y is blocked if:
            # 1. 2y in S' -> 2y in S (already blocked) or 2y == x+a or 2y == 2x.
            # 2. y+a' in S' -> y+a' in S (already blocked) or y+a' == x+b or y+a' == 2x.
            
            # This is getting complex to calculate exactly for all y.
            
            # Alternative: Randomized Greedy.
            # Just pick a random addable element?
            # Or pick the smallest? Largest?
            # Let's try picking the element that is "most central" or has highest degree in some graph?
            
            # Let's stick to the exact minimization but limit the search space.
            # If |addable| > 50, sample 50 candidates.
            
            candidates = addable
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
            
            for x in candidates:
                # Simulate adding x
                new_S = set(S)
                new_S.add(2 * x)
                for a in A:
                    new_S.add(x + a)
                
                # Count how many elements remain addable
                new_addable_count = 0
                # We only care about elements that were addable before?
                # No, elements that were blocked might become unblocked? No, adding elements only adds sums, so it only blocks more.
                # So we only need to check the current `addable` list.
                for y in addable:
                    if y == x:
                        continue
                    # Check if y is still addable with new_S
                    # 1. 2y not in new_S
                    if 2 * y in new_S:
                        continue
                    # 2. y+a not in new_S
                    ok = True
                    for a in A:
                        if y + a in new_S:
                            ok = False
                            break
                    # Also need to check y+x? No, x is not in A yet for the check of y's addability relative to A U {x}?
                    # Wait. If we add x, the set becomes A U {x}.
                    # For y to be addable to A U {x}, it must not collide with sums of A U {x}.
                    # Sums of A U {x} are S U {x+a | a in A} U {2x}.
                    # So we check y against new_S.
                    # But we also need to check if y collides with x itself?
                    # y + x is a sum involving x.
                    # The condition is: y + a' not in S(A U {x}) for all a' in A U {x}.
                    # a' can be x. So y + x must not be in S(A U {x}).
                    # S(A U {x}) includes x+a for a in A.
                    # So we must check if y + x is in new_S.
                    # Note: y + x is a sum of two distinct elements y and x (if y!=x).
                    # Is y+x in new_S?
                    # new_S contains 2x and x+a.
                    # So check if y+x == 2x (implies y=x, skipped) or y+x == x+a (implies y=a, so y in A, but y is in addable so y not in A).
                    # So y+x is NOT in new_S automatically?
                    # Wait. S(A U {x}) = {u+v | u,v in A U {x}, u<=v}.
                    # The sums involving x are {x+x, x+a | a in A}.
                    # The sums not involving x are S(A).
                    # So new_S is correct.
                    # Does y+x appear in new_S?
                    # Only if y+x = x+a => y=a (impossible) or y+x = 2x => y=x (impossible).
                    # So y+x is a NEW sum created by adding y.
                    # The condition for y to be addable is that this new sum y+x does not collide with ANY existing sum in new_S.
                    # So we must check if y+x is in new_S.
                    # My previous check `y+a not in new_S` for `a in A` covers collisions with sums of A.
                    # But we also need to check collision with sums involving x.
                    # Sums involving x are in new_S.
                    # So checking `y+a not in new_S` for `a in A` is NOT enough.
                    # We must check `y+a' not in new_S` for `a' in A U {x}`.
                    # Since x is not in A, we need to check `y+x not in new_S` as well.
                    
                    if y + x in new_S:
                        ok = False
                        break
                        
                    if ok:
                        new_addable_count += 1
                
                if new_addable_count < min_new_addable_count:
                    min_new_addable_count = new_addable_count
                    best_x = x
            
            if best_x is None:
                # Fallback
                best_x = random.choice(candidates)
            
            A.append(best_x)
            S.add(2 * best_x)
            for a in A[:-1]: # A[:-1] is A before adding best_x
                S.add(best_x + a)
        
    return best_set

# EVOLVE-BLOCK-END
