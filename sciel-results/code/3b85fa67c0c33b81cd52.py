import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    target_min = 128
    target_max = 256
    targets = list(range(target_min, target_max + 1))
    
    # Initial solution: Trivial set
    A = set(range(64, 129))
    
    # Helper to compute counts
    def compute_counts(A_set):
        counts = [0] * (2 * n + 1)
        A_list = list(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 2 * n:
                    if i == j:
                        counts[s] += 1
                    else:
                        counts[s] += 2 # Wait, the task says "Pair representations are unordered and permit equal summands".
                        # Usually "multiplicity" in additive combinatorics for unordered pairs with repetition allowed:
                        # The number of pairs {a,b} with a,b in A, a<=b, a+b=S.
                        # If a!=b, the pair {a,b} is one representation.
                        # If a==b, the pair {a,a} is one representation.
                        # The prompt says: "Minimize the largest representation multiplicity".
                        # "Pair representations are unordered and permit equal summands."
                        # This implies the count is the number of sets {a,b} subset of A such that a+b=S.
                        # So if a!=b, it's 1. If a==b, it's 1.
                        # My previous calculation for trivial set:
                        # S=192. Pairs (64,128), (65,127)...(96,96).
                        # Number of pairs:
                        # a ranges from 64 to 96.
                        # For a=64, b=128. Distinct. Count 1.
                        # For a=96, b=96. Equal. Count 1.
                        # Total 33 pairs.
                        # So `counts[s]` should increment by 1 for each valid unordered pair.
                        pass
        
        # Let's re-implement compute_counts correctly for unordered pairs {a,b} with a<=b.
        counts = [0] * (2 * n + 1)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 2 * n:
                    counts[s] += 1
        return counts

    # Initial counts
    counts = compute_counts(A)
    
    # Check validity
    def is_valid(counts_arr):
        for s in targets:
            if counts_arr[s] == 0:
                return False
        return True

    def get_max_mult(counts_arr):
        m = 0
        for s in targets:
            if counts_arr[s] > m:
                m = counts_arr[s]
        return m

    current_max = get_max_mult(counts)
    
    # If initial is invalid (it shouldn't be), fix it
    # Trivial set is always valid.
    
    # Local Search
    # We will try to reduce the max multiplicity.
    # Moves:
    # 1. Remove x from A.
    # 2. Add x to A.
    # 3. Swap x in A with y not in A.
    
    # To speed up, we maintain counts incrementally.
    # But for simplicity and given the small size, we can recompute or update.
    # Updating counts for remove x:
    # For each y in A (y != x), count[x+y] -= 1.
    # For y=x, count[2x] -= 1.
    # Note: The pair {x,y} is counted once.
    # If we remove x, we lose pairs {x,y} for all y in A (y>=x? No, y in A).
    # Since we count unordered pairs {a,b} with a<=b:
    # If we remove x, we remove:
    # - {x, x} if 2x in target range.
    # - {x, y} for all y in A such that y > x. (Since {y,x} is same as {x,y} and y>x means x<y)
    # - What if y < x? The pair {y,x} was counted when processing y and x in the loop j>=i.
    #   So if we remove x, we must decrement counts for:
    #   1. {x, x} -> sum 2x.
    #   2. {x, y} for all y in A, y != x.
    #   Wait, the loop `for i in range(len): for j in range(i, len)` counts {A[i], A[j]}.
    #   If we remove x, we need to know which pairs involved x.
    #   It's easier to just recompute if |A| is small, but let's try incremental.
    
    def update_counts_remove(A_set, counts_arr, x):
        # Decrement counts for all pairs involving x
        # Pairs are {x, y} for y in A_set.
        # Since the set is unordered, {x, y} is the same as {y, x}.
        # In our counting method (i<=j), {x, y} was counted if x<=y or y<=x.
        # Essentially, for every y in A_set:
        # If y == x: pair {x,x} was counted once. Decrement counts[2x] by 1.
        # If y != x: pair {x,y} was counted once. Decrement counts[x+y] by 1.
        # So, for all y in A_set:
        #   s = x + y
        #   counts_arr[s] -= 1
        
        # Note: We must do this BEFORE removing x from A_set if we iterate over A_set?
        # Yes, iterate over current A_set.
        for y in A_set:
            s = x + y
            if s <= 2 * n:
                counts_arr[s] -= 1
        A_set.remove(x)

    def update_counts_add(A_set, counts_arr, x):
        # Increment counts for all pairs involving x
        # For all y in A_set (current A, before adding x):
        #   Pair {x, y} is new.
        #   s = x + y
        #   counts_arr[s] += 1
        # Also, the pair {x, x} is new.
        #   s = 2x
        #   counts_arr[s] += 1
        
        for y in A_set:
            s = x + y
            if s <= 2 * n:
                counts_arr[s] += 1
        s = 2 * x
        if s <= 2 * n:
            counts_arr[s] += 1
        A_set.add(x)

    # We need to be careful: `update_counts_add` adds x to A at the end.
    # `update_counts_remove` removes x from A at the end.
    
    # Let's verify the logic.
    # Initial state: A, counts.
    # Remove x:
    #   For each y in A (including x):
    #      If y==x, s=2x. counts[2x]-=1.
    #      If y!=x, s=x+y. counts[x+y]-=1.
    #   This correctly removes all pairs {x,y} where y in A.
    #   Then remove x from A.
    # Add x:
    #   For each y in A (current, x not in A):
    #      s=x+y. counts[x+y]+=1.
    #   s=2x. counts[2x]+=1.
    #   Then add x to A.
    #   This correctly adds all pairs {x,y} where y in A (old) plus {x,x}.
    
    # This incremental update is O(|A|).
    
    best_max = current_max
    best_A = set(A)
    best_counts = counts[:]
    
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        # Try to find a move that improves the max multiplicity
        # Or maintains it but reduces "variance" or total count?
        # Let's stick to minimizing max multiplicity.
        
        improved = False
        
        # 1. Try removing elements
        # Prioritize elements that contribute to the max sum
        max_sums = [s for s in targets if counts[s] == best_max]
        
        candidates_to_remove = list(A)
        random.shuffle(candidates_to_remove)
        
        for x in candidates_to_remove:
            if time.time() > deadline: break
            
            # Check if removing x is valid
            # We need to ensure no sum becomes 0.
            # A sum s becomes 0 if counts[s] == 1 and x is involved in the only pair for s.
            # Or counts[s] == 0 (already invalid, but we assume valid state).
            
            # Let's simulate removal
            # We need to check if any target sum s has counts[s] == 1 AND x is part of the pair for s.
            # Actually, if counts[s] == 1, the only pair is {a,b}. If x is a or b, removing x makes it 0.
            # So we must check: for all s in targets with counts[s] == 1, does x participate in the pair?
            
            # To do this efficiently, we can precompute for each x, which sums have count 1 and involve x?
            # Or just simulate:
            # Temporarily decrement counts for pairs involving x.
            # Check if any target sum becomes 0.
            # If yes, revert.
            # If no, check new max.
            
            # Optimization: Only try removing x if x is involved in a sum with count == best_max?
            # Not necessarily, but likely.
            
            # Let's just do the simulation for a limited number of candidates to save time.
            # Given |A| ~ 60, and targets ~ 129, simulation is 60 ops.
            
            # Check validity of removal
            valid = True
            # We need to know if x is the "only" provider for any sum with count 1.
            # Let's just decrement and check.
            
            # Save old counts? No, we can just check conditions.
            # For each s in targets:
            #   if counts[s] == 1:
            #      The pair must be {x, s-x} or {s-x, x} or {x,x}.
            #      So s-x must be in A.
            #      And if s-x == x, then 2x==s.
            #      If s-x != x, then s-x in A and s-x != x.
            #      In all cases, if counts[s]==1, x is involved if and only if (s-x) in A.
            #      Wait, if counts[s]==1, there is exactly one pair {a,b}.
            #      If x is in A, and x is not a or b, then removing x doesn't affect s.
            #      If x is a or b, removing x makes count 0.
            #      So: if counts[s] == 1 and (s-x) in A, then removing x breaks s.
            #      (Note: if s-x == x, then x in A is true, so condition holds).
            
            for s in targets:
                if counts[s] == 1:
                    other = s - x
                    if other in A:
                        # x is part of the unique pair for s
                        valid = False
                        break
            
            if not valid:
                continue
            
            # If valid, apply removal
            update_counts_remove(A, counts, x)
            new_max = get_max_mult(counts)
            
            if new_max < best_max:
                best_max = new_max
                best_A = set(A)
                best_counts = counts[:]
                improved = True
                no_improvement_count = 0
                # Accept the move, continue search from here
            else:
                # Revert
                update_counts_add(A, counts, x)
        
        if not improved:
            # Try adding elements?
            # Adding an element increases counts, so max can only increase or stay same.
            # It might allow future removals.
            # Let's try swapping.
            
            # Try swapping x in A with y not in A
            # This is complex. Let's stick to removals first.
            # If stuck, we might need to add a "helper" element to break a bottleneck.
            
            # For now, if no removal improves, we might be at a local optimum.
            # Let's try a random perturbation: Add a random element, then try to remove others.
            
            if random.random() < 0.1: # 10% chance to perturb
                # Pick a random element not in A
                possible_y = [i for i in range(n+1) if i not in A]
                if possible_y:
                    y = random.choice(possible_y)
                    # Add y
                    update_counts_add(A, counts, y)
                    # This might increase max, but we hope to fix it.
                    # We don't update best_max here, just continue loop.
                    # The next iterations will try to remove elements to fix the max.
            
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Reset to best known
                A = set(best_A)
                counts = best_counts[:]
                best_max = get_max_mult(counts)
                no_improvement_count = 0
                max_no_improvement *= 2 # Increase patience
        
        if best_max == 1:
            break
        
    # Return the best A found
    return list(best_A)
