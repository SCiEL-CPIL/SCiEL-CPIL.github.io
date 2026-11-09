import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin for the 90s kill
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    N_RANGE = 129  # 0 to 128 inclusive
    
    # Precompute all possible pair sums (a, b) with a <= b
    # We represent the set A as a boolean array of size 129
    # The sum convolution can be computed via FFT for speed
    # But for local search, incremental updates are better.
    
    # Let's use a boolean array for A and an array for counts of how many times
    # each value in [0, 128] is "used" in a way that allows fast recalculation?
    # No, the interaction is global (pairs).
    # However, if we change one element, the change in sums is local:
    # Adding x adds: for all a in A, a+x (once if x not in A, plus x+x)
    # Removing x removes: for all a in A, a+x
    
    # Strategy:
    # 1. Initialize a valid set (e.g., all even numbers).
    # 2. Compute reps.
    # 3. Iteratively try to remove elements that contribute to the peak,
    #    and add elements that help cover low-count sums, balancing the peak.
    #    Specifically, use a "simulated annealing" or "hill climbing" with a specific objective.
    
    # Objective: Minimize Max Reps, subject to Min Reps >= 1.
    # We can define a cost function: Cost = MaxReps * 1000 + UncoveredCount.
    # This forces coverage first, then minimizes peak.
    
    def compute_reps(A_bool):
        """Compute representation counts for all sums 0..256 using FFT."""
        # Convolution of A with A
        # A is a vector of 0s and 1s of length 129
        # Convolution length is 257
        # Note: FFT convolution computes ordered pairs (a,b) and (b,a) as 2 if a!=b, and 1 if a==b.
        # We need unordered pairs:
        # If a < b, count is 1 in our definition. FFT gives 2 (from a+b and b+a).
        # If a == b, count is 1 in our definition. FFT gives 1.
        # So: 
        #   If s is sum of a!=b, FFT_count = 2 * Unordered_Count
        #   If s is sum of a==b, FFT_count = 1 * Unordered_Count
        # We can recover Unordered_Count by:
        #   Let FFT_s be the value at sum s.
        #   We know that sum of a==b is 2a=s, so s must be even, and a=s/2.
        #   If A[s/2] is 1, then Unordered_Count = (FFT_s - 1) / 2 + 1 = (FFT_s + 1) / 2
        #   Else Unordered_Count = FFT_s / 2
        # Actually, it's easier to just do:
        #   Unordered = floor(FFT_s / 2) + (1 if A[s/2]==1 and s%2==0 else 0)
        # Let's verify:
        #   Case a=b: FFT has 1 at 2a. Floor(1/2)=0. A[a]=1, s=2a even. Result 0+1=1. Correct.
        #   Case a!=b: FFT has 2 at a+b (one from a,b one from b,a). Actually, convolution is sum_{i+j=k} A[i]A[j].
        #   If a != b, terms (a,b) and (b,a) are distinct in convolution if a!=b.
        #   So FFT_k = 2 * (# of pairs {a,b} with a<b, a+b=k) + 1 * (# of pairs {a,a} with 2a=k)
        #   Wait, if a=b, the term is A[a]*A[a]=1. It appears only once in the sum for k=2a?
        #   Yes, i=j=a is the only term.
        #   So FFT_k = 2 * N_distinct + 1 * N_same.
        #   We want N_total = N_distinct + N_same.
        #   N_distinct = (FFT_k - N_same) / 2.
        #   N_total = (FFT_k - N_same)/2 + N_same = (FFT_k + N_same) / 2.
        #   N_same is 1 if A[k/2]==1 and k is even, else 0.
        
        # For simplicity and speed in local search, FFT might be overkill if we do incremental updates.
        # But for the initial calculation and every "restart", FFT is good.
        # Let's use FFT for global re-evaluation and incremental for moves.
        
        # Actually, implementing incremental updates with the unordered logic is tricky.
        # Let's stick to a simpler, robust approach:
        # Use a bitset or boolean array.
        # When we add/remove an element x:
        #   We need to update reps for all sums involving x.
        #   If adding x:
        #      For each a in A (existing):
        #          s = a + x
        #          if a == x: reps[s] += 1 (pair x,x)
        #          else: reps[s] += 1 (pair a,x) ... wait.
        #          If a is in A, and x is being added:
        #          We are adding pairs (a, x) and (x, a) in ordered, or just {a,x} in unordered.
        #          In our count, each unordered pair contributes 1.
        #          So if we add x:
        #             For each a in A:
        #                if a == x: not possible if x not in A yet.
        #                s = a + x. 
        #                This creates a new pair {a, x}. So reps[s] += 1.
        #             Also, we create pair {x, x}. So reps[2x] += 1.
        #          Total additions: |A| (for pairs with existing) + 1 (for x,x).
        #          BUT, we must ensure we don't double count or miss.
        #          The set A becomes A ∪ {x}.
        #          New pairs are {a, x} for a in A, and {x, x}.
        #          So yes, reps[a+x] += 1 for all a in A, and reps[2x] += 1.
        #   If removing x:
        #      Reverse the above.
        #      reps[a+x] -= 1 for all a in A\{x}.
        #      reps[2x] -= 1.
        
        # This incremental update is O(|A|) which is very fast (~40 ops).
        # This allows us to do many steps.
        
        pass # Placeholder, will implement in the main loop logic directly for speed.

    # Initialize A
    # Start with all even numbers?
    # Even numbers: 0, 2, ..., 128. Size 65.
    # Sums are even. 128..256.
    # 128 (even), 130 (even)... 256 (even).
    # Odd numbers in [128, 256] are NOT covered.
    # So all evens is INVALID.
    
    # We need a mix.
    # Let's try to construct a valid set first using a greedy algorithm.
    
    def is_valid_reps(reps):
        return np.all(reps[N_LOW:N_HIGH+1] > 0)
    
    def get_peak(reps):
        return int(np.max(reps[N_LOW:N_HIGH+1]))
    
    # Greedy Construction
    # 1. Start with empty set.
    # 2. Add elements that cover the most "uncovered" sums?
    #    Or simply: 
    #    - To cover 128, we need a+b=128.
    #    - To cover 256, we need a+b=256 -> 128+128. So 128 MUST be in A.
    #    Let's force 128.
    
    A = set([128])
    reps = np.zeros(257, dtype=np.int32)
    # Initial state: A={128}
    reps[256] = 1
    
    # We need to cover 128..255.
    # Let's use a greedy approach to add elements.
    # At each step, pick the element v in [0,128]\A that minimizes the resulting Peak,
    # or covers the most critical gaps.
    # To ensure validity, we can use a "repair" step.
    
    # Better Initialization:
    # Use a random set of size ~40, then repair.
    
    random.seed(42)
    
    # Let's try a specific structured set that is known to have low additive energy or similar.
    # But let's stick to optimization.
    
    best_A = None
    best_peak = float('inf')
    best_reps = None
    
    # Phase 1: Random Restart + Local Search (Fast Incremental)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize A
        # Size estimation: 
        # To cover 129 sums with peak P, we need at least 129 pairs.
        # Pairs ~ K^2/2.
        # If P=4, we can have up to 4*129 = 516 pairs.
        # K^2/2 ~ 516 -> K^2 ~ 1032 -> K ~ 32.
        # So K around 30-40 is a good range.
        
        init_size = 30 + random.randint(0, 10)
        A_list = random.sample(range(0, 129), init_size)
        A = set(A_list)
        A.add(128) # Force 128 for 256
        
        # Compute initial reps
        reps = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                reps[a+A_list[j]] += 1
        
        # Repair: Ensure coverage
        # If not covered, add random elements until covered or limit reached.
        # This is slow. Instead, let's use a local search that accepts invalid states with high penalty.
        
        curr_reps = reps.copy()
        curr_peak = get_peak(curr_reps)
        curr_uncovered = np.sum(curr_reps[N_LOW:N_HIGH+1] == 0)
        curr_cost = curr_peak * 1000 + curr_uncovered * 10000
        
        # Local Search Loop
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        max_steps = 20000
        
        while time.time() < deadline - 5 and T > min_T and steps < max_steps:
            steps += 1
            
            # Choose move
            # 1. Remove an element
            # 2. Add an element
            # 3. Swap an element
            
            move_type = random.random()
            
            if len(A) == 0:
                v = random.randint(0, 128)
                A.add(v)
                # Update reps
                # Add v: pairs with all a in A (now including v? No, update based on old A + v)
                # Actually, my incremental logic above assumed adding to existing A.
                # Here A was empty.
                # Pairs: {v,v} -> 2v += 1.
                curr_reps[2*v] += 1
            else:
                # Pick u to potentially remove/swap
                u = random.choice(list(A))
                
                if move_type < 0.4:
                    # Remove u
                    # Reverse: 
                    # For each a in A\{u}: curr_reps[a+u] -= 1
                    # curr_reps[2u] -= 1
                    A_remove = set(A)
                    A_remove.remove(u)
                    
                    delta_reps = np.zeros(257, dtype=np.int32)
                    for a in A_remove:
                        delta_reps[a+u] -= 1
                    delta_reps[2*u] -= 1
                    
                    new_reps = curr_reps + delta_reps
                    
                    # Check validity/cost
                    new_uncovered = np.sum(new_reps[N_LOW:N_HIGH+1] == 0)
                    new_peak = int(np.max(new_reps[N_LOW:N_HIGH+1])) if np.any(new_reps[N_LOW:N_HIGH+1] > 0) else 0
                    new_cost = new_peak * 1000 + new_uncovered * 10000
                    
                    if new_cost < curr_cost or random.random() < np.exp(-(new_cost - curr_cost) / T):
                        curr_reps = new_reps
                        A = A_remove
                        curr_cost = new_cost
                        curr_peak = new_peak
                        curr_uncovered = new_uncovered
                elif move_type < 0.8:
                    # Add v
                    # v not in A
                    v = random.randint(0, 128)
                    if v in A:
                        continue # or handle as no-op
                    
                    delta_reps = np.zeros(257, dtype=np.int32)
                    # Pairs {a, v} for a in A
                    for a in A:
                        delta_reps[a+v] += 1
                    # Pair {v, v}
                    delta_reps[2*v] += 1
                    
                    new_reps = curr_reps + delta_reps
                    
                    new_uncovered = np.sum(new_reps[N_LOW:N_HIGH+1] == 0)
                    new_peak = int(np.max(new_reps[N_LOW:N_HIGH+1])) if np.any(new_reps[N_LOW:N_HIGH+1] > 0) else 0
                    new_cost = new_peak * 1000 + new_uncovered * 10000
                    
                    if new_cost < curr_cost or random.random() < np.exp(-(new_cost - curr_cost) / T):
                        curr_reps = new_reps
                        A.add(v)
                        curr_cost = new_cost
                        curr_peak = new_peak
                        curr_uncovered = new_uncovered
                else:
                    # Swap u -> v
                    # Remove u, Add v
                    # Net delta:
                    # - Pairs involving u (except u,u handled separately in remove/add)
                    # Let's just compute delta for Remove u + Add v.
                    
                    A_without_u = set(A)
                    A_without_u.remove(u)
                    
                    delta_reps = np.zeros(257, dtype=np.int32)
                    
                    # Part 1: Remove u
                    # For a in A_without_u: delta[a+u] -= 1
                    # delta[2u] -= 1
                    for a in A_without_u:
                        delta_reps[a+u] -= 1
                    delta_reps[2*u] -= 1
                    
                    # Part 2: Add v to (A_without_u)
                    # Wait, the new set is A_without_u ∪ {v}
                    # Pairs added: {a, v} for a in A_without_u, and {v, v}
                    for a in A_without_u:
                        delta_reps[a+v] += 1
                    delta_reps[2*v] += 1
                    
                    # Note: If u == v, delta is 0. We avoid this.
                    if u == v:
                        continue
                    
                    # Also, we must consider the pair {u, v} if it existed?
                    # In the "Remove u" step, we subtracted pairs {a, u} for a in A_without_u.
                    # {u, v} is NOT in A_without_u ∪ {u} if v is not in A.
                    # So {u, v} was not present.
                    # In "Add v", we add pairs {a, v} for a in A_without_u.
                    # Does this include {u, v}? No, u is not in A_without_u.
                    # So {u, v} is correctly not added/removed.
                    
                    new_reps = curr_reps + delta_reps
                    
                    # Ensure no negative reps (shouldn't happen if logic correct, but check)
                    if np.any(new_reps < 0):
                        continue # Invalid move logic, skip
                    
                    new_uncovered = np.sum(new_reps[N_LOW:N_HIGH+1] == 0)
                    new_peak = int(np.max(new_reps[N_LOW:N_HIGH+1])) if np.any(new_reps[N_LOW:N_HIGH+1] > 0) else 0
                    new_cost = new_peak * 1000 + new_uncovered * 10000
                    
                    if new_cost < curr_cost or random.random() < np.exp(-(new_cost - curr_cost) / T):
                        curr_reps = new_reps
                        A.remove(u)
                        A.add(v)
                        curr_cost = new_cost
                        curr_peak = new_peak
                        curr_uncovered = new_uncovered
                        
            T *= cooling
            
            # Update global best
            if curr_uncovered == 0:
                if curr_peak < best_peak:
                    best_peak = curr_peak
                    best_A = set(A)
                    best_reps = curr_reps.copy()
            else:
                # Even if invalid, if cost is lower, keep it as current, but don't update best_A
                # unless we want to track best invalid? No, we need valid.
                pass

    # Phase 2: Polish the best valid set
    if best_A is not None:
        # Try to reduce peak further with a more aggressive local search
        # specifically targeting the peak sums.
        A = set(best_A)
        reps = best_reps.copy()
        curr_peak = best_peak
        
        # For each sum in the target range with count == curr_peak, try to "break" the pair.
        # This means removing one of the elements in the pair.
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Find peak sums
            peak_sums = np.where(reps[N_LOW:N_HIGH+1] == curr_peak)[0] + N_LOW
            
            for s in peak_sums:
                if time.time() > deadline - 2:
                    break
                
                # Find all pairs (a,b) in A such that a+b=s
                pairs = []
                A_list = list(A)
                for i in range(len(A_list)):
                    a = A_list[i]
                    b = s - a
                    if b in A and a <= b:
                        pairs.append((a, b))
                
                # Try removing one element from each pair
                # If we remove 'a', does coverage break?
                # We can just try removing 'a' and checking validity.
                
                for a, b in pairs:
                    # Try removing a
                    # We need to check if any sum becomes 0.
                    # This is O(|A|) per check.
                    
                    # Instead of full check, we can estimate.
                    # Let's just do a full local search step: Remove a, see if valid.
                    
                    # To speed up, we can precompute "criticality" of each element.
                    # But for now, let's just try random swaps from the current set.
                    pass
                
                # Fallback: Random swaps
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    u = random.choice(list(A))
                    v = random.randint(0, 128)
                    if v in A or v == u:
                        continue
                    
                    # Swap u -> v
                    A_without_u = set(A)
                    A_without_u.remove(u)
                    
                    delta_reps = np.zeros(257, dtype=np.int32)
                    for a in A_without_u:
                        delta_reps[a+u] -= 1
                    delta_reps[2*u] -= 1
                    
                    for a in A_without_u:
                        delta_reps[a+v] += 1
                    delta_reps[2*v] += 1
                    
                    new_reps = reps + delta_reps
                    if np.any(new_reps < 0):
                        continue
                    
                    if np.all(new_reps[N_LOW:N_HIGH+1] > 0):
                        new_peak = int(np.max(new_reps[N_LOW:N_HIGH+1]))
                        if new_peak < curr_peak:
                            reps = new_reps
                            A.remove(u)
                            A.add(v)
                            curr_peak = new_peak
                            best_peak = curr_peak
                            best_A = set(A)
                            best_reps = reps.copy()
                            improved = True
                            break

    if best_A is None:
        # Fallback: Return a set that is likely valid even if not optimal
        # e.g., all numbers?
        return list(range(0, 129))
        
    return list(best_A)

