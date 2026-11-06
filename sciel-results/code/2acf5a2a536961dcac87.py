import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Precompute all 4-term APs as bitmasks for fast checking
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We can represent a set as a numpy array of uint64, or a single large integer if we use Python ints (which are arbitrary precision).
    # Given n=512, we can fit the set into 8 uint64s or just use Python's native int which is efficient for bit operations.
    
    # Let's use Python's native int for the set representation.
    # Bit i represents whether i+1 is in the set? Or i represents i?
    # Let's say bit i corresponds to number i+1 (1-indexed).
    # So number 1 is bit 0, number 512 is bit 511.
    
    # Precompute AP masks
    # For each AP, create a mask with 4 bits set.
    # To check if a set S (as int) contains ANY 4-AP:
    # We can't just do (S & ap_mask) == ap_mask for all APs in a loop easily if there are many APs?
    # Number of 4-APs in [1, 512]:
    # a=1..512, d=1..(512-a)//3
    # Sum_{a=1}^{512} floor((512-a)/3) approx 512*170 = 87k APs.
    # Checking 87k APs per individual is too slow for a GA with many generations.
    
    # Alternative fast check:
    # A set S is 4-AP free iff for all a, d: NOT (a in S AND a+d in S AND a+2d in S AND a+3d in S).
    # This is still O(N^2) checks.
    
    # Let's optimize the check.
    # If we represent S as a set of integers, checking a specific AP is fast.
    # But iterating all APs is slow.
    
    # Better approach for validity check in GA:
    # We can precompute for each pair (a, b) with a < b, the required d = b-a.
    # Then check if a+2d and a+3d are in S.
    # This is O(|S|^2). If |S| is around 200, that's 40k operations.
    # 40k operations * 1000 individuals * 100 generations = 4e9 ops. Too slow in Python.
    
    # We need a faster validity check or a smarter search.
    
    # Let's stick to the "additive" logic but make it smarter.
    # Actually, the inspiration said "Randomized greedy with multiple restarts" improved the score.
    # The current program does local search.
    # Let's try a **Simulated Annealing** approach on the set of elements.
    # State: a valid 4-AP-free set.
    # Move: Remove a random element, then try to add random elements (possibly more than 1) to maximize size.
    # Or: Swap an element.
    
    # Wait, the "different strategy" requirement is key.
    # Current: Greedy Seed + Local Search (Add/Swap).
    # New: **Greedy Construction with Lookahead / Backtracking** or **Genetic Algorithm**.
    
    # Let's try a **Genetic Algorithm** but with a **fast validity check** using bitsets.
    # We can use `int` bit operations.
    # To check if S has a 4-AP:
    # We can iterate over possible differences d.
    # For a fixed d, the APs are (a, a+d, a+2d, a+3d).
    # Let S_d be the set of a such that a, a+d, a+2d, a+3d are all in S.
    # We want S_d to be empty.
    
    # Actually, there is a very fast way to check if a set contains a k-AP using convolution or FFT, but for k=4 and n=512, simple loops might be okay if optimized in NumPy.
    
    # Let's use NumPy for the GA.
    # Population size: 50
    # Individual: np.array of bools (size 512)
    
    def is_valid_4ap_free_np(S):
        """
        S: np.array of bools, size 512.
        Returns True if S contains no 4-AP.
        """
        # Convert to list of indices for faster membership? No, NumPy indexing is fast.
        # We can use the property: 
        # For each d, check if there exists a such that S[a] & S[a+d] & S[a+2d] & S[a+3d] is True.
        # We can vectorize over d.
        
        # Max d is (512-1)//3 = 170.
        # For each d in 1..170:
        #   a can range from 0 to 512-3d-1.
        #   Check np.any(S[0:512-3d] & S[d:512-2d] & S[2d:512-d] & S[3d:512])
        
        # This is 170 iterations. Each iteration is a few vectorized ops on arrays of length ~500.
        # 170 * 4 ops * 500 elements = ~340k ops per individual.
        # This is fast enough for a GA.
        
        max_d = (n - 1) // 3
        for d in range(1, max_d + 1):
            # Indices for a, a+d, a+2d, a+3d
            # a ranges from 0 to n - 3d - 1
            end = n - 3 * d
            if end <= 0:
                break
            # S[0:end], S[d:d+end], S[2d:2d+end], S[3d:3d+end]
            if np.any(S[:end] & S[d:d+end] & S[2*d:2*d+end] & S[3*d:3*d+end]):
                return False
        return True

    def greedy_seed_np():
        """
        Fast greedy construction.
        """
        S = np.zeros(n, dtype=bool)
        # Random order for diversity
        order = np.random.permutation(n)
        
        for idx in order:
            x = idx + 1 # 1-indexed value
            i = idx # 0-indexed bit
            
            # Check if adding x creates a 4-AP.
            # We need to check if x is part of any 4-AP with existing elements.
            # This check is expensive if done naively for each x.
            # However, we can just add it and check validity? No, that's O(N) checks per add, O(N^2) total.
            
            # Better: Check specific APs involving x.
            # x can be a, a+d, a+2d, a+3d.
            # We'll just use the fast validity check on the tentative set.
            # To speed up, we can pre-check common cases.
            
            # Actually, for a seed, we can just use the standard greedy logic but vectorized?
            # No, let's just use a simple loop with the fast validity check on the *subset*?
            # No, the fast validity check checks the WHOLE set.
            # If we maintain S, and we add x, we only need to check APs that include x.
            
            # Let's implement a targeted check for adding x to S.
            # x is at index i.
            valid = True
            
            # Case 1: x is a (1st term). a=x.
            # Check if x+d, x+2d, x+3d are in S.
            # d can be 1..(n-x)//3
            # We can check this by iterating d.
            max_d1 = (n - x) // 3
            for d in range(1, max_d1 + 1):
                i1 = i + d
                i2 = i + 2*d
                i3 = i + 3*d
                if i3 < n and S[i1] and S[i2] and S[i3]:
                    valid = False
                    break
            
            if valid:
                # Case 2: x is a+d (2nd term). a = x-d.
                # Check if a, x+d, x+2d are in S.
                max_d2 = (x - 1) // 3
                for d in range(1, max_d2 + 1):
                    ia = i - d
                    i1 = i + d
                    i2 = i + 2*d
                    if ia >= 0 and i2 < n and S[ia] and S[i1] and S[i2]:
                        valid = False
                        break
            
            if valid:
                # Case 3: x is a+2d (3rd term). a = x-2d.
                # Check if a, a+d, x+d are in S.
                max_d3 = (x - 1) // 2
                for d in range(1, max_d3 + 1):
                    ia = i - 2*d
                    i1 = i - d
                    i2 = i + d
                    if ia >= 0 and i2 < n and S[ia] and S[i1] and S[i2]:
                        valid = False
                        break
            
            if valid:
                # Case 4: x is a+3d (4th term). a = x-3d.
                # Check if a, a+d, a+2d are in S.
                max_d4 = (x - 1) // 3
                for d in range(1, max_d4 + 1):
                    ia = i - 3*d
                    i1 = i - 2*d
                    i2 = i - d
                    if ia >= 0 and i2 < n and S[ia] and S[i1] and S[i2]:
                        valid = False
                        break
            
            if valid:
                S[i] = True
        
        return S

    # Genetic Algorithm
    random.seed(42)
    np.random.seed(42)
    
    pop_size = 20
    population = []
    
    # Initialize population with diverse greedy seeds
    for _ in range(pop_size):
        population.append(greedy_seed_np())
    
    best_S = max(population, key=lambda x: np.sum(x))
    best_size = np.sum(best_S)
    
    def crossover(p1, p2):
        # Uniform crossover
        mask = np.random.random(n) < 0.5
        child = np.where(mask, p1, p2)
        return child
    
    def mutate(S, rate=0.05):
        # Randomly flip bits.
        # Flipping a 1 to 0 is always safe (removes elements).
        # Flipping a 0 to 1 might create an AP.
        # We'll flip randomly and then repair.
        new_S = S.copy()
        # Flip some 1s to 0s
        ones = np.where(S)[0]
        if len(ones) > 0:
            num_remove = int(len(ones) * rate * 0.5)
            if num_remove > 0:
                remove_idx = np.random.choice(ones, num_remove, replace=False)
                new_S[remove_idx] = False
        
        # Flip some 0s to 1s
        zeros = np.where(~S)[0]
        if len(zeros) > 0:
            num_add = int(len(zeros) * rate * 0.5)
            if num_add > 0:
                add_idx = np.random.choice(zeros, num_add, replace=False)
                new_S[add_idx] = True
        
        return new_S

    def repair(S):
        # Remove elements to make S valid.
        # Greedy removal: iterate through elements, if removing it fixes a conflict, remove it.
        # Or: while not valid, find an AP, remove one of its elements (randomly or the one with highest "cost").
        
        if is_valid_4ap_free_np(S):
            return S
        
        S = S.copy()
        # We need to find an AP to break.
        # We can reuse the logic from is_valid to find one.
        # Or just randomly remove elements until valid? That's inefficient.
        
        # Let's find one AP.
        max_d = (n - 1) // 3
        for d in range(1, max_d + 1):
            end = n - 3 * d
            if end <= 0:
                break
            mask = S[:end] & S[d:d+end] & S[2*d:2*d+end] & S[3*d:3*d+end]
            if np.any(mask):
                # Found an AP starting at some a
                a_idx = np.argmax(mask)
                a = a_idx
                # Elements: a, a+d, a+2d, a+3d
                # Remove one of them. Let's remove the one that is in the most other APs? 
                # For simplicity, remove the last one (a+3d) or random.
                # Removing a+3d is often good as it's "end" of AP.
                # Let's remove a+3d.
                S[a + 3*d] = False
                # Loop to check again? 
                # This might be slow if we do this one by one.
                # But it's a repair step, hopefully few conflicts.
                # We can break and re-check the whole set in the main loop.
                # Actually, let's just remove one and return. The GA loop will handle the rest?
                # No, we need a valid individual.
                # Let's use a while loop here, but limit iterations.
                break
        
        # Simple iterative repair
        iterations = 0
        while not is_valid_4ap_free_np(S) and iterations < 100:
            # Find an AP
            found = False
            max_d = (n - 1) // 3
            for d in range(1, max_d + 1):
                end = n - 3 * d
                if end <= 0:
                    break
                mask = S[:end] & S[d:d+end] & S[2*d:2*d+end] & S[3*d:3*d+end]
                if np.any(mask):
                    a_idx = np.argmax(mask)
                    a = a_idx
                    # Remove a+3d
                    S[a + 3*d] = False
                    found = True
                    break
            if not found:
                break
            iterations += 1
            
        return S

    generations = 0
    while time.time() < deadline:
        generations += 1
        new_population = []
        
        # Elitism
        best_S = max(population, key=lambda x: np.sum(x))
        new_population.append(best_S.copy())
        
        # Selection and Crossover
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            child = crossover(p1, p2)
            child = mutate(child)
            child = repair(child)
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = max(population, key=lambda x: np.sum(x))
        current_size = np.sum(current_best)
        if current_size > best_size:
            best_size = current_size
            best_S = current_best.copy()

    # Convert back to list of integers (1-indexed)
    result = [i + 1 for i in range(n) if best_S[i]]
    return sorted(result)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
