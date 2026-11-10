import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Generate a diverse pool of candidate numbers
    # We focus on smooth numbers because they have many small factors,
    # which tends to keep the product set compact.
    # We also include small integers to keep the sum set compact.
    
    candidates = set()
    
    # Add small integers 1..100
    for i in range(1, 101):
        candidates.add(i)
        
    # Add 2-smooth numbers (2^a)
    for a in range(0, 14):
        candidates.add(2**a)
        
    # Add 3-smooth numbers (2^a * 3^b)
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2**a) * (3**b)
            if val <= 10000:
                candidates.add(val)
                
    # Add 5-smooth numbers (2^a * 3^b * 5^c)
    for a in range(0, 14):
        for b in range(0, 14):
            for c in range(0, 14):
                val = (2**a) * (3**b) * (5**c)
                if val <= 10000:
                    candidates.add(val)
                    
    # Add some random numbers to ensure diversity
    for _ in range(500):
        candidates.add(random.randint(1, 10000))
        
    candidates = sorted(list(candidates))
    
    # 2. Initialize multiple starting sets
    # Strategy A: Greedy construction using smooth numbers
    # Strategy B: Random sampling from smooth numbers
    # Strategy C: Arithmetic progression (baseline)
    
    starting_sets = []
    
    # Greedy construction
    A_greedy = []
    for _ in range(20):
        best_c = None
        best_s = float('inf')
        # Sample a subset of candidates to speed up greedy init
        sample = random.sample(candidates, min(500, len(candidates)))
        for c in sample:
            if c in A_greedy:
                continue
            new_A = A_greedy + [c]
            s = calc_score(new_A)
            if s < best_s:
                best_s = s
                best_c = c
        if best_c is not None:
            A_greedy.append(best_c)
        else:
            # Fallback if no candidate found in sample
            for c in candidates:
                if c not in A_greedy:
                    A_greedy.append(c)
                    break
    if len(A_greedy) == 20:
        starting_sets.append(A_greedy)
        
    # Random smooth sets
    smooth_pool = [c for c in candidates if c <= 2000] # Restrict to smaller smooths for better start
    for _ in range(5):
        A_rand = random.sample(smooth_pool, 20)
        starting_sets.append(A_rand)
        
    # Arithmetic progression
    starting_sets.append(list(range(1, 21)))
    
    # 3. Simulated Annealing
    best_set = starting_sets[0]
    best_score = calc_score(best_set)
    
    current_set = list(best_set)
    current_score = best_score
    
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will run SA on each starting set, or just one if time is tight.
    # Given the 90s limit, we'll run SA on the best starting set first,
    # and if time permits, restart from other seeds.
    
    # Sort starting sets by initial score to prioritize good ones
    starting_sets.sort(key=lambda x: calc_score(x))
    
    for seed_idx, seed_set in enumerate(starting_sets):
        if time.time() > deadline:
            break
            
        current_set = list(seed_set)
        current_score = calc_score(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = list(current_set)
            
        T = 50.0 + random.uniform(0, 20) # Reset temperature for new seed
        
        while T > T_min:
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            # Mutation types:
            # 1. Swap one element with another from the pool
            # 2. Replace one element with a random one from the pool
            # 3. Local search: try to swap with a neighbor in value space
            
            mutation_type = random.random()
            
            new_set = list(current_set)
            
            if mutation_type < 0.5:
                # Swap with a candidate from the pool
                idx = random.randint(0, 19)
                old_val = new_set[idx]
                
                # Pick a new value. To avoid huge jumps, pick from a window
                # or from the smooth pool.
                # Let's pick from the global candidate pool, but weighted towards
                # values close to the current one or other smooth numbers.
                # For simplicity, random from pool, but maybe restrict range?
                # No, global pool is better for exploration.
                new_val = random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                new_set[idx] = new_val
                
                # Ensure distinctness
                if len(set(new_set)) < 20:
                    continue
                    
            elif mutation_type < 0.8:
                # Local swap: swap two elements in the set with two others
                # Or just replace one with a nearby integer
                idx = random.randint(0, 19)
                old_val = new_set[idx]
                
                # Try a nearby integer
                delta = random.choice([-2, -1, 1, 2, -3, 3, -5, 5])
                new_val = old_val + delta
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in new_set:
                    continue
                    
                new_set[idx] = new_val
                
            else:
                # Swap two elements in the set with each other? No, that doesn't change the set.
                # Swap one element with another element in the set? No.
                # Try a "2-opt" like move: remove two, add two.
                idx1, idx2 = random.sample(range(20), 2)
                old1, old2 = new_set[idx1], new_set[idx2]
                
                new1 = random.choice(candidates)
                new2 = random.choice(candidates)
                
                if new1 == new2:
                    continue
                if new1 == old1 or new1 == old2 or new2 == old1 or new2 == old2:
                    continue
                    
                new_set[idx1] = new1
                new_set[idx2] = new2
                
                if len(set(new_set)) < 20:
                    continue
            
            # Calculate new score
            # Optimization: Only calc if valid
            if len(set(new_set)) != 20:
                continue
                
            new_score = calc_score(new_set)
            
            # Acceptance criterion
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Metropolis criterion
                delta = new_score - current_score
                if delta == 0:
                    # Accept ties to explore flat regions
                    current_set = new_set
                    # Don't update best_score
                else:
                    # delta > 0
                    if random.random() < math.exp(-delta / T):
                        current_set = new_set
                        current_score = new_score
            
            T *= cooling_rate
            
    # 4. Final Local Search Polish
    # Try to improve the best_set found by SA with a few rounds of greedy swaps
    
    if time.time() < deadline - 5:
        A_polish = list(best_set)
        score_polish = best_score
        
        # Try swapping each element with every candidate in a small window
        # This is expensive, so we limit it.
        # Instead, do a few random restarts of local search from best_set
        
        for _ in range(10): # 10 restarts
            if time.time() > deadline - 2:
                break
                
            A_local = list(best_set)
            s_local = score_polish
            
            # Local search: try to swap one element with another to reduce score
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(20):
                    if time.time() > deadline - 2:
                        break
                    old_val = A_local[i]
                    # Try swapping with a few random candidates
                    for _ in range(10):
                        new_val = random.choice(candidates)
                        if new_val == old_val or new_val in A_local:
                            continue
                        A_trial = list(A_local)
                        A_trial[i] = new_val
                        
                        s_trial = calc_score(A_trial)
                        if s_trial < s_local:
                            A_local = A_trial
                            s_local = s_trial
                            improved = True
                            break # Restart inner loop or continue? Continue to find more improvements
                    
                    if improved:
                        break # Restart outer loop to try other indices with new set
                        
            if s_local < best_score:
                best_score = s_local
                best_set = A_local
                
    # Ensure valid output
    best_set = sorted(list(set(best_set)))
    if len(best_set) < 20:
        # Pad if necessary (should not happen with SA)
        for i in range(1, 100):
            if i not in best_set:
                best_set.append(i)
            if len(best_set) == 20:
                break
        best_set = best_set[:20]
    elif len(best_set) > 20:
        best_set = best_set[:20]
        
    return best_set

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    end = time.time()
    score = calc_score(res)
    print(f"Time: {end-start:.2f}s")
    print(f"Score: {score}")
    print(f"Set: {res}")
