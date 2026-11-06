import time
import random

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
    sum_set = set()
    prod_set = set()
    for a in A:
        for b in A:
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Generate diverse structural seeds
    seeds = []
    
    # Seed 1: Arithmetic Progression 1..20
    seeds.append(list(range(1, 21)))
    
    # Seed 2: Powers of 2 (14 elements) + small multiples to reach 20
    # {1, 2, ..., 8192} + {3, 6, 12, 24, 48, 96}
    p2 = [2**i for i in range(14)]
    ext = [3 * (2**i) for i in range(6)]
    seeds.append(p2 + ext)
    
    # Seed 3: 3-smooth numbers <= 100
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96
    smooth3 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96]
    if len(set(smooth3)) == 20:
        seeds.append(smooth3)
    
    # Seed 4: 5-smooth numbers <= 100 (Hamming numbers)
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 27, 30, 32, 36, 40
    smooth5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 25, 27, 30, 32, 36, 40]
    if len(set(smooth5)) == 20:
        seeds.append(smooth5)
        
    # Seed 5: Geometric with ratio 1.5? No, integers.
    # Try a mix of small AP and large GP
    seeds.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    
    best_A = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in seeds:
        if len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
        score = compute_metrics(seed)
        if score < best_score:
            best_score = score
            best_A = seed[:]
            
    A = best_A[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Candidate pool for swaps
    # Focus on small numbers and smooth numbers to encourage structure
    candidate_pool = list(range(1, 101))
    # Add some random large numbers
    candidate_pool += [random.randint(100, 10000) for _ in range(50)]
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Generate a few candidates
        # 1. From pool
        c1 = random.choice(candidate_pool)
        # 2. Neighbor
        c2 = old_val + random.choice([-2, -1, 1, 2])
        # 3. Multiple/Divisor
        c3 = old_val * 2 if old_val * 2 <= 10000 else old_val // 2 if old_val > 1 else 1
        
        candidates = [c1, c2, c3]
        
        A_set = set(A)
        A_set.discard(old_val)
        
        best_cand = None
        best_cand_score = current_score
        
        for c in candidates:
            if c < 1 or c > 10000:
                continue
            if c in A_set:
                continue
            if c == old_val:
                continue
                
            A[idx] = c
            new_score = compute_metrics(A)
            
            if new_score < best_cand_score:
                best_cand_score = new_score
                best_cand = c
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta > 0:
                    prob = random.random() < pow(2.718281828, -delta / T)
                    if prob and (best_cand is None or new_score < best_cand_score):
                         # Only accept if it's the best among tested so far, or just accept?
                         # Standard SA accepts based on current state.
                         # We are doing "best of batch" logic mixed with SA.
                         # Let's stick to "best improvement" for stability, but allow worse if T is high?
                         # Actually, simple hill climbing is often better for this rugged landscape.
                         # Let's try "first improvement" or "best improvement".
                         pass
                         
        # Re-evaluate logic: 
        # If we found a strictly better candidate, take it.
        # If not, we might take a worse one with probability exp(-delta/T).
        # But we need to pick ONE candidate to swap to.
        
        # Let's simplify: Pick the best candidate from the batch.
        # If it's better, swap.
        # If it's worse, swap with probability exp(-delta/T).
        
        # We need to track the best candidate found in this step.
        # The loop above did that.
        
        if best_cand is not None:
            # We found a strictly better candidate
            A[idx] = best_cand
            current_score = best_cand_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            # No strictly better candidate.
            # Check if we should accept a worse one (SA)
            # We need to re-evaluate the candidates to find the "least bad" or pick one randomly.
            # For simplicity and speed, let's just revert if no improvement.
            # This is Hill Climbing. To escape local minima, we can occasionally do a random swap.
            
            # Random restart probability
            if random.random() < 0.001:
                # Perturb
                for _ in range(5):
                    i = random.randint(0, 19)
                    A[i] = random.choice(candidate_pool)
                # Ensure validity
                A_set = set(A)
                if len(A_set) != 20:
                    # Fix duplicates
                    A = list(A_set)
                    while len(A) < 20:
                        A.append(random.randint(1, 10000))
                    A = A[:20]
                # Re-evaluate
                current_score = compute_metrics(A)
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
            else:
                A[idx] = old_val
                
        T *= cooling_rate

    return best_A
