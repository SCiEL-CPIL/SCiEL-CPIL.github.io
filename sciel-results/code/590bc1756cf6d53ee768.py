import time
import random
import sys
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    sum_set = set()
    prod_set = set()
    n = len(A)
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sum_set.add(a + b)
            prod_set.add(a * b)
    return max(len(sum_set), len(prod_set))

def compute_metrics_fast(A):
    """Faster metric computation using local variables and direct set ops."""
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    ls = len(sums)
    lp = len(prods)
    return ls if ls >= lp else lp

def smooth_numbers(limit, max_exponent=6):
    """Generate smooth numbers (numbers with only small prime factors) up to limit."""
    primes = [2, 3, 5, 7, 11, 13]
    smooths = set()
    
    def gen(idx, current):
        if current > limit:
            return
        smooths.add(current)
        if idx >= len(primes):
            return
        p = primes[idx]
        while current * p <= limit:
            current *= p
            gen(idx, current)
            if current > limit:
                break
        gen(idx + 1, current)
    
    gen(0, 1)
    return sorted([s for s in smooths if 1 <= s <= limit])

def generate_structured_candidates():
    """Generate diverse structured initial candidates."""
    candidates = []
    
    # 1. Arithmetic progression
    ap = list(range(1, 21))
    candidates.append(ap)
    
    # 2. Smooth numbers (many factorizations -> product collisions)
    smooths = smooth_numbers(10000)
    if len(smooths) >= 20:
        # Take first 20 smooth numbers
        candidates.append(smooths[:20])
        # Also try a mix of small smooth numbers
        small_smooths = [s for s in smooths if s <= 500]
        if len(small_smooths) >= 20:
            candidates.append(small_smooths[:20])
        # Try 2,3-smooth only
        s23 = set()
        v = 1
        while v <= 10000:
            w = v
            while w <= 10000:
                s23.add(w)
                w *= 3
            v *= 2
        s23 = sorted(s23)
        if len(s23) >= 20:
            candidates.append(s23[:20])
            # Try a subset with more spread
            if len(s23) >= 40:
                candidates.append(s23[::2][:20])
    
    # 3. Geometric progression (limited by 10000)
    gp = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    while len(gp) < 20:
        new_val = random.randint(4097, 10000)
        if new_val not in gp:
            gp.append(new_val)
    candidates.append(gp)
    
    # 4. Mix of small consecutive and smooth numbers
    mixed = list(range(1, 11))
    smooths_large = [s for s in smooth_numbers(10000) if s > 10]
    random.shuffle(smooths_large)
    for s in smooths_large:
        if len(mixed) >= 20:
            break
        if s not in mixed:
            mixed.append(s)
    candidates.append(mixed)
    
    # 5. Random set
    random_set = random.sample(range(1, 10001), 20)
    candidates.append(random_set)
    
    # 6. Powers of 2 with some variations
    powers2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    extra = random.sample(range(513, 10001), 10)
    candidates.append(powers2 + extra)
    
    # 7. Consecutive even numbers
    even_ap = list(range(2, 41, 2))
    candidates.append(even_ap)
    
    # 8. Numbers with many divisors (highly composite-like)
    highly_composite = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 48, 60]
    candidates.append(highly_composite)
    
    # 9. 2,3,5-smooth numbers (more collisions)
    s235 = set()
    v = 1
    while v <= 10000:
        w = v
        while w <= 10000:
            u = w
            while u <= 10000:
                s235.add(u)
                u *= 5
            w *= 3
        v *= 2
    s235 = sorted(s235)
    if len(s235) >= 20:
        candidates.append(s235[:20])
    
    return candidates

def simulated_annealing(A, start_temp, end_temp, start_time, deadline, best_A, best_score):
    """Run simulated annealing from initial state A."""
    current_A = A[:]
    current_score = compute_metrics_fast(current_A)
    
    if current_score < best_score:
        best_score = current_score
        best_A = current_A[:]
    
    n_steps = 0
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute smooth numbers for guided moves
    smooth_pool = smooth_numbers(10000)
    
    while time.time() < deadline:
        # Temperature schedule
        elapsed = time.time() - start_time
        total_time = max(1, deadline - start_time)
        progress = min(1.0, elapsed / total_time)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Pick random index
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        # Generate new value: mix of local, guided, and global moves
        r = random.random()
        if r < 0.35:
            # Local move: small perturbation
            delta = random.randint(-15, 15)
            new_val = old_val + delta
        elif r < 0.60:
            # Medium move: moderate change
            delta = random.randint(-100, 100)
            new_val = old_val + delta
        elif r < 0.80:
            # Guided move: replace with a smooth number
            if len(smooth_pool) > 0:
                new_val = smooth_pool[random.randint(0, len(smooth_pool)-1)]
            else:
                new_val = random.randint(1, 10000)
        else:
            # Global move: random anywhere
            new_val = random.randint(1, 10000)
        
        # Validate
        if new_val < 1 or new_val > 10000:
            continue
        if new_val in current_A and new_val != old_val:
            continue
        
        # Apply move
        current_A[i] = new_val
        new_score = compute_metrics_fast(current_A)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Simulated annealing acceptance
            delta_score = new_score - current_score
            if temp > 0 and random.random() < min(1.0, (-delta_score) / temp):
                current_score = new_score
                no_improve += 1
            else:
                current_A[i] = old_val
                no_improve += 1
        
        n_steps += 1
        
        # Restart if stuck
        if no_improve > max_no_improve:
            # Reset to best and reheat
            current_A = best_A[:]
            current_score = best_score
            no_improve = 0
            temp = start_temp * 0.5
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Generate structured candidates
    candidates = generate_structured_candidates()
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 20:
            continue
        score = compute_metrics_fast(cand)
        if score < best_score:
            best_score = score
            best_A = cand[:]
    
    if best_A is None:
        return list(range(1, 21))
    
    # Run simulated annealing from multiple starting points
    start_temp = 50.0
    end_temp = 0.01
    
    # SA from best candidate
    best_A, best_score = simulated_annealing(
        best_A, start_temp, end_temp, start_time, deadline, best_A, best_score
    )
    
    # If time remains, run SA from other good candidates
    if time.time() < deadline - 8.0:
        # Find second-best candidate
        second_best = None
        second_score = float('inf')
        for cand in candidates:
            if len(cand) != 20:
                continue
            if cand == best_A:
                continue
            score = compute_metrics_fast(cand)
            if score < second_score:
                second_score = score
                second_best = cand
        
        if second_best is not None:
            best_A, best_score = simulated_annealing(
                second_best, start_temp * 0.7, end_temp, start_time, deadline, best_A, best_score
            )
    
    # If time remains, run SA from third candidate
    if time.time() < deadline - 8.0:
        third_best = None
        third_score = float('inf')
        for cand in candidates:
            if len(cand) != 20:
                continue
            if cand == best_A or cand == second_best:
                continue
            score = compute_metrics_fast(cand)
            if score < third_score:
                third_score = score
                third_best = cand
        
        if third_best is not None:
            best_A, best_score = simulated_annealing(
                third_best, start_temp * 0.5, end_temp, start_time, deadline, best_A, best_score
            )
    
    # Final local search refinement
    if time.time() < deadline - 3.0:
        best_A, best_score = final_polish(best_A, deadline)
    
    return best_A

def final_polish(A, deadline):
    """Final high-precision polish: try all single-element replacements with nearby values."""
    current_A = A[:]
    current_score = compute_metrics_fast(current_A)
    best_A = current_A[:]
    best_score = current_score
    smooth_pool = smooth_numbers(10000)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 1.0 and no_improve < max_no_improve:
        improved = False
        for i in range(20):
            if time.time() >= deadline - 1.0:
                break
            old_val = current_A[i]
            # Try nearby values and smooth numbers
            candidates_to_try = set()
            for d in range(-10, 11):
                candidates_to_try.add(old_val + d)
            # Add some smooth numbers near old_val
            for s in smooth_pool:
                if abs(s - old_val) <= 20:
                    candidates_to_try.add(s)
            
            for new_val in candidates_to_try:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                
                current_A[i] = new_val
                new_score = compute_metrics_fast(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    break
                else:
                    current_A[i] = old_val
            else:
                current_A[i] = old_val
        
        if not improved:
            no_improve += 1
    
    return best_A, best_score

if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Score: {compute_metrics(result)}")
