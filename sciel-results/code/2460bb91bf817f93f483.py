import time
import random
import sys
from itertools import combinations

def compute_metrics(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    sums = []
    prods = []
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sums.append(a + b)
            prods.append(a * b)
    # Deduplicate using set
    sum_count = len(set(sums))
    prod_count = len(set(prods))
    return max(sum_count, prod_count)

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

def compute_metrics_fast(A):
    """Faster metric computation using incremental approach for single-element changes."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        a = A[i]
        for j in range(i, n):
            b = A[j]
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))

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
        # Try smooth numbers clustered in different ranges
        mid_smooths = [s for s in smooths if 100 <= s <= 2000]
        if len(mid_smooths) >= 20:
            candidates.append(mid_smooths[:20])
    
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
    
    # 9. Logarithmic spacing
    import math
    log_spaced = []
    for i in range(20):
        val = int(round(math.exp(i * 0.3)))
        if val < 1:
            val = 1
        if val > 10000:
            val = 10000
        if val not in log_spaced:
            log_spaced.append(val)
    while len(log_spaced) < 20:
        new_val = random.randint(1, 10000)
        if new_val not in log_spaced:
            log_spaced.append(new_val)
    candidates.append(log_spaced[:20])
    
    # 10. Two-cluster structure
    cluster1 = list(range(1, 11))
    cluster2 = [x * 100 for x in range(1, 11)]
    candidates.append(cluster1 + cluster2)
    
    return candidates

def simulated_annealing(A, start_temp, end_temp, start_time, deadline, best_A, best_score):
    """Run simulated annealing from initial state A."""
    current_A = A[:]
    current_score = compute_metrics(current_A)
    
    if current_score < best_score:
        best_score = current_score
        best_A = current_A[:]
    
    n_steps = 0
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        # Adaptive temperature schedule
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = min(1.0, elapsed / max(1.0, total_time))
        # Exponential cooling with floor
        temp = max(end_temp, start_temp * (0.995 ** n_steps))
        
        # Pick random index
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        # Generate new value: mix of local and global moves
        r = random.random()
        if r < 0.35:
            # Local move: small perturbation
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        elif r < 0.65:
            # Medium move: moderate change
            delta = random.randint(-80, 80)
            new_val = old_val + delta
        elif r < 0.85:
            # Large local: significant but bounded change
            delta = random.randint(-300, 300)
            new_val = old_val + delta
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
        new_score = compute_metrics(current_A)
        
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
            if temp > 0.01 and delta_score < 100 and random.random() < min(1.0, (-delta_score) / temp):
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
            n_steps = 0
    
    return best_A, best_score

def final_polish(A, deadline):
    """Final high-precision polish: systematic neighborhood search for each element."""
    current_A = A[:]
    current_score = compute_metrics(current_A)
    
    # Systematic local search: for each position, try all values in a window
    for i in range(20):
        if time.time() > deadline:
            break
        old_val = current_A[i]
        # Try values in a window around the current value
        window = 30
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A and new_val != old_val:
                continue
            
            current_A[i] = new_val
            new_score = compute_metrics(current_A)
            
            if new_score < current_score:
                current_score = new_score
                old_val = new_val  # Continue from new best
            else:
                current_A[i] = old_val
    
    # Second pass with smaller window
    for i in range(20):
        if time.time() > deadline:
            break
        old_val = current_A[i]
        window = 10
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A and new_val != old_val:
                continue
            
            current_A[i] = new_val
            new_score = compute_metrics(current_A)
            
            if new_score < current_score:
                current_score = new_score
                old_val = new_val
            else:
                current_A[i] = old_val
    
    return current_A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit, leave buffer)
    
    # Generate structured candidates
    candidates = generate_structured_candidates()
    
    # Evaluate all candidates
    best_A = None
    best_score = float('inf')
    scored_candidates = []
    
    for cand in candidates:
        if len(cand) != 20:
            continue
        score = compute_metrics(cand)
        scored_candidates.append((score, cand))
        if score < best_score:
            best_score = score
            best_A = cand[:]
    
    if best_A is None:
        return list(range(1, 21))
    
    # Sort candidates by score
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run simulated annealing from top candidates
    start_temp = 50.0
    end_temp = 0.01
    
    # SA from best candidate (use most time)
    sa_deadline = deadline - 10.0
    best_A, best_score = simulated_annealing(
        best_A, start_temp, end_temp, start_time, sa_deadline, best_A, best_score
    )
    
    # If time remains, run SA from second-best candidate
    if time.time() < deadline - 8.0 and len(scored_candidates) >= 2:
        second_score, second_best = scored_candidates[1]
        if second_best != best_A:
            best_A, best_score = simulated_annealing(
                second_best, start_temp * 0.8, end_temp, start_time, deadline - 5.0, best_A, best_score
            )
    
    # If time remains, run SA from third-best
    if time.time() < deadline - 6.0 and len(scored_candidates) >= 3:
        third_score, third_best = scored_candidates[2]
        if third_best != best_A:
            best_A, best_score = simulated_annealing(
                third_best, start_temp * 0.6, end_temp, start_time, deadline - 3.0, best_A, best_score
            )
    
    # Final local search refinement (random)
    if time.time() < deadline - 2.0:
        current_A = best_A[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 1.5 and no_improve < max_no_improve:
            i = random.randint(0, 19)
            old_val = current_A[i]
            
            # Try several nearby values
            tried = set()
            for _ in range(15):
                delta = random.randint(-8, 8)
                new_val = old_val + delta
                if new_val in tried:
                    continue
                tried.add(new_val)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                
                current_A[i] = new_val
                new_score = compute_metrics(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_A = current_A[:]
                    break
                else:
                    current_A[i] = old_val
            else:
                no_improve += 1
    
    # Final systematic polish
    if time.time() < deadline - 0.5:
        best_A, best_score = final_polish(best_A, deadline - 0.1)
    
    return best_A

if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Score: {compute_metrics(result)}")
