"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def compute_score(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(U[i] + U[j])
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    
    # The score: 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 1.0
    score = 1.0 + math.log(ratio) / denom
    return score

def local_search(initial_U, max_iters=5000, temperature=1.0):
    """Local search to improve the set."""
    U = sorted(set(initial_U))
    if 0 not in U:
        U.append(0)
    U = sorted(U)
    
    best_U = U[:]
    best_score = compute_score(U)
    current_score = best_score
    
    n = len(U)
    max_val = max(U) if U else 0
    
    for it in range(max_iters):
        # Pick a random element to modify (not 0)
        if n <= 1:
            break
        
        # Choose which element to modify
        idx = random.randint(1, n - 1)  # don't modify 0
        old_val = U[idx]
        
        # Try a few candidate new values
        candidates = []
        # Random perturbation
        delta = random.randint(-max(1, max_val // 10), max(1, max_val // 5))
        new_val = old_val + delta
        if new_val > 0:
            candidates.append(new_val)
        
        # Try values near existing elements
        for other in U:
            if other != old_val:
                candidates.append(other + random.randint(-2, 2))
                candidates.append(other - random.randint(1, 3))
        
        # Try geometric-like steps
        candidates.append(old_val * 2)
        candidates.append(old_val // 2 if old_val > 2 else 1)
        
        best_candidate = None
        best_candidate_score = current_score
        
        for cand in candidates:
            if cand <= 0:
                continue
            new_U = U[:]
            new_U[idx] = cand
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U.append(0)
            new_U = sorted(new_U)
            
            if len(new_U) != n:
                # Size changed, still valid but let's be careful
                pass
            
            cand_score = compute_score(new_U)
            if cand_score > best_candidate_score:
                best_candidate_score = cand_score
                best_candidate = new_U
        
        if best_candidate is not None:
            U = best_candidate
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        else:
            # No improvement, maybe accept a worse move with some probability (simulated annealing)
            if random.random() < 0.1:
                # Pick a random candidate and accept it
                if candidates:
                    valid_cands = [c for c in candidates if c > 0]
                    if valid_cands:
                        cand = random.choice(valid_cands)
                        new_U = U[:]
                        new_U[idx] = cand
                        new_U = sorted(set(new_U))
                        if 0 not in new_U:
                            new_U.append(0)
                        new_U = sorted(new_U)
                        U = new_U
                        current_score = compute_score(U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Generate multiple seed structures
    seeds = []
    
    # Geometric progression
    for ratio in [2, 3, 4, 5]:
        seed = [0]
        val = 1
        for i in range(10):
            seed.append(val)
            val *= ratio
        seeds.append(seed[:8])  # Keep size manageable
    
    # Fibonacci-like
    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    seeds.append(sorted(set(fib[:8])))
    
    # Random sets
    for _ in range(5):
        n = random.randint(5, 10)
        vals = sorted(random.sample(range(1, 200), n - 1))
        seeds.append([0] + vals)
    
    # Power of 2
    seeds.append([0] + [2**i for i in range(1, 10)])
    
    # Mixed: small dense + sparse large
    for _ in range(3):
        small = list(range(0, random.randint(3, 6)))
        large = [small[-1] + random.randint(5, 50) for _ in range(random.randint(3, 6))]
        seeds.append(sorted(set(small + large)))
    
    # Run local search on each seed
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        result_U, result_score = local_search(seed, max_iters=2000)
        if result_score > best_score:
            best_score = result_score
            best_U = result_U
    
    # Final polish: try to improve the best solution further
    if time.time() < deadline - 5:
        result_U, result_score = local_search(best_U, max_iters=3000)
        if result_score > best_score:
            best_score = result_score
            best_U = result_U
    
    # Ensure 0 is in the set and it's sorted
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U
# EVOLVE-BLOCK-END
