import time
import random
import math
from itertools import combinations

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_candidates():
    """
    Generate a diverse pool of candidate integers for replacement.
    Focus on small integers and smooth numbers.
    """
    candidates = set()
    
    # Small integers provide the best sumset density
    candidates.update(range(1, 301))
    
    # 2,3,5-smooth numbers reduce product set size
    max_exp = 20
    for a in range(max_exp):
        p2 = 2 ** a
        if p2 > 10000:
            break
        for b in range(max_exp):
            p23 = p2 * (3 ** b)
            if p23 > 10000:
                break
            for c in range(max_exp):
                val = p23 * (5 ** c)
                if val > 10000:
                    break
                candidates.add(val)
        # Add 7-smooth too
        for b in range(max_exp):
            p7 = 7 ** b
            if p7 > 10000:
                break
            for a in range(max_exp):
                val = p2 * p7
                if val > 10000:
                    break
                candidates.add(val)
                
    return sorted(candidates)

def local_search(A, time_limit, rng, candidate_pool):
    """
    Perform local search on the set A.
    Uses a hill-climbing approach with random restarts and perturbations.
    """
    A = list(A)
    A.sort()
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Phase 1: Standard local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            # Try a subset of candidates to save time
            # Prioritize smooth numbers and small numbers
            for new_val in candidate_pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    # Update A_set
                    A_set.remove(old_val)
                    A_set.add(new_val)
                    # Break to restart the i-loop with updated A
                    break 

    # Phase 2: Perturbation and re-search (Simulated Annealing-like)
    if (time.time() - start_time) < time_limit * 0.5:
        T = 10.0
        for _ in range(10): # 10 restarts
            if (time.time() - start_time) > time_limit:
                break
            
            # Perturb the current best
            A_perturbed = best_A.copy()
            for _ in range(3): # Swap 3 elements
                i = rng.randint(0, len(A_perturbed)-1)
                # Pick a random candidate not in set
                available = [c for c in candidate_pool if c not in A_perturbed]
                if not available:
                    continue
                new_val = rng.choice(available)
                A_perturbed[i] = new_val
            A_perturbed.sort()
            
            # Local search from perturbed state
            A_perturbed, score_p = local_search(A_perturbed, time_limit * 0.1, rng, candidate_pool)
            if score_p < best_score:
                best_score = score_p
                best_A = A_perturbed

    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate the pool of candidates for replacement
    candidate_pool = generate_candidates()
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Arithmetic Progressions
    # The sumset of an AP of length n has size 2n-1.
    # For n=20, |A+A| = 39. This is the theoretical minimum for distinct integers.
    # We need to minimize |A*A|.
    
    # Try various APs
    ap_candidates = []
    for start in range(1, 51):
        for step in range(1, 21):
            last = start + 19 * step
            if last > 10000:
                continue
            A = [start + i * step for i in range(20)]
            ap_candidates.append(A)
            
    # Also add the "inspiration" candidate: S U cS where S=1..10, c=20
    # A = {1,2,...,10, 20,40,...,200}
    # This set is not an AP, but has structure.
    # Let's evaluate it directly.
    S = list(range(1, 11))
    c = 20
    A_insp = S + [c * x for x in S]
    A_insp = sorted(set(A_insp))
    if len(A_insp) == 20:
        ap_candidates.append(A_insp)
        
    # Also add random smooth sets
    smooth_nums = [x for x in candidate_pool if x <= 10000]
    for _ in range(5):
        sample = rng.sample(smooth_nums[:100], 20)
        ap_candidates.append(sorted(sample))
    
    # Run local search on the most promising candidates
    # We can't run on all APs, so we pick a diverse subset
    # Sort by score first? That's expensive. 
    # Just pick a few good APs (small start, small step)
    
    selected_candidates = []
    # Add first 10 APs (small start/step)
    for i in range(min(10, len(ap_candidates))):
        selected_candidates.append(ap_candidates[i])
    # Add the inspiration candidate
    selected_candidates.append(A_insp)
    # Add the random smooth ones
    selected_candidates.extend(ap_candidates[-5:])
    
    num_candidates = len(selected_candidates)
    
    for idx, A in enumerate(selected_candidates):
        if time.time() > deadline:
            break
        
        # Validate
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        
        # Allocate time for this candidate
        remaining_time = deadline - time.time()
        remaining_count = num_candidates - idx
        alloc_time = max(1.0, remaining_time / remaining_count)
        
        improved_A, score = local_search(A, alloc_time, rng, candidate_pool)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # If local search didn't find anything better, use the best initial
    if best_A is None:
        # Fallback to the best AP found in the coarse search
        best_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        best_score = calc_score(best_A)
        
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
