"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _is_valid_add(x, S):
    """Check if x can be added to S without creating conflicts."""
    n = len(S)
    for i in range(n):
        b = S[i]
        for j in range(i + 1, n):
            c = S[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _count_conflicts_fast(x, S):
    """
    Count conflicts involving x in set S (S includes x).
    For each pair (b,c) in S\{x}, check if x|bc or b|xc or c|xb.
    """
    S_list = [y for y in S if y != x]
    n = len(S_list)
    count = 0
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            if (b * c) % x == 0:
                count += 1
            elif (x * c) % b == 0:
                count += 1
            elif (x * b) % c == 0:
                count += 1
    return count

def _greedy_removal(deadline):
    """
    Start with all numbers. Repeatedly remove the number with the highest conflict degree.
    Stop when the set is valid.
    """
    S = list(UNIVERSE)
    S_set = set(S)
    
    # Check if already valid (unlikely for full set)
    while True:
        if time.time() > deadline:
            break
        
        # Find element with max conflict
        max_conf = -1
        max_idx = -1
        for i, x in enumerate(S):
            if time.time() > deadline:
                break
            conf = _count_conflicts_fast(x, S)
            if conf > max_conf:
                max_conf = conf
                max_idx = i
        
        if max_conf == 0:
            # No conflicts, set is valid
            break
        
        # Remove the element with highest conflict
        S.pop(max_idx)
    
    return S

def _greedy_add(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _is_valid_add(x, S):
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, deadline, max_iters=500, rng_seed=42):
    """
    Local search with 1-swap and 2-swap moves.
    """
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        progress = False
        
        # Try 1-swap
        if S and candidates:
            # Try a few random removals
            for attempt in range(3):
                idx_remove = rng.randrange(len(S))
                removed = S[idx_remove]
                S_temp = S[:]
                S_temp.pop(idx_remove)
                S_temp_set = S_set - {removed}
                
                # Try to add a candidate that fits
                trial_candidates = rng.sample(candidates, min(15, len(candidates)))
                for x in trial_candidates:
                    if _is_valid_add(x, S_temp):
                        S = S_temp + [x]
                        S_set = S_temp_set | {x}
                        if removed not in candidates:
                            candidates.append(removed)
                        if x in candidates:
                            candidates.remove(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        progress = True
                        break
                if progress:
                    break
        
        if progress:
            continue
        
        # Try 2-swap
        if len(S) >= 2 and len(candidates) >= 2:
            idxs = rng.sample(range(len(S)), 2)
            idx1, idx2 = idxs
            if idx1 == idx2:
                continue
            
            r1 = S[idx1]
            r2 = S[idx2]
            
            S_temp = S[:]
            idxs_sorted = sorted([idx1, idx2], reverse=True)
            for idx in idxs_sorted:
                S_temp.pop(idx)
            S_temp_set = set(S_temp)
            
            # Try pairs of candidates
            trial_candidates = rng.sample(candidates, min(12, len(candidates)))
            found = False
            for i in range(len(trial_candidates)):
                if found:
                    break
                for j in range(i + 1, len(trial_candidates)):
                    x1 = trial_candidates[i]
                    x2 = trial_candidates[j]
                    if _is_valid_add(x1, S_temp):
                        S_temp_new = S_temp + [x1]
                        S_temp_set_new = S_temp_set | {x1}
                        if _is_valid_add(x2, S_temp_new):
                            S = S_temp_new + [x2]
                            S_set = S_temp_set_new | {x2}
                            for r in [r1, r2]:
                                if r not in candidates:
                                    candidates.append(r)
                            for x in [x1, x2]:
                                if x in candidates:
                                    candidates.remove(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            progress = True
                            found = True
                            break
        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    best_S = None
    best_size = 0

    # Phase 1: Multiple initialization strategies
    # 1. Greedy removal (new strategy per directive)
    if time.time() < deadline - 30:
        S_rem = _greedy_removal(deadline - 30)
        if S_rem and len(S_rem) > best_size:
            best_S = S_rem
            best_size = len(S_rem)

    # 2. Largest first greedy add
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy_add(order_largest, deadline)
    if S1 and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    # 3. Smallest first greedy add
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy_add(order_smallest, deadline)
    if S2 and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)

    # 4. Randomized greedy orders
    for seed in [1, 7, 42, 100, 200, 500, 777, 1234, 5678, 9999, 31415, 27182]:
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed)
        order_rand = list(range(2, NMAX + 1))
        rng.shuffle(order_rand)
        Sr = _greedy_add(order_rand, deadline)
        if Sr and len(Sr) > best_size:
            best_S = Sr
            best_size = len(Sr)

    if not best_S:
        best_S = [2, 3]
        best_size = 2

    # Phase 2: Local search improvement
    if time.time() < deadline - 15:
        S_improved = _local_search(best_S, deadline - 15, max_iters=3000, rng_seed=42)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    # Run on perturbed version
    if time.time() < deadline - 10:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
        S_improved2 = _local_search(S_perturbed, deadline - 10, max_iters=1500, rng_seed=99)
        if S_improved2 and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Another perturbation
    if time.time() < deadline - 5:
        rng_perturb2 = random.Random(456)
        S_perturbed2 = best_S[:]
        for _ in range(rng_perturb2.randint(1, 2)):
            if S_perturbed2:
                idx = rng_perturb2.randrange(len(S_perturbed2))
                S_perturbed2.pop(idx)
        S_improved3 = _local_search(S_perturbed2, deadline - 5, max_iters=800, rng_seed=200)
        if S_improved3 and len(S_improved3) > best_size:
            best_S = S_improved3
            best_size = len(S_improved3)

    # Final polish: try adding back removed elements
    if time.time() < deadline - 2:
        S_final = best_S[:]
        S_final_set = set(S_final)
        for x in range(NMAX, 1, -1):
            if time.time() > deadline - 2:
                break
            if x not in S_final_set:
                if _is_valid_add(x, S_final):
                    S_final.append(x)
                    S_final_set.add(x)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)

    return [int(x) for x in best_S]
