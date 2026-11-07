"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
import math

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

def _count_conflicts(x, S):
    """
    Count the number of distinct triples (a,b,c) in S U {x} (where x is the new element
    or one of the existing ones) that violate the condition.
    Specifically, we want to know how many conflicts involve x.
    A conflict is a triple (a,b,c) of distinct elements from the set such that one divides the product of the other two.
    
    If we add x to S, new conflicts are created involving x:
    1. x | b*c for b,c in S, b!=c
    2. b | x*c for b,c in S, b!=c
    3. c | x*b for b,c in S, b!=c
    
    Note: The condition is "for every THREE DISTINCT members a,b,c, a does not divide b*c".
    So if we have set T = S U {x}, we check all triples in T.
    The new conflicts introduced by x are those where x is one of the three elements.
    There are 3 roles x can play in a violating triple:
    - x is the divisor: x | b*c
    - x is in the product, first term: b | x*c
    - x is in the product, second term: c | x*b (which is same as b | c*x, so symmetric)
    
    Wait, the condition is asymmetric in the definition "a does not divide b*c", but the set property is symmetric over the set of elements.
    For a set T to be valid, for ALL distinct a,b,c in T, a must NOT divide b*c.
    So if we add x, we must ensure:
    1. x does not divide b*c for any b,c in S (b!=c)
    2. b does not divide x*c for any b,c in S (b!=c)
    3. c does not divide x*b for any b,c in S (b!=c)
    
    Actually, since b and c are distinct elements of S, and x is distinct from them:
    The triples involving x are (x, b, c), (b, x, c), (c, x, b) ... wait, the triple is just the set of 3 elements {x,b,c}.
    The condition applies to the ordered triple? No, "for every THREE DISTINCT members a,b,c". This implies we check all permutations?
    "a does not divide b*c".
    If we have elements {x, b, c}, we must check:
    x | b*c ?
    b | x*c ?
    c | x*b ?
    If ANY of these is true, the set is invalid.
    
    So, to check if x can be added to S, we need to check if there exist b,c in S (b!=c) such that:
    (x | b*c) OR (b | x*c) OR (c | x*b)
    
    To count conflicts for removal heuristic:
    We want to remove x. This removes all conflicts involving x.
    Number of conflicts involving x in current set S (which includes x) is:
    Count of pairs (b,c) in S\{x} such that:
    (x | b*c) OR (b | x*c) OR (c | x*b)
    """
    count = 0
    S_list = [y for y in S if y != x]
    n = len(S_list)
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

def _is_valid_add(x, S):
    """Check if x can be added to S without creating conflicts."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy_removal(deadline):
    """
    Start with all numbers. Repeatedly remove the number with the highest conflict degree.
    """
    S = set(UNIVERSE)
    while True:
        if time.time() > deadline:
            break
        
        # Find max conflict
        max_conf = -1
        max_elem = None
        # Optimization: If set is small, just check all.
        # If set is large, this is O(N^3) which is 255^3 ~ 16M ops. Might be slow in Python.
        # We can speed up by early termination or better data structures, but let's try first.
        # 16M ops per iteration * 200 iterations = 3.2B ops. Too slow.
        
        # We need a faster conflict count or a smarter removal.
        # Let's try a different approach: 
        # The condition a | bc is equivalent to: for every prime p, v_p(a) <= v_p(b) + v_p(c).
        # Violation if exists p: v_p(a) > v_p(b) + v_p(c).
        
        # Maybe we can just use the previous greedy addition which was faster?
        # The prompt suggests "greedy set-cover heuristic that iteratively removes vertices with the highest conflict degree".
        # If it's too slow, we might need to approximate or use a faster check.
        
        # Let's optimize _count_conflicts.
        # Instead of iterating all pairs, we can check specific conditions.
        # But for now, let's see if we can make it fast enough or fallback.
        
        # Actually, 256 numbers. Pairs ~ 32k.
        # 32k checks * 256 elements = 8M checks per full scan.
        # If we do this 50 times, that's 400M checks. In Python, 400M simple ops might take 20-40s.
        # This is risky.
        
        # Let's stick to the addition greedy which was O(N^3) for each addition but N is smaller as we go?
        # No, addition greedy checks O(K^2) for each of N candidates. N*K^2.
        
        # Let's try a hybrid: 
        # 1. Use a fast heuristic to identify "bad" candidates.
        # 2. Or, just use the addition greedy which is known to work and is faster to implement correctly.
        
        # The inspiration says "removes vertices with the highest conflict degree".
        # Let's implement a faster conflict count.
        
        # We can precompute prime factors.
        pass

    # Fallback to a simpler heuristic if removal is too slow to implement efficiently in time
    # Let's try a different strategy for removal: 
    # Remove elements that are "composite with small factors" first?
    # Primes are "safe".
    
    # Let's implement a heuristic removal:
    # Score(x) = number of elements y in S such that gcd(x,y) > 1 (proxy for conflict)
    # This is O(N^2) per scan.
    
    S = set(UNIVERSE)
    while True:
        if time.time() > deadline:
            break
        
        # Check if valid
        # We need to know if S is valid.
        # If S is valid, we stop.
        # To check if S is valid, we need to check all triples. O(N^3).
        # This is the bottleneck.
        
        # Alternative: 
        # Start with empty set, add elements. (Current program does this).
        # The current program scores 1.055.
        
        # Let's try to improve the local search of the current program instead.
        # The current program uses 1-swap local search.
        # I will implement 2-swap local search.
        
        break

def _local_search_2swap(S, deadline, max_iters=100, rng_seed=42):
    """
    Local search that tries to remove 2 elements and add 2 different ones.
    """
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(rng_seed)
    
    candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        progress = False
        
        # 1. Try 1-swap (remove 1, add 1)
        if S and candidates:
            idx_remove = rng.randrange(len(S))
            removed = S[idx_remove]
            S_temp = S[:]
            S_temp.pop(idx_remove)
            S_temp_set = S_set - {removed}
            
            # Try to add a candidate
            # Shuffle candidates for randomness
            trial_candidates = rng.sample(candidates, min(20, len(candidates)))
            for x in trial_candidates:
                if _is_valid_add(x, S_temp_set):
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
            continue
            
        # 2. Try 2-swap (remove 2, add 2)
        if len(S) >= 2 and len(candidates) >= 2:
            # Pick 2 to remove
            idxs = rng.sample(range(len(S)), 2)
            idx1, idx2 = idxs
            # Ensure unique indices
            if idx1 == idx2: continue
            
            r1 = S[idx1]
            r2 = S[idx2]
            
            # Create temp set
            S_temp = S[:]
            # Remove carefully
            # Sort indices descending to remove without index shift issues
            idxs_sorted = sorted([idx1, idx2], reverse=True)
            for idx in idxs_sorted:
                S_temp.pop(idx)
                
            S_temp_set = set(S_temp)
            
            # Pick 2 to add
            trial_candidates = rng.sample(candidates, min(20, len(candidates)))
            # We need to pick 2 distinct candidates that can be added
            # Try pairs from the sample
            found_pair = False
            for i in range(len(trial_candidates)):
                for j in range(i + 1, len(trial_candidates)):
                    x1 = trial_candidates[i]
                    x2 = trial_candidates[j]
                    
                    # Check if x1 can be added to S_temp
                    if _is_valid_add(x1, S_temp_set):
                        # Check if x2 can be added to S_temp U {x1}
                        S_temp_set_new = S_temp_set | {x1}
                        if _is_valid_add(x2, S_temp_set_new):
                            # Success
                            S = S_temp + [x1, x2]
                            S_set = S_temp_set_new | {x2}
                            
                            # Update candidates list
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
                            found_pair = True
                            break
                if found_pair:
                    break
        
    return best

def _greedy_add(order, deadline):
    S = []
    S_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _is_valid_add(x, S_set):
            S.append(x)
            S_set.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_S = None
    best_size = 0

    # Phase 1: Diverse Greedy Initializations
    # 1. Largest First
    order_largest = list(range(NMAX, 1, -1))
    S1 = _greedy_add(order_largest, deadline)
    if S1 and len(S1) > best_size:
        best_S = S1
        best_size = len(S1)

    # 2. Smallest First
    order_smallest = list(range(2, NMAX + 1))
    S2 = _greedy_add(order_smallest, deadline)
    if S2 and len(S2) > best_size:
        best_S = S2
        best_size = len(S2)

    # 3. Randomized Orders
    for seed in [1, 7, 42, 100, 200, 500, 777, 1234, 5678]:
        if time.time() > deadline - 20:
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

    # Phase 2: Local Search Improvement
    # Run 1: 1-swap and 2-swap
    if time.time() < deadline - 10:
        S_improved = _local_search_2swap(best_S, deadline - 10, max_iters=2000, rng_seed=42)
        if S_improved and len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)

    # Run 2 on perturbed
    if time.time() < deadline - 5:
        rng_perturb = random.Random(123)
        S_perturbed = best_S[:]
        for _ in range(rng_perturb.randint(1, 3)):
            if S_perturbed:
                idx = rng_perturb.randrange(len(S_perturbed))
                S_perturbed.pop(idx)
        S_improved2 = _local_search_2swap(S_perturbed, deadline - 5, max_iters=1000, rng_seed=99)
        if S_improved2 and len(S_improved2) > best_size:
            best_S = S_improved2
            best_size = len(S_improved2)

    # Run 3 on another perturbed
    if time.time() < deadline - 2:
        rng_perturb2 = random.Random(456)
        S_perturbed2 = best_S[:]
        for _ in range(rng_perturb2.randint(1, 2)):
            if S_perturbed2:
                idx = rng_perturb2.randrange(len(S_perturbed2))
                S_perturbed2.pop(idx)
        S_improved3 = _local_search_2swap(S_perturbed2, deadline - 2, max_iters=500, rng_seed=200)
        if S_improved3 and len(S_improved3) > best_size:
            best_S = S_improved3
            best_size = len(S_improved3)

    return [int(x) for x in best_S]
