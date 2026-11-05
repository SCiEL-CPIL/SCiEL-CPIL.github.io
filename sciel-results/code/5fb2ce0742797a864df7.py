"""Rewritten program using Iterated Local Search (ILS) with fast 1-in-1-out moves."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    MAX_VAL = 511
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        if not factors[i]: # i is prime
            for j in range(i, MAX_VAL + 1, i):
                factors[j].add(i)
    
    def is_coprime_to_set(a, S):
        fa = factors[a]
        for b in S:
            if fa & factors[b]: # intersection is non-empty
                return False
        return True

    def get_score(S):
        return sum(1.0 / (N - a) for a in S)

    def greedy_fill(S):
        """Add the best possible coprime numbers to S until no more can be added."""
        changed = True
        while changed:
            changed = False
            best_a = None
            best_val = 0.0
            for a in range(MAX_VAL, 0, -1):
                if a not in S:
                    if is_coprime_to_set(a, S):
                        val = 1.0 / (N - a)
                        if val > best_val:
                            best_val = val
                            best_a = a
            if best_a is not None:
                S.append(best_a)
                changed = True
        return S

    def local_search_1in1out(S):
        """Fast local search using 1-out 1-in swaps and additions."""
        S = list(S)
        current_score = get_score(S)
        no_improve = 0
        max_iters = 500 # More iterations since each is fast
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding one
            best_add = None
            best_add_val = 0.0
            for a in range(MAX_VAL, 0, -1):
                if a not in S:
                    if is_coprime_to_set(a, S):
                        val = 1.0 / (N - a)
                        if val > best_add_val:
                            best_add_val = val
                            best_add = a
            if best_add is not None:
                S.append(best_add)
                current_score += best_add_val
                improved = True
                continue

            # Try 1-out 1-in swap
            best_swap_score = current_score
            best_swap_a = None
            best_swap_b = None
            
            for a in S:
                # Temporarily remove a
                S_temp = S
                # We don't want to modify S yet.
                # Check if any b can replace a
                # b must be coprime to S \ {a}
                # This is equivalent to b is coprime to S AND b is coprime to a? No.
                # b must be coprime to all x in S where x != a.
                
                # Optimization: If b is coprime to S, it's definitely coprime to S\{a}.
                # But b might share factors with a, which is fine if a is removed.
                
                for b in range(MAX_VAL, 0, -1):
                    if b == a or b in S:
                        continue
                    
                    # Check if b is coprime to S \ {a}
                    # We can check if b is coprime to S, OR if the only conflict is with a.
                    # Simpler: just check against S \ {a}
                    # To speed up, we can precompute conflicts?
                    
                    # Efficient check:
                    # If factors[b] & (union of factors of S \ {a}) is empty.
                    # This is still O(|S|).
                    
                    # Let's just do the check
                    ok = True
                    for x in S:
                        if x == a:
                            continue
                        if factors[b] & factors[x]:
                            ok = False
                            break
                    if ok:
                        val = current_score - (1.0 / (N - a)) + (1.0 / (N - b))
                        if val > best_swap_score:
                            best_swap_score = val
                            best_swap_a = a
                            best_swap_b = b
                            improved = True

            if best_swap_a is not None:
                S.remove(best_swap_a)
                S.append(best_swap_b)
                current_score = best_swap_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 20:
                    break
                    
        return S, current_score

    # 1. Initial Greedy Solution
    S_init = []
    S_init = greedy_fill(S_init)
    
    # 2. Local Search
    S_best, score_best = local_search_1in1out(S_init)
    
    # 3. Iterated Local Search
    start_time = time.time()
    timeout = 115.0
    
    while time.time() - start_time < timeout:
        # Perturb the best solution
        k = random.randint(2, 6)
        S_perturbed = list(S_best)
        
        for _ in range(k):
            if S_perturbed:
                idx = random.randint(0, len(S_perturbed) - 1)
                S_perturbed.pop(idx)
        
        # Refill greedily
        S_perturbed = greedy_fill(S_perturbed)
        
        # Local Search on perturbed
        S_candidate, score_candidate = local_search_1in1out(S_perturbed)
        
        if score_candidate > score_best:
            S_best = S_candidate
            score_best = score_candidate
            
    return S_best
# EVOLVE-BLOCK-END
