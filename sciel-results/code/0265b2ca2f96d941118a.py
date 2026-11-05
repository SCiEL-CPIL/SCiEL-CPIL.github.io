import time
import math
import random
import sys

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # Precompute LCMs for all subsets of [1..N] to speed up greedy checks
    # There are 2^22 = 4,194,304 subsets. This is feasible in Python if optimized or we can compute on the fly.
    # Computing on the fly with memoization might be faster if we don't visit all subsets.
    # However, the greedy algorithm visits at most N steps.
    # Let's stick to the efficient greedy logic from the current program which is already quite fast.
    
    def try_greedy(order):
        S = []
        L = 1
        sums = {0}
        
        for x in order:
            L_new = math.lcm(L, x)
            w_new = L_new // x
            scale = L_new // L
            
            collision = False
            
            # Optimization based on the insight:
            # If x does not divide L, it is ALWAYS safe to add.
            # Proof:
            # We check if s1 * scale + w_new == s2 * scale for s1, s2 in sums.
            # (s1 - s2) * scale == -w_new
            # s1 - s2 == -w_new / scale = -L/x
            # Since s1, s2 are integers, L/x must be an integer.
            # If L % x != 0, L/x is not an integer, so no collision.
            
            if L % x != 0:
                collision = False
            else:
                # x divides L, so scale = 1?
                # L_new = L. scale = 1.
                # w_new = L / x.
                # Check if (s + w_new) in sums.
                for s in sums:
                    if (s + w_new) in sums:
                        collision = True
                        break
            
            if not collision:
                S.append(x)
                L = L_new
                if scale == 1:
                    # sums = sums union (sums + w_new)
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                else:
                    # sums = {s * scale for s in sums} union {s * scale + w_new for s in sums}
                    scaled_sums = {s * scale for s in sums}
                    new_sums = set(scaled_sums)
                    for s in scaled_sums:
                        new_sums.add(s + w_new)
                    sums = new_sums
                
                if time.time() - start_time > timeout - 2:
                    break
        return S

    def is_dissociated(S):
        """Exact verification using integer arithmetic."""
        if not S:
            return True
        L = 1
        for x in S:
            L = math.lcm(L, x)
        vals = [L // x for x in S]
        k = len(S)
        if k > 20:
            # Fallback to a heuristic or just assume true if greedy said so?
            # But we want to be sure. 2^20 is 1M. 2^22 is 4M.
            # We can use a set of sums.
            pass
        
        sums = {0}
        for v in vals:
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)

    best_set = []
    
    # 1. Try descending order (often good for maximizing size in such problems?)
    order_desc = sorted(candidates, reverse=True)
    s1 = try_greedy(order_desc)
    if len(s1) > len(best_set):
        best_set = s1
        
    # 2. Try ascending order
    order_asc = sorted(candidates)
    s2 = try_greedy(order_asc)
    if len(s2) > len(best_set):
        best_set = s2
        
    # 3. Try random orders
    # Increase restarts if time permits
    max_restarts = 500
    for i in range(max_restarts):
        if time.time() - start_time > timeout - 5:
            break
        random.shuffle(candidates)
        s_r = try_greedy(candidates)
        if len(s_r) > len(best_set):
            best_set = s_r
            
    # 4. Local Search / Improvement
    # Try to add elements that were rejected, or swap elements to increase size.
    # Since the greedy method is order-dependent, we can try to "repair" or improve.
    
    # Let's try a more targeted search:
    # Iterate through all subsets of size len(best_set) + 1?
    # C(22, 10) = 646k. C(22, 11) = 705k.
    # If best_set is size 9, we can try to find size 10.
    
    current_best_size = len(best_set)
    
    # If we have time, try to find a larger set via random sampling of larger subsets
    # or by trying to add one more element to the current best set.
    
    # Strategy: Try to add each missing element to best_set and see if it remains dissociated.
    # If yes, add it. Repeat.
    missing = set(candidates) - set(best_set)
    improved = True
    while improved and time.time() - start_time < timeout - 10:
        improved = False
        for x in list(missing):
            test_set = best_set + [x]
            if is_dissociated(test_set):
                best_set = test_set
                missing.remove(x)
                improved = True
                current_best_size += 1
                break # Restart loop with new best_set
                
    # If we still have time, try to find a set of size current_best_size + 1 from scratch
    # by random sampling or local search on a random starting point.
    if time.time() - start_time < timeout - 20:
        # Try to find a set of size current_best_size + 1
        target_size = current_best_size + 1
        # Random restarts with a bias towards larger sets?
        # Or just try more random orders.
        for _ in range(200):
            if time.time() - start_time > timeout - 5:
                break
            random.shuffle(candidates)
            s_r = try_greedy(candidates)
            if len(s_r) > len(best_set):
                best_set = s_r
                current_best_size = len(best_set)

    if not best_set:
        best_set = [1]
        
    return best_set
