import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_val = 256
    S = list(range(2, 129))  # Small numbers: 2..128
    L = list(range(129, 257)) # Large numbers: 129..256
    
    # Precompute reciprocal sums for L subsets
    # We need to calculate sum of 1/l for l in L that are blocked by a set of S elements.
    # Instead of tracking sets, we track a bitmask or just iterate?
    # |L| = 128. Bitmask is 2^128, too big for Python int? No, Python int is arbitrary precision.
    # But iterating 128 items is fast enough.
    
    # Precompute B_s: list of l in L such that lcm(s, l) <= 256
    B_s = {}
    for s in S:
        blocked = []
        for l in L:
            if math.lcm(s, l) <= max_val:
                blocked.append(l)
        B_s[s] = blocked
        
    # Precompute weight of blocked l for each s
    # w_block_s = sum(1/l for l in B_s[s])
    # But we need the sum of NEWLY blocked l.
    
    # Precompute incompatibility in S
    # C_s: list of u in S such that lcm(s, u) <= 256
    C_s = {}
    for s in S:
        conflicts = []
        for u in S:
            if s != u and math.lcm(s, u) <= max_val:
                conflicts.append(u)
        C_s[s] = conflicts

    best_score = 0
    best_set = []
    
    # Base score from L (if S' is empty)
    base_L_score = sum(1.0 / l for l in L)
    
    # Greedy Construction
    # We maintain current_set (list of s in S)
    # And current_blocked (set of l in L that are blocked)
    
    def get_score(current_set, current_blocked):
        s_score = sum(1.0 / s for s in current_set)
        # l_score is base_L_score - sum(1/l for l in current_blocked)
        l_score = base_L_score - sum(1.0 / l for l in current_blocked)
        return s_score + l_score

    # Run multiple greedy passes with different tie-breaking or ordering
    # Since the problem is small, we can also do a local search
    
    def run_greedy():
        current_set = []
        current_blocked = set()
        remaining = set(S)
        
        while True:
            best_gain = 0
            best_s = None
            
            for s in remaining:
                # Check compatibility with current_set
                compatible = True
                for u in current_set:
                    if math.lcm(s, u) <= max_val:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Calculate newly blocked l
                new_blocked = set(B_s[s]) - current_blocked
                loss = sum(1.0 / l for l in new_blocked)
                gain = (1.0 / s) - loss
                
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            
            if best_s is None:
                break
            
            current_set.append(best_s)
            current_blocked.update(B_s[best_s])
            remaining.remove(best_s)
            
        return current_set, current_blocked

    def local_search(initial_set, initial_blocked):
        current_set = initial_set[:]
        current_blocked = initial_blocked.copy()
        
        # Convert to sets for faster lookup
        current_set_set = set(current_set)
        
        improved = True
        while improved:
            improved = False
            
            # Try adding
            for s in S:
                if s in current_set_set:
                    continue
                # Check compatibility
                compatible = True
                for u in current_set:
                    if math.lcm(s, u) <= max_val:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                new_blocked = set(B_s[s]) - current_blocked
                loss = sum(1.0 / l for l in new_blocked)
                gain = (1.0 / s) - loss
                
                if gain > 1e-9:
                    current_set.append(s)
                    current_set_set.add(s)
                    current_blocked.update(new_blocked)
                    improved = True
                    # Restart loop to re-evaluate
                    break
            
            if improved:
                continue
                
            # Try removing
            for i in range(len(current_set)):
                s = current_set[i]
                # Calculate gain of removing s
                # We unblock l in B_s[s] that are NOT blocked by other elements in current_set
                others_blocked = set()
                for u in current_set:
                    if u != s:
                        others_blocked.update(B_s[u])
                
                # l that are blocked ONLY by s
                only_blocked_by_s = set(B_s[s]) - others_blocked
                gain = - (1.0 / s) + sum(1.0 / l for l in only_blocked_by_s)
                
                if gain > 1e-9:
                    current_set.pop(i)
                    current_set_set.remove(s)
                    # Recalculate blocked
                    current_blocked = set()
                    for u in current_set:
                        current_blocked.update(B_s[u])
                    improved = True
                    break
            
            if improved:
                continue

            # Try swapping
            # Remove s1, add s2
            for i in range(len(current_set)):
                s1 = current_set[i]
                for s2 in S:
                    if s2 in current_set_set:
                        continue
                    # Check if s2 compatible with current_set \ {s1}
                    compatible = True
                    for u in current_set:
                        if u == s1:
                            continue
                        if math.lcm(s2, u) <= max_val:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Calculate change in blocked set
                    # Old blocked: current_blocked
                    # New set: (current_set \ {s1}) U {s2}
                    # New blocked: (others_blocked) U B_s[s2]
                    others_blocked = set()
                    for u in current_set:
                        if u != s1:
                            others_blocked.update(B_s[u])
                    
                    new_blocked = others_blocked | set(B_s[s2])
                    
                    # Gain = (1/s2 - 1/s1) + (sum(1/l for l in current_blocked) - sum(1/l for l in new_blocked))
                    # Note: sum(1/l for l in current_blocked) is constant for this comparison step relative to base?
                    # Score = Sum(1/s) + BaseL - Sum(1/l in blocked)
                    # Delta Score = (1/s2 - 1/s1) - (Sum(1/l in new_blocked) - Sum(1/l in current_blocked))
                    
                    # To avoid summing 128 items every time, we can precompute prefix sums?
                    # Or just sum. 128 * 127 * 127 is ~2M ops. Fine.
                    
                    old_loss = sum(1.0 / l for l in current_blocked)
                    new_loss = sum(1.0 / l for l in new_blocked)
                    
                    delta = (1.0 / s2 - 1.0 / s1) - (new_loss - old_loss)
                    
                    if delta > 1e-9:
                        current_set[i] = s2
                        current_set_set.remove(s1)
                        current_set_set.add(s2)
                        current_blocked = new_blocked
                        improved = True
                        break
                if improved:
                    break
                    
        return current_set, current_blocked

    # Run greedy and then local search
    # We can run this a few times with slight perturbations if time permits
    # But greedy is deterministic. Let's try one run.
    
    g_set, g_blocked = run_greedy()
    ls_set, ls_blocked = local_search(g_set, g_blocked)
    
    score = get_score(ls_set, ls_blocked)
    best_score = score
    best_set = ls_set[:]
    
    # Try a few random restarts if time allows
    import random
    random.seed(42)
    
    for _ in range(5):
        if time.time() > deadline:
            break
        
        # Perturb initial set
        # Random subset of size ~ len(g_set)
        # Or just run greedy with random tie-breaking?
        # Greedy is fast. Let's try modifying the gain calculation with noise?
        # Or just random local search from empty?
        
        # Let's try random local search from a random valid set
        # Generate a random valid set
        rand_set = []
        rand_blocked = set()
        for s in random.sample(S, len(S)):
            compatible = True
            for u in rand_set:
                if math.lcm(s, u) <= max_val:
                    compatible = False
                    break
            if compatible:
                rand_set.append(s)
                rand_blocked.update(B_s[s])
                
        ls_set2, ls_blocked2 = local_search(rand_set, rand_blocked)
        score2 = get_score(ls_set2, ls_blocked2)
        
        if score2 > best_score:
            best_score = score2
            best_set = ls_set2[:]
            
    # Final check
    # Ensure the set is valid
    # (Local search should maintain validity)
    
    # Construct final result: best_set from S and all compatible l from L
    # Wait, the problem asks for a subset of [2, 256].
    # My best_set is only from S.
    # I need to add the compatible l from L.
    
    final_S = best_set
    final_L = [l for l in L if all(math.lcm(s, l) > max_val for s in final_S)]
    
    final_result = final_S + final_L
    
    return final_result
