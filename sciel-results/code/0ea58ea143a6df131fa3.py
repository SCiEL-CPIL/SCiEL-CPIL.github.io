import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 192
    universe = list(range(1, n + 1))
    
    def can_add(S, x):
        """Check if x can be added to S without violating constraints.
        
        Constraint: No selected integer divides the sum of any 3 DISTINCT OTHER selected integers.
        
        Two checks needed:
        1. x does not divide sum of any 3 distinct elements in S
        2. For each y in S, y does not divide (x + u + v) for any u,v in S\{y}, u!=v
        """
        S_list = list(S)
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S
        if K >= 3:
            # Use residue counting: for each residue r mod x, count how many elements have that residue
            # We need to know if there exist 3 distinct elements with residues summing to 0 mod x
            # This is complex; for K < 50, direct triple loop is fine but let's optimize slightly
            # Precompute pair sums mod x
            pair_sums_mod = defaultdict(int)
            for i in range(K):
                for j in range(i + 1, K):
                    ps = (S_list[i] + S_list[j]) % x
                    pair_sums_mod[ps] += 1
            
            for w in S_list:
                needed = (-w) % x
                if pair_sums_mod.get(needed, 0) > 0:
                    # Need to verify the pair doesn't include w
                    # Find if there's a pair (u,v) with u+v ≡ needed (mod x) and w not in {u,v}
                    for i in range(K):
                        if S_list[i] == w:
                            continue
                        for j in range(i + 1, K):
                            if S_list[j] == w:
                                continue
                            if (S_list[i] + S_list[j] + w) % x == 0:
                                return False
                    # If we get here, all pairs summing to needed include w, so no valid triple
                    # But wait, we need to be more careful. Let's just do direct check for correctness.
                    pass
            
            # Simpler: direct triple check but with early termination
            for i in range(K):
                u = S_list[i]
                for j in range(i + 1, K):
                    v = S_list[j]
                    for k in range(j + 1, K):
                        w = S_list[k]
                        if (u + v + w) % x == 0:
                            return False
        
        # Check 2: For each y in S, y divides sum of any triple in S U {x} that includes x
        # The triple is {x, u, v} where u, v in S \ {y} and u != v
        for y in S_list:
            if K < 3:
                continue
            
            # Build list of elements in S excluding y
            others = [z for z in S_list if z != y]
            L = len(others)
            if L < 2:
                continue
            
            # Check if any pair in 'others' sums to -x mod y
            # Use residue counting for speed
            residue_count = defaultdict(int)
            for z in others:
                residue_count[z % y] += 1
            
            # For each u in others, we need v such that (x + u + v) % y == 0
            # i.e., v % y == (-x - u) % y
            # And v != u, v in others
            for i in range(L):
                u = others[i]
                needed_res = (-x - u) % y
                # Check if there's a v in others with residue needed_res and v != u
                # Count how many have that residue
                cnt = residue_count.get(needed_res, 0)
                if cnt > 0:
                    # Need at least one v with that residue, and if u also has that residue, need cnt >= 2
                    if u % y == needed_res:
                        if cnt >= 2:
                            return False
                    else:
                        if cnt >= 1:
                            return False
        
        return True

    def greedy_build(candidate_order, S=None):
        """Build a valid set using greedy approach with given candidate order."""
        if S is None:
            S = set()
        for x in candidate_order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, time_limit_fn):
        """Improve S via local search: try additions, then swaps."""
        S = set(S)
        improved = True
        while improved and time_limit_fn():
            improved = False
            
            # Try to add any element not in S
            not_in_S = [x for x in range(1, n + 1) if x not in S]
            if not_in_S:
                random.shuffle(not_in_S)
                for x in not_in_S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
            
            # If no add possible, try swap: remove one, try to add multiple
            if not improved and S:
                # Try multiple random removals
                best_swap_S = None
                best_swap_size = len(S)
                
                for _ in range(5):  # Try up to 5 random swaps
                    if not time_limit_fn():
                        break
                    
                    r = random.choice(list(S))
                    S_trial = set(S)
                    S_trial.remove(r)
                    
                    # Try to add as many as possible
                    not_in_S_trial = [x for x in range(1, n + 1) if x not in S_trial]
                    random.shuffle(not_in_S_trial)
                    added_count = 0
                    for x in not_in_S_trial:
                        if can_add(S_trial, x):
                            S_trial.add(x)
                            added_count += 1
                    
                    if len(S_trial) > best_swap_size:
                        best_swap_size = len(S_trial)
                        best_swap_S = set(S_trial)
                
                if best_swap_S is not None:
                    S = best_swap_S
                    improved = True
        
        return S

    best_S = set()
    best_size = 0
    
    # Phase 1: Build safe core from upper half [96, 192]
    # Large numbers are unlikely to divide sums of 3 smaller numbers
    upper_half = list(range(n, 95, -1))  # 192 down to 96
    lower_half = list(range(95, 0, -1))   # 95 down to 1
    
    # Multiple restarts with different strategies
    while time.time() - start_time < time_limit:
        # Strategy 1: Start with upper half, then add lower
        S = greedy_build(upper_half + lower_half)
        
        # Strategy 2: Start with upper half only, then randomly add from lower
        if random.random() < 0.5:
            S2 = greedy_build(upper_half)
            lower_candidates = list(range(95, 0, -1))
            random.shuffle(lower_candidates)
            S = greedy_build(lower_candidates, S2)
        
        # Strategy 3: Randomized order with upper half bias
        else:
            candidates = list(range(n, 0, -1))
            # Bias towards upper half by shuffling but keeping upper half more likely earlier
            # Actually, let's just do a mixed approach
            if random.random() < 0.7:
                # 70%: descending
                pass
            else:
                # 30%: random
                random.shuffle(candidates)
            S = greedy_build(candidates)
        
        # Local search refinement
        S = local_search(S, lambda: time.time() - start_time < time_limit)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
