import time
import random
from itertools import combinations

def _can_add(S, x):
    """
    Check if adding x to set S maintains the validity condition.
    S is a set of integers. x is an integer not in S.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements in S U {x}.
    Optimized to O(m^2) where m = len(S).
    """
    S_list = list(S)
    m = len(S_list)
    
    # 1) Check if x divides the sum of any 3 distinct elements in S.
    #    We need: NOT (exists a,b,c in S, a<b<c, (a+b+c) % x == 0)
    if m >= 3:
        # Map residue to count of elements having that residue mod x
        # We need to know if there are 3 distinct elements summing to 0 mod x.
        # This is a 3-sum problem mod x.
        # Simple approach: iterate pairs (a, b), check if (a+b+c) % x == 0.
        # Optimize: For each pair (a,b), target_c_res = (-a-b) % x.
        # We need to know if there is a c in S distinct from a,b with c % x == target_c_res.
        
        # To handle distinctness efficiently:
        # Count residues of all elements in S.
        res_counts = {}
        for v in S_list:
            r = v % x
            res_counts[r] = res_counts.get(r, 0) + 1
        
        for i in range(m - 1):
            a = S_list[i]
            ra = a % x
            for j in range(i + 1, m):
                b = S_list[j]
                rb = b % x
                target_r = (-ra - rb) % x
                
                # We need a c such that c % x == target_r and c != a and c != b.
                # Total count of target_r is res_counts[target_r].
                # How many of {a, b} have residue target_r?
                # a has residue ra. If ra == target_r, a is one of them.
                # b has residue rb. If rb == target_r, b is one of them.
                
                count_needed = 1
                # If a is a candidate for c (i.e., a % x == target_r), it's not available as a distinct c.
                # But we are picking c from S \ {a, b}.
                # So, available_count = res_counts[target_r] - (1 if ra == target_r else 0) - (1 if rb == target_r else 0)
                
                avail = res_counts.get(target_r, 0)
                if ra == target_r:
                    avail -= 1
                if rb == target_r:
                    avail -= 1
                    
                if avail > 0:
                    return False
                        
    # 2) Check for each y in S, if y divides (x + u + v) for any distinct u,v in S \ {y}.
    #    Equivalent: (u + v) % y == (-x) % y
    for y in S_list:
        target = (-x) % y
        # We need a pair u, v in S \ {y} such that (u + v) % y == target.
        
        # Count residues of S \ {y} mod y.
        res_counts_y = {}
        for v in S_list:
            if v == y:
                continue
            r = v % y
            res_counts_y[r] = res_counts_y.get(r, 0) + 1
        
        # Iterate u in S \ {y}
        for u in S_list:
            if u == y:
                continue
            ru = u % y
            needed_r = (target - ru) % y
            
            # We need a v in S \ {y, u} such that v % y == needed_r.
            avail = res_counts_y.get(needed_r, 0)
            # If u itself has residue needed_r, it is one of the candidates, but we need v != u.
            # So subtract 1 if u is a candidate.
            if ru == needed_r:
                avail -= 1
                
            if avail > 0:
                return False
                
    return True

def _is_valid(S):
    """Full validation."""
    S_list = list(S)
    m = len(S_list)
    for idx_a, a in enumerate(S_list):
        others = [v for v in S_list if v != a]
        if len(others) < 3:
            continue
        
        # Check if a divides sum of any 3 in others
        # We can reuse the logic from _can_add part 1, but S=others, x=a.
        # However, _can_add checks if adding x to S is valid.
        # Here we check if a is valid given S=others.
        # This is symmetric.
        if not _can_add(set(others), a):
            return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy selection following a specific order."""
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(current_set, x):
            current_set.add(x)
        if len(current_set) >= 48:
            break
    return current_set

def _local_search_improve(S, deadline):
    """Try to improve set S by adding elements or doing swaps."""
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and time.time() < deadline and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # 1. Try adding any unselected element
        for x in range(1, 193):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    improved = True
                    if len(S) >= 48:
                        return S
                    break
        
        # 2. If no direct add, try 1-out 2-in swap
        if not improved and len(S) < 48:
            best_swap = None
            
            # Try removing one element and adding two
            # To speed up, only try removing elements that are "small" or "divisible"
            # Or just iterate all.
            for a in list(S):
                if time.time() > deadline:
                    break
                trial_set = S - {a}
                
                # Find all elements that can be added to trial_set
                addable = []
                for x in range(1, 193):
                    if x in trial_set:
                        continue
                    if _can_add(trial_set, x):
                        addable.append(x)
                
                # Try pairs from addable
                if len(addable) >= 2:
                    # Sort addable? Maybe larger ones are better?
                    # Just check first few pairs
                    for i in range(len(addable)):
                        for j in range(i+1, len(addable)):
                            x = addable[i]
                            y = addable[j]
                            if _can_add(trial_set | {x}, y):
                                best_swap = (a, x, y)
                                break
                        if best_swap:
                            break
                if best_swap:
                    break
            
            if best_swap:
                a, x, y = best_swap
                S.remove(a)
                S.add(x)
                S.add(y)
                improved = True

    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 192
    best_set = set()
    
    # Strategy 1: Descending Greedy (Large numbers are safer)
    order = list(range(n, 0, -1))
    current_set = _greedy_with_order(order, deadline)
    if len(current_set) > len(best_set):
        best_set = current_set.copy()
    
    if len(best_set) >= 48:
        return list(best_set)

    # Strategy 2: Randomized Greedy with local search
    rng = random.Random(42)
    
    # Try different orders
    num_restarts = 500
    
    for attempt in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate a random order, biased towards larger numbers
        # We can mix descending and random
        if attempt % 2 == 0:
            order = list(range(n, 0, -1))
            # Perturb
            for _ in range(5):
                i = rng.randint(0, n-2)
                order[i], order[i+1] = order[i+1], order[i]
        else:
            # Random shuffle, but keep large numbers somewhat prioritized?
            # Or just random.
            order = list(range(1, n + 1))
            rng.shuffle(order)
        
        current_set = _greedy_with_order(order, deadline)
        
        # Local search
        current_set = _local_search_improve(current_set, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            if len(best_set) >= 48:
                break
    
    # Final local search on best_set
    if time.time() < deadline - 2:
        best_set = _local_search_improve(best_set, deadline)
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    
    # Fallback
    return [192]
