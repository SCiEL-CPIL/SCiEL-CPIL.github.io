import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
K = 3  # number of distinct other elements

def _check_addition(x, S):
    """
    Check if x can be added to S.
    S is a list of integers (a valid set).
    Condition: No element in S ∪ {x} divides the sum of any 3 distinct OTHER elements.
    Since S is valid, we only need to check:
    1. x divides sum of 3 distinct elements from S.
    2. Any a in S divides (x + b + c) where b, c are distinct from a and x.
    """
    S_set = set(S)
    n_S = len(S)
    
    # 1. Check if x divides sum of any 3 distinct elements from S
    # For efficiency: check if there exist a, b, c in S, a != b != c, such that (a+b+c) % x == 0
    if n_S >= 3:
        # Precompute: for each a in S, find if there exist b, c in S \ {a}, b != c, such that (a + b + c) % x == 0
        # i.e., (b + c) % x == (-a) % x
        # For each a, we need to check if there exist b, c in S \ {a}, b != c, with (b+c) % x == (-a) % x
        # This is O(n_S^2) per a, total O(n_S^3) but with early termination
        for a in S:
            target = (-a) % x
            # Find b, c in S \ {a}, b != c, such that (b + c) % x == target
            # We can use a set for quick lookup
            for b in S:
                if b == a:
                    continue
                c = (target - b) % x
                # c must be in S, c != a, c != b
                # But c is a residue, not necessarily in S
                # We need actual elements c_val in S such that c_val % x == c
                # Let's iterate over S to find c
                for c_val in S:
                    if c_val == a or c_val == b:
                        continue
                    if (c_val % x) == c:
                        return False
    # Wait, the above is O(n_S^3) which is the same as the original. Let me optimize.
    # Better: For each a in S, compute needed = (-a) % x
    # For each b in S \ {a}, compute needed_c = (needed - b) % x
    # Check if there exists c in S \ {a, b} such that c % x == needed_c
    # We can precompute residues: for each residue r in [0, x-1], which elements in S have that residue?
    # But x can be up to 192, so this is feasible.
    
    # Let me redo this more efficiently:
    # Actually, the original triple loop is O(n_S^3) with n_S <= 48, so 48^3 = 110,592
    # That's acceptable. Let's keep it simple but use early termination.
    
    # Let me just use the original approach but with a set for quick membership
    if n_S >= 3:
        for a, b, c in combinations(S, 3):
            if (a + b + c) % x == 0:
                return False
    
    # 2. Check if any a in S divides (x + b + c) where b, c are distinct from a and x
    # For each a in S: x + b + c must not be divisible by a
    # i.e., (x + b + c) % a != 0 for all b, c in S \ {a}, b != c
    # Equivalent: for each a, for each b in S \ {a}, for each c in S \ {a, b}:
    #   (x + b + c) % a != 0
    # This is O(n_S^3) again.
    
    if n_S >= 2:
        for a in S:
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if (x + b + c) % a == 0:
                        return False
    
    return True

def _check_addition_optimized(x, S):
    """
    Optimized version using residue counting.
    """
    n_S = len(S)
    
    # 1. Check if x divides sum of any 3 distinct elements from S
    # i.e., exists a, b, c in S, distinct, with (a+b+c) % x == 0
    if n_S >= 3:
        # For each a, we need b, c in S \ {a}, b != c, with (b+c) % x == (-a) % x
        # Precompute residue counts: for each residue r in [0, x-1], list of elements with that residue
        residue_map = {}
        for elem in S:
            r = elem % x
            if r not in residue_map:
                residue_map[r] = []
            residue_map[r].append(elem)
        
        for a in S:
            target = (-a) % x
            # Need b, c in S \ {a}, b != c, with (b + c) % x == target
            # For each b in S \ {a}, need c in S \ {a, b} with c % x == (target - b) % x
            for b in S:
                if b == a:
                    continue
                needed_c_res = (target - b) % x
                # Check if there exists c in S \ {a, b} with c % x == needed_c_res
                # We can check residue_map[needed_c_res] for any element that is not a or b
                if needed_c_res in residue_map:
                    c_list = residue_map[needed_c_res]
                    # Check if any c in c_list is not a and not b
                    found = False
                    for c in c_list:
                        if c != a and c != b:
                            found = True
                            break
                    if found:
                        return False
                    # Also need to check if c_list has at least 2 elements that are not a or b
                    # Actually, the above loop already checks for at least one valid c
    # Wait, there's a subtle issue. The above checks if there exist b, c such that (a+b+c) % x == 0.
    # But it might miss cases where b and c are the same element? No, because we skip c == b.
    # Actually, the inner loop iterates over all b in S \ {a}, and for each b, checks if there's a c in S \ {a, b} with the right residue.
    # This is correct.
    
    # 2. Check if any a in S divides (x + b + c) where b, c are distinct from a and x
    # For each a in S: (x + b + c) % a != 0 for all b, c in S \ {a}, b != c
    # Equivalent: for each a, for all b, c in S \ {a}, b != c: (x + b + c) % a != 0
    # i.e., for each a, the set { (b + c) % a : b, c in S \ {a}, b != c } must not contain (-x) % a
    
    if n_S >= 2:
        for a in S:
            target = (-x) % a
            # Check if there exist b, c in S \ {a}, b != c, with (b + c) % a == target
            # Precompute residue map for S \ {a}
            residue_map_a = {}
            for elem in S:
                if elem == a:
                    continue
                r = elem % a
                if r not in residue_map_a:
                    residue_map_a[r] = []
                residue_map_a[r].append(elem)
            
            # For each b in S \ {a}, check if there exists c in S \ {a, b} with c % a == (target - b) % a
            for b in S:
                if b == a:
                    continue
                needed_c_res = (target - b) % a
                if needed_c_res in residue_map_a:
                    c_list = residue_map_a[needed_c_res]
                    found = False
                    for c in c_list:
                        if c != b:  # c != a is already guaranteed since we excluded a
                            found = True
                            break
                    if found:
                        return False
    
    return True

def _greedy_order(candidates, S, deadline):
    """
    Greedy selection with a heuristic score.
    Score: prefer elements that are "safe" (don't conflict with current S) and that leave more options.
    """
    S_set = set(S)
    remaining = [x for x in candidates if x not in S_set]
    
    # Precompute for each remaining x whether it's valid to add
    valid = []
    for x in remaining:
        if _check_addition_optimized(x, S):
            valid.append(x)
    
    if not valid:
        return S
    
    # Heuristic: sort valid candidates by some score
    # Option 1: Descending (large numbers first)
    # Option 2: Ascending (small numbers first)
    # Option 3: Random
    # Let's try a mix: use a score based on "constraint tightness"
    
    # For now, let's try descending order (largest first) as it tends to work well
    valid.sort(reverse=True)
    
    # Pick the first valid candidate
    x = valid[0]
    S.append(x)
    S_set.add(x)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit (10s safety margin)
    
    best_set = []
    
    # Strategy 1: Deterministic Greedy (Descending)
    S = []
    for x in range(192, 0, -1):
        if _check_addition_optimized(x, S):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
    if len(best_set) >= MAX_SIZE:
        return best_set[:MAX_SIZE]
    
    # Strategy 2: Randomized Greedy with multiple restarts
    for i in range(1000):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if _check_addition_optimized(x, S):
                S.append(x)
        
        # Local search: try to add more
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_set = set(S)
            remaining = [x for x in range(1, 193) if x not in S_set]
            for x in remaining:
                if _check_addition_optimized(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= MAX_SIZE:
                return best_set[:MAX_SIZE]
    
    # Strategy 3: If we have a good set, try to improve by swapping
    if len(best_set) > 0:
        S = list(best_set)
        S_set = set(S)
        remaining = [x for x in range(1, 193) if x not in S_set]
        
        # Try removing one element and adding another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S)):
                for x in remaining:
                    # Try swapping S[i] with x
                    new_S = S[:i] + S[i+1:]
                    if _check_addition_optimized(x, new_S):
                        new_S.append(x)
                        if len(new_S) > len(S):
                            S = new_S
                            S_set = set(S)
                            remaining = [y for y in range(1, 193) if y not in S_set]
                            improved = True
                            break
                if improved:
                    break
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= MAX_SIZE:
                return best_set[:MAX_SIZE]
    
    return best_set
