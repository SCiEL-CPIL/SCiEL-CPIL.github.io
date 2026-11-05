# EVOLVE-BLOCK-START
import time

N = 192
MAX_CARD = 48

def _is_valid_add(c, S):
    """
    Check if c can be added to set S (list of ints).
    1. c must not divide sum of any 3 distinct elements in S.
    2. For each a in S, a must not divide (c + x + y) for any distinct x, y in S\{a}.
    """
    n = len(S)
    
    # Optimization: If c is very large, it's unlikely to divide sums of small numbers.
    # Max sum of 3 elements in S is roughly 3 * max(S).
    # If c > 3 * max(S), it can't divide any sum. But c <= 192, S elements <= 192.
    
    # Check 1: c divides x+y+z for x,y,z in S, x!=y!=z
    # Iterate pairs x, y. Let rem = (-x-y) % c. Check if rem is in S and rem != x and rem != y.
    # This reduces O(n^3) to O(n^2).
    S_set = set(S)
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            # We need z such that x + y + z == k*c
            # z = k*c - x - y
            # Since x,y,z <= 192, sum <= 576.
            # c can be small or large.
            # Instead of checking all k, we can just check if there is a z in S such that (x+y+z) % c == 0.
            # This is hard to do in O(1) without precomputation.
            # However, we can just iterate z? That's O(n^3).
            # Let's stick to O(n^3) but with tight loops, or try the set approach if c is large?
            # Actually, for c=1, it fails immediately. For c > 192, it's trivial.
            # Let's just do the triple loop but break early.
            pass
            
    # Let's use the direct triple loop but optimized.
    # Check 1
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            s2 = x + y
            for k in range(j + 1, n):
                z = S[k]
                if (s2 + z) % c == 0:
                    return False

    # Check 2: For each a in S, check if a divides c + x + y for x, y in S\{a}
    for a in S:
        # We need x, y in S, x != a, y != a, x != y
        # such that (c + x + y) % a == 0
        # Iterate x, y
        for i in range(n):
            x = S[i]
            if x == a:
                continue
            for j in range(i + 1, n):
                y = S[j]
                if y == a:
                    continue
                # x and y are distinct and not a
                if (c + x + y) % a == 0:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy 1: Descending Greedy
    # Large numbers are generally safer to include because they have fewer multiples
    # in the range [1, 192] that can be formed by sums of other numbers.
    selected = []
    candidates = list(range(N, 0, -1))
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_CARD:
            break
            
        if _is_valid_add(c, selected):
            selected.append(c)
            
    current_size = len(selected)
    
    # Strategy 2: Local Search / Repair
    # Try to swap elements to allow more additions.
    # Or simply try to add any remaining element that fits, even if it disrupts order?
    # The greedy approach already considers all elements in descending order.
    # If an element was rejected, it was because of interactions with already selected elements.
    # Swapping a selected element for a rejected one might open up space for others.
    
    # Given time constraints, a simple "try to add any unselected element" loop
    # might not work if the set is "closed".
    # Let's try a few random restarts with different orderings if time permits.
    
    best_set = list(selected)
    
    # If we have time, try a few random permutations to see if we get a larger set.
    import random
    rng = random.Random(42)
    
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 50:
        attempts += 1
        # Shuffle candidates, but bias towards larger numbers?
        # Or just pure random?
        cand = list(range(1, N + 1))
        rng.shuffle(cand)
        
        S = []
        for c in cand:
            if len(S) >= MAX_CARD:
                break
            if _is_valid_add(c, S):
                S.append(c)
                
        if len(S) > len(best_set):
            best_set = list(S)
            
    return best_set
# EVOLVE-BLOCK-END
