import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit

    def can_add(S, x):
        """Check if S U {x} is valid. O(n) time."""
        # Invalid triple (x, b, c) with b,c in S, b!=c:
        # needs (x+b) in S, (x+c) in S, (b+c) in S or (b+c)==x
        # For each b in S with (x+b) in S, check if any c in S (c!=b)
        # has (x+c) in S and (b+c) in S or (b+c)==x
        # Precompute for speed: for each b, list of c where (b+c) in S or b+c==x
        # But that depends on x. Instead:
        # For each b in S where (x+b) in S:
        #   for each c in S where c != b and (x+c) in S:
        #     if (b+c) in S or (b+c) == x: return False
        # To make this O(n), note: for fixed x, the set of b with (x+b) in S is small.
        # Actually worst case could still be O(n^2) but in practice much less.
        # Let's use a different approach:
        # For the new x, the forbidden condition involves pairs (b,c) in S.
        # We need: exists b,c in S, b!=c, such that (x+b) in S AND (x+c) in S AND ((b+c) in S OR b+c==x)
        # Let's iterate over b in S where (x+b) in S. For each such b,
        # we need a c != b in S where (x+c) in S and (b+c) in S or b+c==x.
        # Precompute: for each possible sum s, which pairs (b,c) give b+c=s?
        # That's expensive. Let's just do it directly but efficiently.
        
        # Collect b's that have x+b in S
        # And for each such b, we need to check c's
        # Optimization: build a set of "c candidates" for each b
        # Actually, let's just do the double loop but break early.
        # For n ~ 70, n^2 ~ 5000, which is fine.
        
        # Better: for each b in S with (x+b) in S,
        #   we need c in S, c != b, (x+c) in S, and (b+c) in S or b+c == x
        # Precompute: for this x, which elements c have (x+c) in S?
        # That's: c in S and (x+c) in S, i.e., c in S ∩ (S - x)
        # Let C = {c in S : (x+c) in S}
        # Then for each b in S with (x+b) in S (i.e., b in S ∩ (S-x)),
        #   check if there exists c in C, c != b, with (b+c) in S or b+c == x
        
        # Build C as a set for O(1) lookup
        # But we still need to check (b+c) in S for each pair...
        # Let's just do: for each b in B (where B = {b in S: x+b in S}),
        #   for each c in C (where C = {c in S: x+c in S}),
        #     if c != b and ((b+c) in S or b+c == x): return False
        # |B| and |C| are typically small.
        
        B = [b for b in S if (x + b) in S]
        if not B:
            return True
        C = [c for c in S if (x + c) in S]
        if not C:
            return True
        C_set = set(C)
        for b in B:
            for c in C:
                if c == b:
                    continue
                if (b + c) in S or (b + c) == x:
                    return False
        return True

    def is_valid(S):
        """Full validity check."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a + b) not in S:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    if (a + c) in S and (b + c) in S:
                        return False
        return True

    best_set = set()
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    def greedy_construct(order):
        """Greedy construction given a fixed order of candidates."""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_iters=2000):
        """Improve S by trying additions and swaps."""
        S = set(S)
        not_in_S = set(range(1, 129)) - S
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            progress = False
            
            # Try to add any element
            for x in list(not_in_S):
                if time.time() > deadline:
                    break
                if can_add(S, x):
                    S.add(x)
                    not_in_S.discard(x)
                    progress = True
                    update_best(S)
                    break
            
            if progress:
                continue
            
            # Try swaps: remove y from S, add x from not_in_S
            # Only try a limited number of swaps per iteration
            if not_in_S:
                x = random.choice(list(not_in_S))
                # Try to find a y in S such that S - {y} + {x} is valid
                S_list = list(S)
                random.shuffle(S_list)
                # Try up to 20 random y's
                for y in S_list[:20]:
                    if time.time() > deadline:
                        break
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        S = S_temp
                        S.add(x)
                        not_in_S.discard(x)
                        not_in_S.add(y)
                        progress = True
                        break
            
            if not progress:
                # No addition and no swap worked; try a random restart of swap attempts
                # or break to avoid infinite loop
                # Try a few more random swaps
                if not_in_S and S:
                    x = random.choice(list(not_in_S))
                    S_list = list(S)
                    random.shuffle(S_list)
                    for y in S_list[:50]:
                        if time.time() > deadline:
                            break
                        S_temp = S - {y}
                        if can_add(S_temp, x):
                            S = S_temp
                            S.add(x)
                            not_in_S.discard(x)
                            not_in_S.add(y)
                            progress = True
                            break
                if not progress:
                    break
        return S

    # Phase 1: Multiple greedy constructions
    random.seed(42)
    
    # Ascending
    S = greedy_construct(list(range(1, 129)))
    update_best(S)
    
    # Descending
    S = greedy_construct(list(range(128, 0, -1)))
    update_best(S)
    
    # Several random orders
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        update_best(S)
    
    # Phase 2: Local search from best set
    if time.time() < deadline:
        S = local_search(best_set, max_iters=5000)
        update_best(S)
    
    # Phase 3: Try more random restarts with local search
    for i in range(30):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = greedy_construct(candidates)
        S = local_search(S, max_iters=2000)
        update_best(S)
    
    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
    
    # Verify
    # if not is_valid(best_set):
    #     print("WARNING: invalid set found!")
    
    result = [int(x) for x in sorted(best_set)]
    return result
