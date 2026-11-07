import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit

    def can_add(S, x):
        """Check if S U {x} is valid."""
        # Case 1: x is a member.
        # Need b, c in S, b != c, such that x+b in S, x+c in S, and (b+c in S or b+c == x).
        V = [v for v in S if (x + v) in S]
        if len(V) >= 2:
            for i in range(len(V)):
                b = V[i]
                for j in range(i + 1, len(V)):
                    c = V[j]
                    if (b + c) in S or (b + c) == x:
                        return False
        
        # Case 2: x is a sum.
        # Need a, b, c in S, distinct, such that a+b=x (WLOG) and (a+c in S or a+c=x) and (b+c in S or b+c=x).
        # If a+b=x, then a+c=x implies c=b (not distinct), so a+c must be in S.
        # Similarly b+c must be in S.
        
        for a in S:
            b = x - a
            if b == a:
                continue
            if b in S:
                # Check if there exists c in S, c!=a, c!=b, such that a+c in S and b+c in S.
                S_minus_a = set(y - a for y in S)
                S_minus_b = set(y - b for y in S)
                common = S & S_minus_a & S_minus_b
                if a in common:
                    common.discard(a)
                if b in common:
                    common.discard(b)
                if common:
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
