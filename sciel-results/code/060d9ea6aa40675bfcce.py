"""Find large subset of [1,180] with no three distinct elements a,b,c where 1/a=1/b+1/c."""
import time
import random

def _is_forbidden_third(x, y):
    """Given x,y, compute z = xy/(x+y). Return z if it's a valid distinct integer in [1,180], else None."""
    num = x * y
    den = x + y
    if den == 0:
        return None
    if num % den != 0:
        return None
    z = num // den
    if z < 1 or z > 180:
        return None
    if z == x or z == y:
        return None
    return z

def _greedy_construct(order, N=180):
    """Greedy: add elements in given order if no conflict with existing set."""
    S = set()
    for x in order:
        conflict = False
        for y in S:
            if x == y:
                continue
            z = _is_forbidden_third(x, y)
            if z is not None and z in S:
                conflict = True
                break
        if not conflict:
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    N = 180
    all_elements = list(range(1, N + 1))
    
    best_set = _greedy_construct(all_elements)
    best_size = len(best_set)
    
    # Precompute conflict graph for speed: for each x, list of y such that _is_forbidden_third(x,y) is valid
    # Actually, let's precompute for each pair (x,y) the third element
    # For speed in greedy, precompute for each x, the set of (y, z) pairs where z = xy/(x+y) is valid
    
    # Better: precompute a dict: for each x, a set of y values such that adding x when y is present creates conflict
    # i.e., there exists z in S with {x,y,z} forbidden. But z depends on S.
    # Instead, for each x and y, precompute z. Then in greedy, for new x, check all y in S:
    #   z = third(x,y); if z in S, conflict.
    
    # Let's precompute third_element[x][y] for all x,y
    third = {}
    for x in range(1, N + 1):
        third[x] = {}
        for y in range(1, N + 1):
            if x == y:
                continue
            z = _is_forbidden_third(x, y)
            if z is not None:
                third[x][y] = z
    
    def greedy_with_order(order):
        S = set()
        for x in order:
            conflict = False
            for y in S:
                if x in third and y in third[x]:
                    z = third[x][y]
                    if z in S:
                        conflict = True
                        break
            if not conflict:
                S.add(x)
        return S
    
    # Multiple strategies for ordering
    strategies = []
    
    # Strategy 1: natural order
    strategies.append(list(range(1, N + 1)))
    # Strategy 2: reverse order
    strategies.append(list(range(N, 0, -1)))
    # Strategy 3: random orders
    random.seed(42)
    for _ in range(20):
        order = all_elements[:]
        random.shuffle(order)
        strategies.append(order)
    
    # Try each strategy
    for order in strategies:
        if time.time() > deadline:
            break
        S = greedy_with_order(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local improvement: try to add elements not in best_set
    if time.time() < deadline:
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x in current:
                    continue
                conflict = False
                for y in current:
                    if x in third and y in third[x]:
                        z = third[x][y]
                        if z in current:
                            conflict = True
                            break
                if not conflict:
                    current.add(x)
                    improved = True
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # More random restarts with local improvement
    random.seed(123)
    for restart in range(50):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        S = greedy_with_order(order)
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                conflict = False
                for y in S:
                    if x in third and y in third[x]:
                        z = third[x][y]
                        if z in S:
                            conflict = True
                            break
                if not conflict:
                    S.add(x)
                    improved = True
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Try swapping: remove one element, try to add two
    if time.time() < deadline:
        current = set(best_set)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Try removing a random element and adding others
            if len(current) < 2:
                break
            removed = random.choice(list(current))
            current.remove(removed)
            # Try to add elements
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in range(1, N + 1):
                    if x in current:
                        continue
                    conflict = False
                    for y in current:
                        if x in third and y in third[x]:
                            z = third[x][y]
                            if z in current:
                                conflict = True
                                break
                    if not conflict:
                        current.add(x)
                        improved = True
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
            else:
                # Revert
                current.add(removed)
                # Re-run local improvement to get back
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for x in range(1, N + 1):
                        if x in current:
                            continue
                        conflict = False
                        for y in current:
                            if x in third and y in third[x]:
                                z = third[x][y]
                                if z in current:
                                    conflict = True
                                    break
                        if not conflict:
                            current.add(x)
                            improved = True
    
    return list(best_set)
