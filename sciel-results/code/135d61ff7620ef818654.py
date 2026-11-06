import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 180
    
    # Precompute all forbidden triples: distinct a,b,c in [1,N] with 1/a = 1/b + 1/c
    # a = bc/(b+c), so bc = a(b+c)
    # a must be integer, a < min(b,c)
    triples = []
    triple_set = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            # c = ab/(b-a) must be integer and > b (since c > a and we want distinct)
            # bc = a(b+c) => bc - ac = ab => c(b-a) = ab => c = ab/(b-a)
            num = a * b
            den = b - a
            if den <= 0:
                continue
            if num % den == 0:
                c = num // den
                if c > b and c <= N:
                    t = (a, b, c)
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
                # Also check if c could be between a and b (but c > a required for distinctness with a smallest)
                # Actually c = ab/(b-a), and since b > a, c = ab/(b-a) > a always
                # And we need c != b, which is guaranteed since c = ab/(b-a) = b * a/(b-a), 
                # c = b only if a/(b-a) = 1, i.e., a = b-a, i.e., b = 2a, then c = a*2a/a = 2a = b. 
                # So when b = 2a, c = b, not distinct. Skip.
                if c == b:
                    pass  # not distinct, skip
                elif c < b and c > a:
                    # c is between a and b, still a valid triple with distinct elements
                    t = (a, min(b,c), max(b,c))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    # Actually let me redo this more carefully
    # 1/a = 1/b + 1/c => a = bc/(b+c)
    # WLOG let a < b < c (all distinct)
    # Then a = bc/(b+c) < b < c
    # c = ab/(b-a)
    triples = []
    triple_set = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            den = b - a
            if den <= 0:
                continue
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c > b and c <= N:
                t = (a, b, c)
                triple_set.add(t)
                triples.append(t)
            elif c > a and c < b and c != a and c != b:
                # c is between a and b, reorder
                t = (a, c, b)
                triple_set.add(t)
                triples.append(t)
    
    # Build conflict graph: for each element, which triples does it participate in
    # For local search, we need to check if adding an element creates a triple
    
    # Strategy: Start with all elements, then greedily remove to fix conflicts
    # Then try to add back
    
    # Better: Greedy construction
    # Sort elements, add one by one if it doesn't create a triple
    
    def creates_triple(elem, current_set):
        """Check if adding elem to current_set creates a forbidden triple."""
        if elem in current_set:
            return False
        for (a, b, c) in triples:
            # Check if elem completes this triple
            if elem == a:
                if b in current_set and c in current_set:
                    return True
            elif elem == b:
                if a in current_set and c in current_set:
                    return True
            elif elem == c:
                if a in current_set and b in current_set:
                    return True
        return False
    
    # Greedy: add elements in some order
    # Try multiple orderings
    
    best_set = set(range(91, 181))  # seed baseline
    best_size = len(best_set)
    
    def try_add(elem, s):
        if elem in s:
            return s
        if not creates_triple(elem, s):
            s.add(elem)
            return s
        return s
    
    def greedy(order):
        s = set()
        for elem in order:
            if not creates_triple(elem, s):
                s.add(elem)
        return s
    
    # Try different greedy orders
    random.seed(42)
    
    # Order 1: ascending
    s1 = greedy(list(range(1, N+1)))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # Order 2: descending
    s2 = greedy(list(range(N, 0, -1)))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    # Order 3: high to low (favor large elements)
    s3 = greedy(list(range(N, 0, -1)))
    if len(s3) > best_size:
        best_set = s3
        best_size = len(s3)
    
    # Local search: try to add missing elements
    def local_search(s, max_iters=5000):
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            missing = [x for x in range(1, N+1) if x not in s]
            random.shuffle(missing)
            for elem in missing:
                if not creates_triple(elem, s):
                    s.add(elem)
                    improved = True
        return s
    
    # Apply local search to best
    best_set = local_search(best_set, max_iters=10000)
    best_size = len(best_set)
    
    # Try remove-then-add: remove one element, try to add multiple others
    def remove_then_add(s, max_iters=20000):
        s = set(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            # Try removing each element and see if we can add more
            elems = list(s)
            random.shuffle(elems)
            for rem in elems:
                s2 = set(s)
                s2.discard(rem)
                # Try to add missing elements
                missing = [x for x in range(1, N+1) if x not in s2]
                random.shuffle(missing)
                added = 0
                for elem in missing:
                    if not creates_triple(elem, s2):
                        s2.add(elem)
                        added += 1
                if len(s2) > len(s):
                    s = s2
                    improved = True
                    break
        return s
    
    if time.time() < deadline - 5:
        best_set = remove_then_add(best_set, max_iters=50000)
        best_size = len(best_set)
    
    # More local search rounds
    for _ in range(5):
        if time.time() > deadline - 3:
            break
        old_size = len(best_set)
        best_set = local_search(best_set, max_iters=10000)
        if len(best_set) > old_size:
            best_size = len(best_set)
    
    if time.time() < deadline - 3:
        best_set = remove_then_add(best_set, max_iters=50000)
        best_size = len(best_set)
    
    # Final local search
    if time.time() < deadline - 2:
        best_set = local_search(best_set, max_iters=20000)
        best_size = len(best_set)
    
    return list(best_set)
