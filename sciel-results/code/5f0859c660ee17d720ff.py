"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _valid_sidon(a):
    """Check if set a has all distinct pairwise sums a+b (a<=b)."""
    sums = set()
    n = len(a)
    for i in range(n):
        for j in range(i, n):
            s = a[i] + a[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_construct():
    """Greedy construction starting from {1,2,5,11}."""
    a, sums = [1, 2, 5, 11], set()
    for i, x in enumerate(a):
        for y in a[i:]:
            sums.add(x + y)
    for x in range(12, 285):
        new = {x + y for y in a} | {2 * x}
        if len(new) == len(a) + 1 and not new & sums:
            a.append(x)
            sums |= new
    return a

def _local_search(a, time_limit=10.0):
    """Try to improve the set by swapping elements or adding new ones."""
    start_time = time.time()
    best = list(a)
    best_size = len(best)
    
    def _sums_of(arr):
        s = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s.add(arr[i] + arr[j])
        return s
    
    best_sums = _sums_of(best)
    
    # Try adding elements
    for x in range(1, 285):
        if x in best:
            continue
        new_arr = best + [x]
        new_sums = best_sums | {x + y for y in best} | {2 * x}
        if len(new_sums) == len(best_sums) + len(best) + 1:
            best = new_arr
            best_sums = new_sums
            best_size = len(best)
            if time.time() - start_time > time_limit:
                break
    
    # Try swapping elements to free up space
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i, old in enumerate(best):
            for x in range(1, 285):
                if x == old or x in best:
                    continue
                # Try replacing old with x
                candidate = best[:i] + [x] + best[i+1:]
                candidate.sort()
                cand_sums = _sums_of(candidate)
                # Check if we can add any new element
                for y in range(1, 285):
                    if y in candidate:
                        continue
                    new_arr = candidate + [y]
                    new_sums = cand_sums | {y + z for z in candidate} | {2 * y}
                    if len(new_sums) == len(cand_sums) + len(candidate) + 1:
                        best = new_arr
                        best_sums = new_sums
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    a = _greedy_construct()
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5:
        a = _local_search(a, time_limit=min(30, deadline - time.time() - 5))
    
    # Phase 3: Try more greedy passes with different orderings
    while time.time() < deadline - 3:
        improved = False
        for x in range(284, 0, -1):  # Try from high to low
            if x in a:
                continue
            new = {x + y for y in a} | {2 * x}
            if len(new) == len(a) + 1 and not new & {a[i] + a[j] for i in range(len(a)) for j in range(i, len(a))}:
                a.append(x)
                a.sort()
                improved = True
                break
        if not improved:
            break
    
    # Ensure prescribed elements are present
    for req in [1, 2, 5, 11]:
        if req not in a:
            a.append(req)
            a.sort()
    
    return a
# EVOLVE-BLOCK-END

