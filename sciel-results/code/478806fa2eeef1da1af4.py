import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_lattice_points(max_val=10000):
    """
    Generate all numbers of the form 2^x * 3^y <= max_val.
    These have the property that their product is also 2^(x1+x2) * 3^(y1+y2),
    so the product set size is bounded by the number of distinct exponent pairs.
    """
    points = []
    x = 0
    while 2 ** x <= max_val:
        y = 0
        while (2 ** x) * (3 ** y) <= max_val:
            val = (2 ** x) * (3 ** y)
            points.append((x, y, val))
            y += 1
        x += 1
    return points

def lattice_local_search(A, time_limit, rng, lattice_points, max_val=10000):
    """
    Local search on lattice points. Replace elements with other 2^x*3^y values
    to minimize max(|A+A|, |A*A|).
    """
    A = list(A)
    A.sort()
    
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    # Precompute exponent pairs for each lattice point
    exp_map = {}
    for x, y, val in lattice_points:
        exp_map[val] = (x, y)
    
    # Build a list of candidate values sorted by frequency of appearance
    # (more common values in the lattice are more likely to produce overlaps)
    candidate_vals = [val for _, _, val in lattice_points]
    
    # Sort candidates: prefer smaller values (better sumset density) and
    # values with "central" exponents (more product overlaps)
    def candidate_priority(val):
        x, y = exp_map[val]
        # Central exponents produce more product overlaps
        central_penalty = abs(x - 4) + abs(y - 3)
        return (central_penalty, val)
    
    candidate_vals_sorted = sorted(candidate_vals, key=candidate_priority)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            # Try candidates from the sorted list
            for new_val in candidate_vals_sorted:
                if new_val in A_set or new_val == old_val:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    A_set.remove(old_val)
                    A_set.add(new_val)
                    break  # Restart i-loop with updated A
    
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds
    deadline = start_time + 85.0
    
    rng = random.Random(42)
    
    # Generate lattice points
    lattice_points = generate_lattice_points(10000)
    candidate_vals = [val for _, _, val in lattice_points]
    
    best_A = None
    best_score = float('inf')
    
    # Strategy: Start with structured initial sets based on lattice geometry
    
    initial_sets = []
    
    # 1. Dense cluster: pick 20 points from a small region of the lattice
    #    Points with similar exponents will have products with similar exponents
    #    Small values give better sumset density
    # Try clusters centered at various (x,y) positions
    for cx, cy in [(0,0), (1,0), (0,1), (2,0), (1,1), (3,0), (2,1), (1,2)]:
        # Find 20 lattice points closest to (cx, cy) in exponent space
        def exp_dist(x, y):
            return abs(x - cx) + abs(y - cy)
        
        nearby = sorted(lattice_points, key=lambda p: (exp_dist(p[0], p[1]), p[2]))
        # Take first 20 distinct values
        vals = []
        seen = set()
        for _, _, val in nearby:
            if val not in seen:
                vals.append(val)
                seen.add(val)
            if len(vals) == 20:
                break
        if len(vals) == 20:
            initial_sets.append(sorted(vals))
    
    # 2. Arithmetic progression (known good baseline)
    for start in [1, 2, 3]:
        for step in [1, 2, 3]:
            ap = [start + i * step for i in range(20)]
            if ap[-1] <= 10000:
                initial_sets.append(ap)
    
    # 3. Mixed: some small numbers + some smooth numbers
    small = list(range(1, 11))
    smooth = [2**x * 3**y for x in range(5) for y in range(5) if 2**x * 3**y > 10 and 2**x * 3**y <= 10000]
    if len(smooth) >= 10:
        mixed = small + sorted(smooth)[:10]
        if len(set(mixed)) == 20:
            initial_sets.append(sorted(mixed))
    
    # 4. Geometric-like progression in value space
    for base in [1, 2, 3, 4, 5]:
        geo = [base + i * (i + 1) // 2 for i in range(20)]  # Triangular numbers offset
        if max(geo) <= 10000 and len(set(geo)) == 20:
            initial_sets.append(sorted(geo))
    
    # Deduplicate initial sets
    seen_sets = set()
    unique_initial = []
    for s in initial_sets:
        key = tuple(sorted(s))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_initial.append(s)
    
    initial_sets = unique_initial
    
    num_candidates = len(initial_sets)
    
    for idx, A in enumerate(initial_sets):
        if time.time() > deadline:
            break
        
        # Validate
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        
        # Allocate time for this candidate
        remaining_time = deadline - time.time()
        remaining_count = num_candidates - idx
        alloc_time = max(1.0, remaining_time / remaining_count)
        
        improved_A, score = lattice_local_search(A, alloc_time, rng, lattice_points)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
    
    # If local search didn't find anything better, use the best initial
    if best_A is None:
        best_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        best_score = calc_score(best_A)
    
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
