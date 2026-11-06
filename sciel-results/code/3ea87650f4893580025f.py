import time

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _get_residue_classes(points, p):
    classes = {}
    for v in points:
        r = v % p
        if r not in classes:
            classes[r] = set()
        classes[r].add(v)
    return classes

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    points = set(range(N + 1))
    primes = _primes(N + 1)
    
    # We will process primes in order.
    # For small primes, we use lookahead to choose the best residue class to remove.
    # For larger primes, we just pick the smallest class.
    
    idx = 0
    while idx < len(primes):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        if p > len(points):
            break
            
        classes = _get_residue_classes(points, p)
        
        if len(classes) < p:
            # Already admissible for p
            idx += 1
            continue
            
        # Need to remove one class
        # Sort classes by size (ascending)
        sorted_classes = sorted(classes.items(), key=lambda item: len(item[1]))
        
        best_points = None
        best_score = -1 # We want to maximize remaining size, or minimize cost
        
        # Lookahead depth: 1 step (next prime)
        # Try the top K smallest classes
        K = min(5, len(sorted_classes))
        
        for r, vals in sorted_classes[:K]:
            # Simulate removal
            new_points = points.difference(vals)
            
            # Evaluate next prime
            # Find next prime q > p such that q <= len(new_points)
            # Since we are iterating primes, the next relevant prime is primes[idx+1]
            # But only if primes[idx+1] <= len(new_points)
            
            if idx + 1 < len(primes):
                q = primes[idx+1]
                if q <= len(new_points):
                    # Check if new_points covers all residues mod q
                    q_classes = _get_residue_classes(new_points, q)
                    if len(q_classes) == q:
                        # We will have to remove something for q too
                        # Estimate cost: size of smallest q-class
                        min_q_size = min(len(v) for v in q_classes.values())
                        # Total removed so far + estimated future removal
                        # We want to minimize total removal.
                        # Current removal: len(vals)
                        # Future removal: min_q_size
                        # Score = - (len(vals) + min_q_size)
                        score = -(len(vals) + min_q_size)
                    else:
                        # No removal needed for q
                        score = -len(vals)
                else:
                    # q is not active
                    score = -len(vals)
            else:
                score = -len(vals)
                
            if score > best_score:
                best_score = score
                best_points = new_points
        
        if best_points is not None:
            points = best_points
            
        idx += 1
        
    return list(points)
