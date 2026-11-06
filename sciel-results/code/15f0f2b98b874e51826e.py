<<<<<<< SEARCH
def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    Uses conflict-aware heuristic: remove the class with minimum new removals
    (i.e., maximum overlap with already-removed elements).
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Track which elements have been removed
    removed = set()
    
    for p in small_primes:
        if p > len(points):
            break
        
        # Check if all p residue classes are present
        class_counts = {}
        for v in points:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        if len(class_counts) < p:
            continue  # Already missing at least one class
        
        # All p classes present. Choose which class to remove.
        # Heuristic: remove the class whose elements have the most overlap
        # with already-removed elements (minimizes new removals)
        
        best_r = None
        best_new_removals = float('inf')
        
        for r in range(p):
            if r not in class_counts:
                continue
            # Count how many elements in this class are NOT already removed
            # (i.e., would be newly removed)
            new_removals = 0
            for v in points:
                if v % p == r and v not in removed:
                    new_removals += 1
            
            if new_removals < best_new_removals:
                best_new_removals = new_removals
                best_r = r
        
        # Remove the chosen class
        if best_r is not None:
            for v in list(points):
                if v % p == best_r:
                    points.discard(v)
                    removed.add(v)
    
    return sorted(points)
=======
def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    Uses lookahead heuristic: for each candidate residue class to remove,
    estimate how many future primes would still be fully covered (bad) vs
    how many would gain slack (good). Prefer removals that create slack
    for future primes.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Track which elements have been removed
    removed = set()
    
    for idx, p in enumerate(small_primes):
        if p > len(points):
            break
        
        # Check if all p residue classes are present
        class_counts = {}
        for v in points:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        if len(class_counts) < p:
            continue  # Already missing at least one class
        
        # All p classes present. Choose which class to remove.
        # Lookahead: for each candidate r, estimate impact on future primes.
        best_r = None
        best_score = float('inf')
        
        # Pre-compute: for each future prime q, which residue classes are present
        future_primes = small_primes[idx+1:]
        
        for r in range(p):
            if r not in class_counts:
                continue
            
            # Count new removals
            new_removals = sum(1 for v in points if v % p == r and v not in removed)
            
            # Lookahead score: for each future prime q, check if removing
            # class r mod p would help (create a missing class) or hurt
            lookahead_score = 0
            for q in future_primes:
                if q > len(points):
                    break
                # Check if q is currently fully covered
                q_classes = set(v % q for v in points)
                if len(q_classes) < q:
                    continue  # q already has slack
                
                # If we remove elements with v%p==r, does q lose full coverage?
                # i.e., is there some s mod q where ALL elements with v%q==s
                # have v%p==r?
                for s in range(q):
                    all_removed = True
                    for v in points:
                        if v % q == s and v % p != r:
                            all_removed = False
                            break
                    if all_removed:
                        # Removing class r saves prime q
                        lookahead_score -= 2
                        break
                else:
                    # q remains fully covered; small penalty
                    lookahead_score += 0.1
            
            total_score = new_removals + lookahead_score
            
            if total_score < best_score:
                best_score = total_score
                best_r = r
        
        # Remove the chosen class
        if best_r is not None:
            for v in list(points):
                if v % p == best_r:
                    points.discard(v)
                    removed.add(v)
    
    return sorted(points)
>>>>>>> REPLACE

