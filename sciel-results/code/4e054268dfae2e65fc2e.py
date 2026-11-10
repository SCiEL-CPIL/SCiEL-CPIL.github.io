def _make_admissible(points, primes, deadline):
    """Make the set admissible by removing points."""
    point_set = set(points)
    # For each prime, track residue counts
    residue_counts = {}
    for p in primes:
        counts = [0] * p
        for v in point_set:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        k = len(point_set)
        # Check which primes are violated (all classes covered)
        for p in primes:
            if p > k:
                break
            counts = residue_counts[p]
            if all(c > 0 for c in counts):
                # Need to empty one class - pick the least populated
                min_class = min(range(p), key=lambda r: counts[r])
                # Remove all points in that class
                to_remove = [v for v in point_set if v % p == min_class]
                for v in to_remove:
                    point_set.remove(v)
                    for q in primes:
                        if q > len(point_set) + len(to_remove):
                            break
                        residue_counts[q][v % q] -= 1
                changed = True
                break  # Restart check since size changed
    return list(point_set)
