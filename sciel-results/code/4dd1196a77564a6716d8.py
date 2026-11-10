import numpy as np

def construct():
    base = 51975
    # Get all divisors of base
    divisors = [d for d in range(1, base + 1) if base % d == 0]
    # Filter m > 1 and sort descending
    moduli = sorted([d for d in divisors if d > 1], reverse=True)
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    
    for m in moduli:
        # For each possible residue r, count how many residues are already covered
        # We want to pick the r with the fewest already covered residues (maximizing new coverage)
        
        # Vectorized approach:
        # covered[r::m] gives the coverage status for the class [m, r]
        # We need to find r that minimizes sum(covered[r::m])
        
        # Efficient way:
        # Create an array of sums for all r
        # This is tricky to vectorize directly without loops over r because the slices are different.
        # However, m is at most 51975. The loop over r is at most 51975 iterations.
        # Inside, sum(covered[r::m]) is O(base/m).
        # Total work per m is O(base).
        # Total work is O(num_moduli * base) = 47 * 51975 ~ 2.4M ops. This is fast.
        
        best_r = 0
        min_covered_count = base # Max possible covered
        
        for r in range(m):
            # Count already covered
            cnt = covered[r::m].sum()
            if cnt < min_covered_count:
                min_covered_count = cnt
                best_r = r
                
        classes.append([m, best_r])
        covered[best_r::m] = 1
        
    return classes
