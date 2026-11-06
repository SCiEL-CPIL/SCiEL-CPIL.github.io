"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending - large moduli first to establish sparse baseline
    divisors_desc = sorted(divisors, reverse=True)
    
    # We'll use a greedy approach but with a twist:
    # First pass: pick classes for large moduli that cover many uncovered residues
    # Second pass: local search to improve coverage
    
    import numpy as np
    
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_moduli = set()
    
    # Precompute for each divisor m, the number of residues it covers
    # and which residues
    # For efficiency, we'll process moduli in descending order
    
    # First greedy pass: large moduli first
    for m in divisors_desc:
        if m in used_moduli:
            continue
        
        # Find residue r that maximizes marginal gain
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count how many uncovered residues this class would cover
            indices = np.arange(r, base, m)
            gain = np.sum(covered[indices] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
            # Early termination if we cover everything new
            if best_gain == len(indices):
                break
        
        if best_gain > 0:
            classes.append([m, best_r])
            used_moduli.add(m)
            indices = np.arange(best_r, base, m)
            covered[indices] = 1
    
    # Now do local search: try to improve by swapping residues or adding/removing classes
    # We have all divisors available once each, so we can only change residues, not add/remove moduli
    # Actually, re-reading: "With every divisor of 51975 available once" - so we can use each divisor at most once
    # The seed uses all divisors. Let's check if using all is optimal or if we should skip some.
    
    # Let's try a different approach: use all divisors but optimize residues via local search
    # Start with the greedy solution above (which used all divisors that had positive gain)
    
    # Check how many are covered
    uncovered_count = np.sum(covered == 0)
    
    # Local search: for each class, try changing its residue to see if it improves coverage
    # We'll do multiple passes
    for iteration in range(50):
        improved = False
        for i, (m, r) in enumerate(classes):
            # Try all possible residues for this modulus
            # First, temporarily remove this class
            old_indices = np.arange(r, base, m)
            old_covered = covered[old_indices].copy()
            covered[old_indices] = 0
            
            # Find best new residue
            best_r = r
            best_uncovered = np.sum(covered == 0)
            
            for new_r in range(m):
                new_indices = np.arange(new_r, base, m)
                # Compute what coverage would look like
                # New covered = current covered OR new class
                temp_covered = covered.copy()
                temp_covered[new_indices] = 1
                new_uncovered = np.sum(temp_covered == 0)
                
                if new_uncovered < best_uncovered:
                    best_uncovered = new_uncovered
                    best_r = new_r
            
            # Restore and apply best
            covered[old_indices] = old_covered
            if best_r != r:
                covered[old_indices] = 0
                new_indices = np.arange(best_r, base, m)
                covered[new_indices] = 1
                classes[i] = [m, best_r]
                improved = True
        
        if not improved:
            break
    
    # Final count
    uncovered_count = np.sum(covered == 0)
    
    return classes
# EVOLVE-BLOCK-END
