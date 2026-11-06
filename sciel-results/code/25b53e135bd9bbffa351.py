import time
import math
import random

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = 0.0
    best_classes = []
    
    # Precompute GCDs for moduli in [20, 200]
    moduli = list(range(20, 201))
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def check_disjoint(m, r, classes):
        """
        Check if (m, r) is disjoint from all classes in 'classes'.
        classes is a list of [m, r] pairs.
        """
        for m2, r2 in classes:
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True

    def greedy_randomized():
        classes = []
        # We iterate moduli in increasing order to prioritize small m (high 1/m)
        # But we randomize the residue choice to explore the space
        for m in moduli:
            # Generate a random permutation of residues for this m
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                if check_disjoint(m, r, classes):
                    classes.append([m, r])
                    found = True
                    break
            # If no residue found, we skip this modulus
            # We don't backtrack in this simple greedy, but multiple restarts handle this
        return classes

    def local_search(classes):
        """
        Try to improve a given set of classes by adding missing ones
        or swapping residues to allow additions.
        """
        current_classes = [c[:] for c in classes]
        current_score = sum(1.0 / m for m, r in current_classes)
        
        # Map of modulus to index in current_classes for quick lookup
        m_to_idx = {m: i for i, (m, r) in enumerate(current_classes)}
        
        improved = True
        while improved:
            improved = False
            # Try to add any modulus not currently present
            for m in moduli:
                if m in m_to_idx:
                    continue
                
                # Try to find a valid residue for m
                # To be deterministic or fast, we can try all residues
                # But to keep it fast, we might just try a few random ones or all
                # Given m <= 200, trying all is fine
                valid_residue = None
                for r in range(m):
                    if check_disjoint(m, r, current_classes):
                        valid_residue = r
                        break
                
                if valid_residue is not None:
                    current_classes.append([m, valid_residue])
                    m_to_idx[m] = len(current_classes) - 1
                    current_score += 1.0 / m
                    improved = True
        
        # Optional: Try to swap residues to enable more additions?
        # This is complex. The "add missing" loop above is usually sufficient 
        # if the initial greedy was good, but greedy might have blocked things.
        # A better local search would involve removing a class to add two others.
        # Given time constraints, we rely on multiple randomized restarts.
        
        return current_classes, current_score

    # Main loop: Multi-start greedy + local improvement
    iterations = 0
    while time.time() < deadline:
        # Generate a random candidate
        candidate = greedy_randomized()
        
        # Apply local search (just adding missing moduli)
        improved_candidate, score = local_search(candidate)
        
        if score > best_score:
            best_score = score
            best_classes = improved_candidate
            
        iterations += 1
        
        # If we found a very high score, we might converge.
        # Max possible score is sum(1/m for m in 20..200) ~ 2.99
        # If we are close to the theoretical max, we can stop earlier or continue to be safe.
        # Theoretical max if all moduli could be included:
        # sum(1/m for m in range(20, 201))
        
    return best_classes

# Theoretical max check
# sum(1/m for m in range(20, 201))
