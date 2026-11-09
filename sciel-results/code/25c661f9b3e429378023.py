import math
import random
import time

def construct():
    max_mod = 128
    
    # The key insight: If gcd(m1, m2) == 1, the classes ALWAYS intersect.
    # Thus, all selected moduli must be pairwise non-coprime.
    # The largest set of pairwise non-coprime numbers in [2, 128] is the set of even numbers.
    # Size: 64.
    # We attempt to find residues for this set.
    
    target_moduli = [m for m in range(2, max_mod + 1) if m % 2 == 0]
    
    # If we can't solve for all 64, we might need to drop some.
    # But let's try to solve for all 64 first.
    
    def is_compatible(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def solve_csp(moduli):
        """
        Randomized backtracking to find residues for the given moduli.
        """
        if not moduli:
            return {}
        
        # Heuristic: Order moduli by how constrained they are.
        # A modulus m is constrained by g = gcd(m, m').
        # Smaller g means stricter constraint (more residues are invalid).
        # Actually, larger g means fewer invalid residues?
        # Condition: r != r' mod g.
        # Number of invalid residues for a given r' is 1 (if we fix r').
        # Wait, for a pair (m, m'), if g=2, half the residues are invalid relative to a fixed r'.
        # If g=4, 1/4 are invalid.
        # So smaller g is "stricter" in terms of probability of conflict?
        # No, if g=2, you have 50% chance of conflict. If g=4, 25%.
        # So variables with many small-g neighbors are more constrained.
        
        # Let's just use a random order or sorted by value.
        # Sorting by value might help as smaller moduli have fewer residues to try?
        # No, larger moduli have more residues, so more options.
        # Smaller moduli have fewer options, so assign them first?
        
        sorted_moduli = sorted(moduli)
        
        assignment = {}
        
        def backtrack(idx):
            if idx == len(sorted_moduli):
                return True
            
            m = sorted_moduli[idx]
            
            # Try residues in random order to escape local minima
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                valid = True
                for m2, r2 in assignment.items():
                    if not is_compatible(r, m, r2, m2):
                        valid = False
                        break
                
                if valid:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False

        # Try multiple random restarts
        for _ in range(50):
            random.shuffle(sorted_moduli) # Re-shuffle order each restart? 
            # Actually, keeping a fixed order but randomizing residues is often better.
            # Let's keep sorted order but randomize residues inside.
            if backtrack(0):
                return assignment
        return None

    # Time budget
    start_time = time.time()
    timeout = 50
    
    assignment = solve_csp(target_moduli)
    
    if assignment is None:
        # If 64 is too hard, try a subset, e.g. multiples of 4
        # Or just return a small valid set to be safe
        # But 64 should be solvable.
        # Let's try a smaller subset if it fails.
        # For now, if it fails, we return an empty list? No, must be nonempty.
        # Let's try just the first 10 even numbers.
        small_moduli = target_moduli[:10]
        assignment = solve_csp(small_moduli)
        
        if assignment is None:
            # Absolute fallback: single class
            return [[0, 2]]

    # Convert to result
    result = [[assignment[m], m] for m in assignment.keys()]
    return result
