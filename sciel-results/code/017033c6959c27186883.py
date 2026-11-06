import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    max_time = 55.0  # Safety margin before 60s
    
    # Candidate moduli: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    
    best_count = 0
    best_solution = []
    
    def is_compatible(r_new, m_new, existing):
        for r_old, m_old in existing:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def solve_greedy_randomized(moduli_order):
        current = []
        # We will maintain a list of (residue, modulus)
        # To speed up, we can just check against current list
        
        # Dynamic ordering: 
        # We pick the next modulus from the remaining ones that has the smallest domain.
        # Initially, domain size is m.
        # As we assign, we can pre-filter? 
        # For simplicity, we just iterate through the shuffled list.
        
        for m in moduli_order:
            residues = list(range(m))
            random.shuffle(residues)
            found = False
            for r in residues:
                if is_compatible(r, m, current):
                    current.append((r, m))
                    found = True
                    break
            if not found:
                # If we can't assign, we stop this attempt
                # Or we could backtrack, but that's expensive.
                # Since we are maximizing, we just return what we have.
                pass
        return current

    # Run multiple random restarts
    attempts = 0
    while time.time() - start_time < max_time:
        attempts += 1
        random.shuffle(moduli)
        current = solve_greedy_randomized(moduli)
        
        if len(current) > best_count:
            best_count = len(current)
            best_solution = current
            
        # Early stop if we found a perfect solution (all 64 moduli)
        if best_count == len(moduli):
            break
            
    # Convert to list of lists as required
    return [[r, m] for r, m in best_solution]
# EVOLVE-BLOCK-END
