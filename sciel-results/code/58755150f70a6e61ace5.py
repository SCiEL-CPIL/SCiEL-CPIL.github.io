import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_valid(selected, m, r):
        """Check if (m, r) is disjoint from all in selected."""
        for (m2, r2) in selected:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_residue(selected, m):
        """Find any valid residue for modulus m given selected pairs."""
        for r in range(m):
            if is_valid(selected, m, r):
                return r
        return None
    
    def score(selected):
        return sum(1.0 / m for m, r in selected)
    
    # Beam search
    beam_width = 50
    beam = [[]]  # Start with empty set
    
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
        
        new_beam = []
        for solution in beam:
            # Option 1: skip this modulus
            new_beam.append(solution)
            
            # Option 2: try to add this modulus with various residues
            # Try a few residues to keep diversity
            added = False
            for r in range(m):
                if is_valid(solution, m, r):
                    new_solution = solution + [(m, r)]
                    new_beam.append(new_solution)
                    added = True
                    break  # Only add one per solution per m for now
            
            if not added:
                pass  # Already added the skip option
        
        # Keep top beam_width by score
        new_beam.sort(key=lambda x: score(x), reverse=True)
        beam = new_beam[:beam_width]
    
    # Return best solution
    best = max(beam, key=lambda x: score(x))
    return [[m, r] for m, r in best]
