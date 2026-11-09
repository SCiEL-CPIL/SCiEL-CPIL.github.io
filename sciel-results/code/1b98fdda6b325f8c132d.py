import time
import math
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Candidate moduli: all integers from 20 to 200
    # We want to maximize sum of 1/m, so prefer smaller moduli
    # Sort by 1/m descending (i.e., m ascending)
    candidates = list(range(20, 201))
    candidates.sort(key=lambda m: -1.0/m)  # larger 1/m first, i.e., smaller m first
    
    best_score = 0.0
    best_classes = []
    
    # Greedy with backtracking
    def is_compatible(placed, m, r):
        """Check if (m, r) is compatible with all placed classes."""
        for m2, r2 in placed:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_residue(placed, m):
        """Find a compatible residue for modulus m."""
        for r in range(m):
            if is_compatible(placed, m, r):
                return r
        return None
    
    def backtrack(placed, remaining, depth=0):
        nonlocal best_score, best_classes
        if time.time() > deadline:
            return
        
        # Calculate current score
        current_score = sum(1.0/m for m, r in placed)
        if len(placed) > len(best_classes) or current_score > best_score:
            best_score = current_score
            best_classes = list(placed)
        
        if not remaining:
            return
        
        # Try to place the next modulus (smallest m first for better score)
        m = remaining[0]
        rest = remaining[1:]
        
        # Try all residues for this modulus
        for r in range(m):
            if is_compatible(placed, m, r):
                new_placed = placed + [(m, r)]
                backtrack(new_placed, rest, depth + 1)
                if time.time() > deadline:
                    return
        
        # Also try skipping this modulus
        backtrack(placed, rest, depth + 1)
    
    # Start with greedy: place moduli in order of smallest m
    # But use backtracking to explore alternatives
    # For efficiency, limit the search depth
    
    # First, try pure greedy
    placed = []
    for m in candidates:
        if time.time() > deadline:
            break
        r = find_residue(placed, m)
        if r is not None:
            placed.append((m, r))
    
    current_score = sum(1.0/m for m, r in placed)
    if current_score > best_score:
        best_score = current_score
        best_classes = list(placed)
    
    # Now try local search / hill climbing to improve
    # Try swapping residues or replacing moduli
    
    def local_search(classes, max_iters=10000):
        nonlocal best_score, best_classes
        current = list(classes)
        current_score = sum(1.0/m for m, r in current)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(len(current)):
                m_i, r_i = current[i]
                # Try changing residue
                for r_new in range(m_i):
                    if r_new == r_i:
                        continue
                    if is_compatible([(m, r) for j, (m, r) in enumerate(current) if j != i], m_i, r_new):
                        new_current = list(current)
                        new_current[i] = (m_i, r_new)
                        # Check full compatibility
                        valid = True
                        for j in range(len(new_current)):
                            if j == i:
                                continue
                            m_j, r_j = new_current[j]
                            g = math.gcd(m_i, m_j)
                            if r_new % g == r_j % g:
                                valid = False
                                break
                        if valid:
                            # Score doesn't change with residue, but this might enable other changes
                            pass
                # Try removing this and adding a different modulus
                for m_new in candidates:
                    if m_new == m_i:
                        continue
                    if any(m == m_new for m, r in current):
                        continue
                    without_i = [(m, r) for j, (m, r) in enumerate(current) if j != i]
                    r_new = find_residue(without_i, m_new)
                    if r_new is not None:
                        new_score = sum(1.0/m for m, r in without_i) + 1.0/m_new
                        if new_score > current_score + 1e-12:
                            new_current = without_i + [(m_new, r_new)]
                            current = new_current
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = list(current)
                            break
                if improved and time.time() > deadline:
                    break
        return current
    
    # Run local search
    if time.time() < deadline:
        improved_classes = local_search(best_classes)
        if sum(1.0/m for m, r in improved_classes) > best_score:
            best_score = sum(1.0/m for m, r in improved_classes)
            best_classes = list(improved_classes)
    
    # Convert to required format
    result = [[m, r] for m, r in best_classes]
    return result
