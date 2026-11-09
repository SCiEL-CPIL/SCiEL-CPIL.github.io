import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def score(classes):
        return sum(1.0 / m for m, r in classes)
    
    def is_compatible(m, r, chosen):
        for (m2, r2) in chosen:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def find_valid_residue(m, chosen):
        """Find a residue r for modulus m that's compatible with all chosen."""
        for r in range(m):
            if is_compatible(m, r, chosen):
                return r
        return None
    
    def greedy_increasing():
        """Greedy: process moduli in increasing order, find valid residue for each."""
        chosen = []
        for m in range(20, 201):
            if time.time() > deadline:
                break
            r = find_valid_residue(m, chosen)
            if r is not None:
                chosen.append((m, r))
        return chosen
    
    def greedy_decreasing():
        """Greedy: process moduli in decreasing order."""
        chosen = []
        for m in range(200, 19, -1):
            if time.time() > deadline:
                break
            r = find_valid_residue(m, chosen)
            if r is not None:
                chosen.append((m, r))
        return chosen
    
    def local_search_improve(classes, max_iterations=500):
        """Try to improve by adding moduli or swapping."""
        chosen = list(classes)
        chosen_set = set(m for m, r in chosen)
        current_score = score(chosen)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a missing modulus
            for m in range(20, 201):
                if m in chosen_set:
                    continue
                r = find_valid_residue(m, chosen)
                if r is not None:
                    new_classes = chosen + [(m, r)]
                    new_score = score(new_classes)
                    if new_score > current_score:
                        chosen = new_classes
                        chosen_set.add(m)
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i, (m_remove, r_remove) in enumerate(chosen):
                if time.time() > deadline:
                    return chosen
                chosen_without = chosen[:i] + chosen[i+1:]
                for m_add in range(20, 201):
                    if m_add in chosen_set or m_add == m_remove:
                        continue
                    r_add = find_valid_residue(m_add, chosen_without)
                    if r_add is not None:
                        new_classes = chosen_without + [(m_add, r_add)]
                        new_score = score(new_classes)
                        if new_score > current_score:
                            chosen = new_classes
                            chosen_set.remove(m_remove)
                            chosen_set.add(m_add)
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        return chosen
    
    # Strategy 1: Greedy increasing
    best = greedy_increasing()
    best_score = score(best)
    
    # Strategy 2: Greedy decreasing
    if time.time() < deadline - 10:
        candidate = greedy_decreasing()
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    # Strategy 3: Local search improvement on best
    if time.time() < deadline - 20:
        improved = local_search_improve(best, max_iterations=1000)
        if score(improved) > best_score:
            best = improved
            best_score = score(best)
    
    # Strategy 4: Try different starting points with local search
    if time.time() < deadline - 30:
        # Start with just small moduli and build up
        small_start = []
        for m in range(20, 40):
            r = find_valid_residue(m, small_start)
            if r is not None:
                small_start.append((m, r))
        
        candidate = local_search_improve(small_start, max_iterations=500)
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    return [[m, r] for m, r in best]
