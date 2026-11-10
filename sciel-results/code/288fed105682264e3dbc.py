"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Generate all odd divisors of base greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # increasing order

    # Precompute for each divisor m, the list of residues covered by each class r
    # For speed, we'll use a different approach

    # Greedy construction with local search
    def score(classes):
        """Compute number of uncovered residues modulo base."""
        covered = bytearray(base)
        for m, r in classes:
            for x in range(r, base, m):
                covered[x] = 1
        return base - sum(covered)

    def greedy_construct():
        """Greedy: moduli in increasing order, each takes the class covering most uncovered residues."""
        covered = bytearray(base)
        classes = []
        for m in divisors:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = sum(1 for x in range(r, base, m) if not covered[x])
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            for x in range(best_r, base, m):
                covered[x] = 1
        return classes, score(classes)

    def local_search(classes, max_iter=500):
        """Local search: try swapping residues for each modulus to improve score."""
        classes = [list(c) for c in classes]
        current_score = score(classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            for i, (m, r) in enumerate(classes):
                # Try all possible residues for this modulus
                best_r = r
                best_score = current_score
                
                # Compute covered without this class
                covered_without = bytearray(base)
                for j, (mj, rj) in enumerate(classes):
                    if j != i:
                        for x in range(rj, base, mj):
                            covered_without[x] = 1
                
                # Try each residue
                for r_new in range(m):
                    # Count how many new residues would be covered
                    new_covered = 0
                    for x in range(r_new, base, m):
                        if not covered_without[x]:
                            new_covered += 1
                    
                    new_score = current_score - new_covered
                    if new_score < best_score:
                        best_score = new_score
                        best_r = r_new
                
                if best_r != r:
                    classes[i][1] = best_r
                    current_score = best_score
                    improved = True
            
            if not improved:
                break
        
        return classes, current_score

    # Initial greedy
    classes, current_score = greedy_construct()
    best_classes = [list(c) for c in classes]
    best_score = current_score

    # Local search
    classes, current_score = local_search(classes)
    if current_score < best_score:
        best_classes = [list(c) for c in classes]
        best_score = current_score

    # Try multiple random restarts with perturbed greedy
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturbed greedy: randomize the order of moduli slightly
        random_divs = list(divisors)
        random.shuffle(random_divs)
        # Keep small moduli first but shuffle larger ones
        random_divs.sort(key=lambda d: (d > 100, random.random()))
        
        covered = bytearray(base)
        classes = []
        for m in random_divs:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = sum(1 for x in range(r, base, m) if not covered[x])
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            for x in range(best_r, base, m):
                covered[x] = 1
        
        classes, current_score = local_search(classes, max_iter=200)
        if current_score < best_score:
            best_classes = [list(c) for c in classes]
            best_score = current_score

    return best_classes
# EVOLVE-BLOCK-END
