import math
import random

def construct():
    # New approach: Multi-start Randomized Greedy with optimized residue conflict checking.
    # The key optimization is using a boolean array to mark forbidden residues for a candidate modulus m,
    # based on the existing set of classes. This reduces the inner loop complexity significantly.
    
    candidates = list(range(20, 201))
    best_classes = []
    best_score = 0.0
    
    def greedy_solve(order):
        classes = []
        current_score = 0.0
        
        for m in order:
            # Mark forbidden residues for m
            forbidden = [False] * m
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                # Residues r such that r % g == r_j % g
                start = r_j % g
                for r in range(start, m, g):
                    forbidden[r] = True
            
            # Find the first valid residue
            valid_res = -1
            for r in range(m):
                if not forbidden[r]:
                    valid_res = r
                    break
            
            if valid_res != -1:
                classes.append([m, valid_res])
                current_score += 1.0 / m
        
        return classes, current_score

    # 1. Try standard greedy: sorted by m ascending (highest 1/m first)
    sorted_asc = sorted(candidates)
    classes, score = greedy_solve(sorted_asc)
    if score > best_score:
        best_score = score
        best_classes = classes

    # 2. Try sorted by m descending
    sorted_desc = sorted(candidates, reverse=True)
    classes, score = greedy_solve(sorted_desc)
    if score > best_score:
        best_score = score
        best_classes = classes

    # 3. Try randomized restarts
    # We bias the random ordering towards smaller moduli to prioritize high-value terms
    # but allow some flexibility.
    for _ in range(200):
        # Create a random order
        # To ensure we don't waste time, we can just shuffle
        order = candidates[:]
        random.shuffle(order)
        
        # Or, a better heuristic: sort by a random key + m?
        # Let's stick to simple shuffle for diversity, but maybe weight smaller m?
        # Actually, just shuffling is a good baseline for multi-start.
        
        classes, score = greedy_solve(order)
        if score > best_score:
            best_score = score
            best_classes = classes

    # 4. Try a "greedy with lookahead" or specific heuristic orders
    # Sort by m, but break ties randomly? No, m are distinct.
    # Let's try sorting by m * random_factor?
    for _ in range(50):
        weights = [random.random() for _ in candidates]
        # Sort by a combination of m (ascending is good) and random noise
        # We want small m to be first, but allow some larger m to be earlier if it helps packing?
        # Actually, small m are always better for score. The issue is packing.
        # Let's sort by m, but with a small probability of swapping adjacent elements?
        # Simpler: Just do more shuffles.
        order = candidates[:]
        random.shuffle(order)
        classes, score = greedy_solve(order)
        if score > best_score:
            best_score = score
            best_classes = classes

    return best_classes
