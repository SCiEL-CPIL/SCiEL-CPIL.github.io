import random
import itertools
from collections import Counter

def construct():
    def count_collisions(s):
        """Count the number of duplicate unordered pair sums (including diagonals) in set s."""
        s = sorted(s)
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        c = Counter(sums)
        return sum(v - 1 for v in c.values() if v > 1)
    
    def optimize(s):
        """Greedily swap elements to maximize collisions with multiple passes."""
        s = list(s)
        n = len(s)
        for iteration in range(500):
            improved = False
            current_score = count_collisions(s)
            # Try all positions and all possible values
            for i in range(n):
                best_swap = None
                best_score = current_score
                for new_val in range(129):
                    if new_val == s[i]:
                        continue
                    if new_val in s:
                        continue
                    old_val = s[i]
                    s[i] = new_val
                    score = count_collisions(s)
                    s[i] = old_val
                    if score > best_score:
                        best_score = score
                        best_swap = new_val
                if best_swap is not None:
                    s[i] = best_swap
                    current_score = best_score
                    improved = True
            if not improved:
                break
        return s
    
    def optimize_with_restarts(s, n_restarts=10):
        """Run optimization with random perturbations to escape local optima."""
        best_set = list(s)
        best_score = count_collisions(s)
        
        for r in range(n_restarts):
            # Perturb: randomly change 1-3 elements
            perturbed = list(s)
            n_changes = random.randint(1, 3)
            for _ in range(n_changes):
                idx = random.randint(0, len(perturbed) - 1)
                new_val = random.randint(0, 128)
                attempts = 0
                while new_val in perturbed and attempts < 50:
                    new_val = random.randint(0, 128)
                    attempts += 1
                if new_val not in perturbed:
                    perturbed[idx] = new_val
            
            # Optimize the perturbed set
            optimized = optimize(perturbed)
            score = count_collisions(optimized)
            if score > best_score:
                best_score = score
                best_set = list(optimized)
        
        return best_set
    
    # Start with several random initializations and structured ones
    random.seed(42)
    best_set = None
    best_collision = -1
    
    # Generate diverse initial sets
    initial_sets = []
    
    # Random sets
    for trial in range(30):
        s = random.sample(range(129), 12)
        initial_sets.append(s)
    
    # Structured starts designed to create many collisions
    structured_starts = [
        # Arithmetic progressions (many pair sum collisions)
        list(range(0, 129, 11))[:12],
        list(range(0, 129, 10))[:12],
        list(range(0, 129, 9))[:12],
        # Quadratic-like patterns
        [i * i % 129 for i in range(12)],
        # Clustered values
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128],
        # Symmetric around center
        [0, 1, 2, 3, 4, 5, 6, 7, 121, 122, 123, 124],
        # Pairs that sum to same value
        [0, 1, 2, 3, 125, 126, 127, 128, 4, 5, 6, 7],
        # Dense clusters
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        # Geometric-like
        [1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 9, 17],
    ]
    
    for start in structured_starts:
        if len(set(start)) == 12 and all(0 <= x <= 128 for x in start):
            initial_sets.append(list(start))
    
    # Optimize each initial set
    for s in initial_sets:
        optimized = optimize_with_restarts(s, n_restarts=5)
        col = count_collisions(optimized)
        if col > best_collision:
            best_collision = col
            best_set = list(optimized)
    
    # Final verification
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return list(best_set)
