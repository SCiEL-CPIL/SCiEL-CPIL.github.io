import time
import random
from fractions import Fraction

def construct():
    # Set a fixed seed for reproducibility
    random.seed(42)
    
    # Safety margin: Target < 60s, Kill at 90s. Use 55s.
    deadline = time.time() + 55
    n = 256
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
    def can_add(x, S):
        """Check if x can be added to S without forming a 3-AP."""
        # Case 1: x is first term: x, x+d, x+2d
        max_d1 = (n - x) // 2
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                return False
        
        # Case 2: x is second term: x-d, x, x+d
        max_d2 = min(x - 1, n - x)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S:
                return False
        
        # Case 3: x is third term: x-2d, x-d, x
        max_d3 = x // 2
        for d in range(1, max_d3 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                return False
        
        return True

    def get_conflicts(x, S):
        """Find all elements in S that form a 3-AP with x and another element in S."""
        conflicts = set()
        
        # Case 1: x is first term: x, x+d, x+2d
        max_d1 = (n - x) // 2
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                conflicts.add(x + d)
                conflicts.add(x + 2 * d)
                
        # Case 2: x is second term: x-d, x, x+d
        max_d2 = min(x - 1, n - x)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S:
                conflicts.add(x - d)
                conflicts.add(x + d)
                
        # Case 3: x is third term: x-2d, x-d, x
        max_d3 = x // 2
        for d in range(1, max_d3 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                conflicts.add(x - 2 * d)
                conflicts.add(x - d)
                
        return conflicts

    def greedy_construct(order=None):
        if order is None:
            order = list(range(1, n + 1))
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        S = S.copy()
        current_weight = sum(weights[x - 1] for x in S)
        
        while time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Try to add any element
            for x in range(1, n + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    current_weight += weights[x - 1]
                    improved = True
            
            if time.time() > deadline - 2:
                break
                
            # Phase 2: Try swaps (remove conflicts, add x)
            for x in range(1, n + 1):
                if x in S:
                    continue
                conflicts = get_conflicts(x, S)
                if not conflicts:
                    continue
                
                # Check if removing all conflicts allows adding x
                # And if the weight gain is positive
                conflict_weight = sum(weights[c - 1] for c in conflicts)
                if weights[x - 1] > conflict_weight:
                    # S - conflicts + {x} is valid
                    S -= conflicts
                    S.add(x)
                    current_weight += weights[x - 1] - conflict_weight
                    improved = True
            
            if time.time() > deadline - 2:
                break

            # Phase 3: Random Kick to escape local optima
            if not improved:
                # Remove a few random elements to allow re-optimization
                # Prefer removing low weight (high index) elements to minimize loss
                if len(S) > 5:
                    # Pick 2-3 random elements
                    remove_count = min(3, max(1, len(S) // 10))
                    if remove_count > 0:
                        # Sort S by weight (descending) to pick from the end (low weight)
                        sorted_S = sorted(S, key=lambda x: weights[x-1])
                        # Pick from the lowest 20% or just random
                        # Random is better for exploration
                        to_remove = random.sample(list(S), remove_count)
                        for x in to_remove:
                            S.remove(x)
                            current_weight -= weights[x - 1]
                        # Force re-evaluation
                        improved = True

        return S, current_weight

    # Initial greedy solution
    best_S = greedy_construct()
    best_weight = sum(weights[x - 1] for x in best_S)
    
    # Run local search
    S, w = local_search(best_S)
    if w > best_weight:
        best_S = S
        best_weight = w
        
    # Try a few random restarts if time permits
    while time.time() < deadline - 5:
        # Random initialization: shuffle order
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_construct(order)
        
        # Local search on this random start
        S, w = local_search(S)
        
        if w > best_weight:
            best_S = S
            best_weight = w

    return sorted(best_S)
