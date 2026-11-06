"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = 192
    max_k = 48
    k = 3
    
    # Precompute all triples of distinct numbers from 1..192
    # For each triple (a,b,c), the sum is a+b+c
    # A selected number x conflicts with triple (a,b,c) if x divides (a+b+c)
    # and x is not in {a,b,c}
    
    # We'll use a bitset approach for speed
    # Represent the set of selected numbers as a Python int bitmask
    # bit i is set if number (i+1) is selected
    
    # Precompute: for each number x in 1..192, which triples does it conflict with?
    # Actually, let's think differently.
    # For a given set S, x in S is valid if for all triples (a,b,c) with a,b,c in S\{x}, 
    # x does NOT divide a+b+c.
    
    # Alternative: precompute for each pair (a,b) the sums a+b, and for each c, the sum a+b+c.
    # This is getting complex. Let's use a smarter incremental approach.
    
    # Key insight: when we add a new number x to a set S, we need to check:
    # 1. x does not divide the sum of any 3 elements in S
    # 2. For each element y in S, y does not divide the sum of any 3 elements in (S\{y}) ∪ {x}
    #    which means: y does not divide the sum of any triple that includes x and two elements from S\{y}
    
    # Precompute for each x, the set of triples (a,b,c) where x divides a+b+c and a,b,c are distinct and none is x
    # Actually, let's precompute conflicts differently.
    
    # For each number x, precompute the set of pairs (a,b) such that for some c, 
    # x divides a+b+c. This is still complex.
    
    # Let's use a different strategy: 
    # Precompute for each x and each pair (a,b) with a<b, the set of c such that x | (a+b+c)
    # Then when checking if x can be added to S, we check if there exist a,b,c in S with x | (a+b+c)
    
    # Actually, let's just use the O(n^2) incremental check with local search as suggested by the knowledge.
    
    def can_add(x, selected):
        """Check if x can be added to selected without violating constraints."""
        sel = list(selected)
        # Check x against all triples in selected
        for a, b, c in combinations(sel, 3):
            if (a + b + c) % x == 0:
                return False
        # Check each y in selected against triples that include x
        for y in sel:
            others = [v for v in sel if v != y]
            # Triples from others + {x} that include x and 2 from others
            for a, b in combinations(others, 2):
                if (a + b + x) % y == 0:
                    return False
        return True
    
    def is_valid(selected):
        """Check if the entire set is valid."""
        sel = list(selected)
        for x in sel:
            others = [v for v in sel if v != x]
            for a, b, c in combinations(others, 3):
                if (a + b + c) % x == 0:
                    return False
        return True
    
    def greedy_descending():
        """Greedy: try numbers from 192 down to 1."""
        selected = []
        for x in range(192, 0, -1):
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def greedy_ascending():
        """Greedy: try numbers from 1 up to 192."""
        selected = []
        for x in range(1, 193):
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    def local_search(initial):
        """Local search: try to add or swap elements to improve the set."""
        selected = set(initial)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add a number
            for x in range(1, 193):
                if x not in selected:
                    if can_add(x, selected):
                        selected.add(x)
                        improved = True
                        break
            # Try to swap: remove one, add another
            if not improved:
                for y in list(selected):
                    for x in range(1, 193):
                        if x not in selected:
                            new_set = (selected - {y}) | {x}
                            if is_valid(new_set) and len(new_set) >= len(selected):
                                # Only accept if it's the same size or larger
                                # For same size, we might not improve, so skip unless larger
                                if len(new_set) > len(selected):
                                    selected = new_set
                                    improved = True
                                    break
                    if improved:
                        break
        return selected
    
    # Start with greedy descending
    best = greedy_descending()
    best_size = len(best)
    
    # Try local search from the greedy result
    if time.time() < deadline:
        ls_result = local_search(best)
        if len(ls_result) > best_size:
            best = ls_result
            best_size = len(ls_result)
    
    # Try greedy ascending
    if time.time() < deadline:
        asc = greedy_ascending()
        if len(asc) > best_size:
            best = asc
            best_size = len(asc)
        if time.time() < deadline:
            ls_result = local_search(asc)
            if len(ls_result) > best_size:
                best = ls_result
                best_size = len(ls_result)
    
    # Try random restarts with local search
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            # Random subset as starting point
            random.seed(_)
            start_set = set()
            for x in range(1, 193):
                if random.random() < 0.3:
                    start_set.add(x)
            # Make it valid by removing conflicting elements
            valid_start = []
            for x in sorted(start_set):
                if can_add(x, valid_start):
                    valid_start.append(x)
            ls_result = local_search(valid_start)
            if len(ls_result) > best_size:
                best = ls_result
                best_size = len(ls_result)
    
    # Try a few more targeted strategies
    if time.time() < deadline:
        # Greedy with random order
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random.seed(100 + _)
            order = list(range(1, 193))
            random.shuffle(order)
            selected = []
            for x in order:
                if can_add(x, selected):
                    selected.append(x)
            ls_result = local_search(selected)
            if len(ls_result) > best_size:
                best = ls_result
                best_size = len(ls_result)
    
    # Final verification
    best_list = list(best)
    if not is_valid(best_list):
        # Fallback to greedy descending
        best_list = greedy_descending()
    
    return best_list
# EVOLVE-BLOCK-END
