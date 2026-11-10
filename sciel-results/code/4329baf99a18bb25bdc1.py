"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N = 180
    
    def has_triple_in_set(S, a, b, c):
        """Check if {a,b,c} all in S and form a forbidden triple."""
        if a == b or a == c or b == c:
            return False
        # 1/a = 1/b + 1/c => a = bc/(b+c) => a(b+c) = bc => abc + a²c = bc... 
        # Actually: 1/a = 1/b + 1/c => bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a²
        # Check all permutations
        for x, y, z in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
            if (y - x) * (z - x) == x * x and y != z:
                return True
        return False
    
    def count_conflicts(S, elem):
        """Count how many forbidden triples would be formed if elem is in S."""
        count = 0
        S_set = set(S)
        for b in S_set:
            if b == elem:
                continue
            for c in S_set:
                if c == elem or c == b:
                    continue
                # Check if {elem, b, c} forms a forbidden triple
                for x, y, z in [(elem,b,c),(elem,c,b),(b,elem,c),(b,c,elem),(c,elem,b),(c,b,elem)]:
                    if (y - x) * (z - x) == x * x and y != z:
                        count += 1
                        break
        return count
    
    def is_safe(S, elem):
        """Check if adding elem to S creates any forbidden triple."""
        S_set = set(S)
        for b in S_set:
            if b == elem:
                continue
            for c in S_set:
                if c == elem or c == b:
                    continue
                # Check if {elem, b, c} forms a forbidden triple
                for x, y, z in [(elem,b,c),(elem,c,b),(b,elem,c),(b,c,elem),(c,elem,b),(c,b,elem)]:
                    if (y - x) * (z - x) == x * x and y != z:
                        return False
        return True
    
    # Start with the known good set: [91, 180]
    S = set(range(91, 181))
    
    # Try to add each smaller number if it's safe
    for x in range(1, 91):
        if time.time() > deadline:
            break
        if is_safe(S, x):
            S.add(x)
    
    # Local search: try to swap or add/remove
    # Try adding numbers that were excluded
    for x in range(1, 91):
        if time.time() > deadline:
            break
        if x not in S and is_safe(S, x):
            S.add(x)
    
    # Now try to remove some elements to add more
    # For each element in S, try removing it and see if we can add more
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_elem in sorted(S):
            if time.time() > deadline:
                break
            S_temp = S - {remove_elem}
            added = 0
            for x in range(1, 181):
                if time.time() > deadline:
                    break
                if x not in S_temp and is_safe(S_temp, x):
                    S_temp.add(x)
                    added += 1
            if len(S_temp) > len(S):
                S = S_temp
                improved = True
    
    # Final local search: random restarts
    best_S = S
    best_size = len(S)
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation
        S_curr = set(best_S)
        # Remove 3-5 random elements
        to_remove = random.sample(list(S_curr), min(5, len(S_curr)))
        for x in to_remove:
            S_curr.discard(x)
        
        # Greedily add back
        for x in range(1, 181):
            if time.time() > deadline - 5:
                break
            if x not in S_curr and is_safe(S_curr, x):
                S_curr.add(x)
        
        if len(S_curr) > best_size:
            best_S = S_curr
            best_size = len(S_curr)
    
    return list(best_S)
# EVOLVE-BLOCK-END
