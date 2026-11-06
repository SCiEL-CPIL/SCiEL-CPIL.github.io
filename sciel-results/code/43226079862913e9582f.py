import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin (90s limit)
    
    n = 192
    max_size = 48
    
    # Precompute for each element a, the set of residues r such that (b+c) % a == r
    # for all pairs b,c from [1..192], b<c, b,c != a
    # This allows fast checking when adding a new element.
    
    # For each a in [1..192], precompute pair_sums_mod[a] as a dict:
    #   pair_sums_mod[a][r] = list of (b,c) pairs where (b+c) % a == r, b<c, b,c != a, 1<=b,c<=192
    # But storing all pairs is too much. Instead, for each a, store just the count or a set of sums.
    
    # Better: For each a, precompute which sums s (where s = b+c, 2<=s<=382, b,c in [1..192], b<c, b,c!=a)
    # satisfy s % a == 0. But we need to know if ANY such pair exists in our current set.
    
    # Alternative efficient approach:
    # When checking if candidate x can be added to set S:
    # Check 1: For all triples (b,c,d) in S, (b+c+d) % x != 0
    # Check 2: For all a in S, for all pairs (b,c) in S\{a}, (b+c+x) % a != 0
    
    # Precompute triple_sums_mod: for each x in [1..192], for each triple sum s, whether s % x == 0
    # But triples are C(48,3) = 17296 which is manageable per check.
    
    # Let's use a smarter precomputation:
    # For each a in [1..192], precompute a set of "bad pair sums" modulo a.
    # Actually, let's just optimize the can_add function with numpy or better data structures.
    
    def can_add(selected, candidate):
        """Check if candidate can be added to selected without violating constraints."""
        sel = list(selected)
        m = len(sel)
        
        # Check 1: candidate divides sum of any 3 distinct elements in selected
        if m >= 3:
            for i in range(m):
                for j in range(i+1, m):
                    s2 = sel[i] + sel[j]
                    for k in range(j+1, m):
                        if (s2 + sel[k]) % candidate == 0:
                            return False
        
        # Check 2: for each a in selected, a divides (b+c+candidate) for distinct b,c in selected, b,c != a
        for idx_a, a in enumerate(sel):
            # Collect pairs (b,c) from sel where b != a, c != a
            others = [sel[i] for i in range(m) if i != idx_a]
            mo = len(others)
            if mo >= 2:
                for i in range(mo):
                    for j in range(i+1, mo):
                        if (others[i] + others[j] + candidate) % a == 0:
                            return False
        
        return True
    
    # Greedy: try to build a large set
    # Key insight: larger numbers are less likely to divide sums of smaller numbers
    # Also, numbers that are prime or have few divisors are better candidates
    
    best = []
    
    # Try descending order first (largest numbers first)
    for x in range(n, 0, -1):
        if time.time() > deadline - 3.0:
            break
        if can_add(best, x):
            best.append(x)
        if len(best) >= max_size:
            break
    
    current_best = len(best)
    best_set = list(best)
    
    # Try multiple random orderings with different biases
    candidates = list(range(1, n+1))
    
    if time.time() < deadline - 5.0:
        for restart in range(100):
            if time.time() > deadline - 3.0:
                break
            
            # Different ordering strategies
            if restart % 4 == 0:
                random.shuffle(candidates)
            elif restart % 4 == 1:
                # Descending with random perturbation
                candidates = sorted(range(1, n+1), reverse=True)
                random.shuffle(candidates)
            elif restart % 4 == 2:
                # Prioritize larger numbers
                candidates = list(range(n, 0, -1))
                random.shuffle(candidates)
            else:
                # Random but biased toward larger numbers
                candidates = sorted(range(1, n+1), key=lambda x: random.random() + x/n*0.5, reverse=True)
            
            curr = []
            for x in candidates:
                if len(curr) >= max_size:
                    break
                if can_add(curr, x):
                    curr.append(x)
            if len(curr) > current_best:
                current_best = len(curr)
                best_set = curr
    
    # Local search: try to swap elements to improve
    if time.time() < deadline - 5.0 and current_best < max_size:
        best_set = list(best_set)
        
        for iteration in range(500):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(best_set)):
                if time.time() > deadline - 3.0:
                    break
                temp_set = best_set[:i] + best_set[i+1:]
                temp_set_set = set(temp_set)
                
                # Try adding a candidate not in the set, prioritizing larger numbers
                for x in range(n, 0, -1):
                    if x in temp_set_set:
                        continue
                    if can_add(temp_set, x):
                        new_set = temp_set + [x]
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            improved = True
                            break
                        elif len(new_set) == len(best_set):
                            # Same size, but maybe better for future additions
                            if min(new_set) > min(best_set):
                                best_set = new_set
                                improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_set[:max_size]
