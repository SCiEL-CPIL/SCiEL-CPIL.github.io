"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_forbidden(N=180):
    """
    Precompute for each pair (b, c) with b != c, the value a = b*c/(b+c)
    such that 1/a = 1/b + 1/c.
    We need a, b, c all distinct and in [1, N].
    For a given a, the forbidden pairs (b, c) are those where b*c % (b+c) == 0
    and a = b*c // (b+c), with a != b, a != c.
    
    We'll build: for each a, a set of (b, c) pairs that would form a triple with a.
    Actually, more useful: for each element a, which other elements can't be in the set
    together with a? That's complex. 
    
    Better approach: For each a in [1, N], precompute the set of pairs (b, c) 
    such that 1/a = 1/b + 1/c, i.e., b*c = a*(b+c), i.e., (b-a)*(c-a) = a^2.
    
    So for a given a, b-a and c-a are divisors of a^2. Let d = b-a, then c-a = a^2/d.
    So b = a + d, c = a + a^2/d, where d divides a^2 and d != 0.
    Also need b != c, b != a, c != a, and 1 <= b, c <= N.
    
    For checking validity of a set S: for each a in S, check if there exist b, c in S
    with b != a, c != a, b != c, and (b-a)*(c-a) = a^2.
    """
    # For each a, store a dict: mapping from b to c (or just a set of (b,c) pairs)
    # Actually, for fast checking: for each a, store a set of valid b values (b != a)
    # such that there exists c in [1,N] with c != a, c != b, and (b-a)*(c-a) = a^2.
    # But we need to check if both b AND c are in S.
    
    # Let's precompute for each a: a list of (b, c) pairs where b < c, b != a, c != a,
    # 1 <= b, c <= N, and (b-a)*(c-a) = a^2.
    
    forbidden_pairs = [set() for _ in range(N + 1)]  # forbidden_pairs[a] = set of (b,c) with b<c
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d of a^2
        # d can be positive or negative
        # b = a + d, c = a + a2/d
        # Need 1 <= b <= N, 1 <= c <= N, b != a, c != a, b != c
        # d divides a^2, d != 0
        # Let's iterate over divisors
        for d in range(1, int(a2 ** 0.5) + 1):
            if a2 % d != 0:
                continue
            # d and a2//d are both divisors
            for dd in (d, a2 // d):
                if dd == 0:
                    continue
                b = a + dd
                c = a + a2 // dd
                # Also check negative divisors
                for sign in (1, -1):
                    b1 = a + sign * dd
                    c1 = a + sign * (a2 // dd)
                    # Ensure b1 != a, c1 != a, b1 != c1, 1 <= b1, c1 <= N
                    if b1 == a or c1 == a or b1 == c1:
                        continue
                    if 1 <= b1 <= N and 1 <= c1 <= N:
                        # Store the pair with b < c for consistency
                        if b1 < c1:
                            forbidden_pairs[a].add((b1, c1))
                        else:
                            forbidden_pairs[a].add((c1, b1))
    
    return forbidden_pairs

def _is_valid(S, forbidden_pairs):
    """Check if set S has no three distinct elements a, b, c with 1/a = 1/b + 1/c."""
    S_set = set(S)
    for a in S:
        for (b, c) in forbidden_pairs[a]:
            if b in S_set and c in S_set:
                return False
    return True

def _count_violations(S, forbidden_pairs):
    """Count the number of violating triples in S."""
    S_set = set(S)
    count = 0
    for a in S:
        for (b, c) in forbidden_pairs[a]:
            if b in S_set and c in S_set:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin under 90s kill
    
    N = 180
    rng = random.Random(42)
    
    # Precompute forbidden pairs
    forbidden_pairs = _precompute_forbidden(N)
    
    # Start with the baseline: [91, 180]
    best = set(range(91, N + 1))
    best_size = len(best)
    
    # Phase 1: Try to add elements greedily from below 91
    # For each element from 1 to 90, try to add it if it doesn't create a violation
    candidates = list(range(1, 91))
    rng.shuffle(candidates)
    
    current = set(best)
    for elem in candidates:
        if time.time() > deadline - 10:
            break
        # Check if adding elem creates a violation
        current.add(elem)
        if _is_valid(current, forbidden_pairs):
            best = set(current)
            best_size = len(best)
        else:
            current.remove(elem)
    
    # Phase 2: Local search - try to swap elements
    # For each element in best, try removing it and adding a different element
    all_elements = set(range(1, N + 1))
    
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try adding elements not in best
        missing = all_elements - best
        missing_list = list(missing)
        rng.shuffle(missing_list)
        
        for add_elem in missing_list:
            if time.time() > deadline - 5:
                break
            
            # Try adding add_elem, then removing one element to fix violations
            candidate = set(best)
            candidate.add(add_elem)
            
            if _is_valid(candidate, forbidden_pairs):
                best = candidate
                best_size = len(best)
                improved = True
                break
            
            # If not valid, try removing one element to make it valid
            # Find which elements are involved in violations with add_elem
            violations = []
            for a in candidate:
                for (b, c) in forbidden_pairs[a]:
                    if b in candidate and c in candidate:
                        violations.append((a, b, c))
            
            if violations:
                # Try removing each element that appears in violations
                elements_to_try = set()
                for (a, b, c) in violations:
                    elements_to_try.add(a)
                    elements_to_try.add(b)
                    elements_to_try.add(c)
                
                for remove_elem in elements_to_try:
                    if remove_elem == add_elem:
                        continue
                    candidate2 = set(candidate)
                    candidate2.remove(remove_elem)
                    if _is_valid(candidate2, forbidden_pairs):
                        best = candidate2
                        best_size = len(best)
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            # Try random restarts with different seeds
            if iteration % 20 == 0:
                # Create a new random set and try to optimize
                new_set = set()
                elements = list(range(1, N + 1))
                rng.shuffle(elements)
                for elem in elements:
                    if time.time() > deadline - 5:
                        break
                    new_set.add(elem)
                    if _is_valid(new_set, forbidden_pairs):
                        pass
                    else:
                        new_set.remove(elem)
                
                if len(new_set) > best_size:
                    best = new_set
                    best_size = len(best)
    
    return list(best)

# EVOLVE-BLOCK-END
