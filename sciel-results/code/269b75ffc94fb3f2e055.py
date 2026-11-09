"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N+1))
    
    # Precompute all conflict triples: (a, b, c) where 1/a = 1/b + 1/c
    # a = k*m*n, b = k*m*(m+n), c = k*n*(m+n), gcd(m,n)=1
    triples = set()
    for k in range(1, 181):
        for m in range(1, 181):
            for n in range(1, 181):
                if m >= n:
                    continue
                if _gcd(m, n) != 1:
                    continue
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    if len({a, b, c}) == 3:
                        triples.add((min(a,b,c), min(max(a,b), max(b,c)), max(a,b,c)))
    
    # Build conflict adjacency: for each pair (x,y), check if there exists z such that (x,y,z) is a triple
    # For efficiency, build a dict: for each pair (x,y), list of z that complete a triple
    pair_conflicts = {}  # (x,y) -> set of z
    for (a, b, c) in triples:
        for (x, y, z) in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
            key = (x, y)
            if key not in pair_conflicts:
                pair_conflicts[key] = set()
            pair_conflicts[key].add(z)
    
    # Also build for each element, the set of other elements it conflicts with (appears in a triple with)
    element_conflicts = {i: set() for i in range(1, N+1)}
    for (a, b, c) in triples:
        element_conflicts[a].update([b, c])
        element_conflicts[b].update([a, c])
        element_conflicts[c].update([a, b])
    
    # Precompute for each element x, the list of triples it participates in
    element_triples = {i: [] for i in range(1, N+1)}
    for (a, b, c) in triples:
        element_triples[a].append((a, b, c))
        element_triples[b].append((a, b, c))
        element_triples[c].append((a, b, c))
    
    def try_add(s, x):
        """Try to add x to set s, return True if valid"""
        for (a, b, c) in element_triples[x]:
            if x == a:
                if b in s and c in s:
                    return False
            elif x == b:
                if a in s and c in s:
                    return False
            elif x == c:
                if a in s and b in s:
                    return False
        return True
    
    def local_search(s):
        """Local search: repeatedly try to add random elements"""
        for iteration in range(200):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for x in range(1, N+1):
                if x not in s and try_add(s, x):
                    s.add(x)
                    improved = True
            if not improved:
                break
        
        # Try swaps: remove one, add one or more
        for iteration in range(100):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for x in list(s):
                test_set = s - {x}
                # Try to add elements not in test_set
                added = 0
                for y in range(1, N+1):
                    if y not in test_set and try_add(test_set, y):
                        test_set.add(y)
                        added += 1
                if len(test_set) > len(s):
                    s = test_set
                    improved = True
                    break
            if not improved:
                break
        
        # Final: try all single additions again
        for x in range(1, N+1):
            if x not in s and try_add(s, x):
                s.add(x)
        
        return s
    
    # Minimum-degree greedy: always select the vertex with fewest remaining conflicts
    def min_degree_greedy(remaining_elements):
        """Build a valid set by greedily adding elements with minimum conflict degree"""
        result = set()
        remaining = set(remaining_elements)
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            
            # For each candidate, count how many elements in remaining it conflicts with
            # (i.e., how many would be removed if we add it)
            best_x = None
            min_conflicts = float('inf')
            candidates = []
            
            for x in remaining:
                # Count conflicts: elements y in remaining such that adding x would conflict with y
                conflicts_in_remaining = 0
                for y in remaining:
                    if y == x:
                        continue
                    # Check if x and y are in a triple together
                    # i.e., there exists z such that (x, y, z) is a triple
                    if (x, y) in pair_conflicts or (y, x) in pair_conflicts:
                        conflicts_in_remaining += 1
                
                if conflicts_in_remaining < min_conflicts:
                    min_conflicts = conflicts_in_remaining
                    candidates = [x]
                elif conflicts_in_remaining == min_conflicts:
                    candidates.append(x)
            
            if not candidates:
                break
                
            # Pick randomly from candidates
            best_x = random.choice(candidates)
            result.add(best_x)
            remaining.remove(best_x)
            
            # Remove conflicting elements from remaining
            to_remove = set()
            for y in remaining:
                for (a, b, c) in element_triples[y]:
                    if y == a:
                        if b in result and c in result:
                            to_remove.add(y)
                            break
                    elif y == b:
                        if a in result and c in result:
                            to_remove.add(y)
                            break
                    elif y == c:
                        if a in result and b in result:
                            to_remove.add(y)
                            break
            remaining -= to_remove
        
        return result
    
    # Best set found
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Try multiple strategies
    # Strategy 1: Standard greedy with different orderings
    orderings = [
        list(range(1, N+1)),           # ascending
        list(range(N, 0, -1)),         # descending
        list(range(91, N+1)) + list(range(1, 91)),  # large first
    ]
    
    # Add random orderings
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        # Greedy build
        result = set()
        for x in order:
            if try_add(result, x):
                result.add(x)
        
        # Local search
        result = local_search(result)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Strategy 2: Minimum-degree greedy with random restarts
    for restart in range(30):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random subset of elements to start with
        if random.random() < 0.5:
            remaining = set(range(1, N+1))
        else:
            # Start with a random subset
            remaining = set(random.sample(range(1, N+1), random.randint(80, 150)))
        
        result = min_degree_greedy(remaining)
        result = local_search(result)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Final polish: try all single additions
    for x in range(1, N+1):
        if x not in best_set and try_add(best_set, x):
            best_set.add(x)
            best_size = len(best_set)
    
    return list(best_set)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# EVOLVE-BLOCK-END
