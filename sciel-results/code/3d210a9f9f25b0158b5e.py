import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # This is equivalent to (b-a)(c-a) = a^2
    forbidden = set()
    for a in range(1, 91):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and b <= 180 and c <= 180:
                    forbidden.add((a, b, c))
    
    # Build adjacency: for each element, which pairs conflict with it
    # If we include element x, then for each forbidden triple containing x,
    # we cannot include both of the other two elements
    # This is a 3-uniform hypergraph, so we need a different approach
    
    # For branch and bound, we need to check if adding an element creates conflicts
    # with existing elements in the solution
    
    def can_add(x, S):
        """Check if adding x to S keeps it valid."""
        # For each forbidden triple containing x, check if the other two are in S
        for (a, b, c) in forbidden:
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            elif x == c:
                if a in S and b in S:
                    return False
        return True
    
    # Start with a greedy solution: include all numbers > 90, then greedily add smaller ones
    S = set(range(91, 181))
    
    for a in range(1, 91):
        if time.time() > deadline - 10:
            break
        if can_add(a, S):
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    # Now do local search to improve
    random.seed(123)
    all_nums = set(range(1, 181))
    excluded = list(all_nums - S)
    
    # Precompute for each element, the list of forbidden triples it's in
    triples_by_elem = {i: [] for i in range(1, 181)}
    for (a, b, c) in forbidden:
        triples_by_elem[a].append((a, b, c))
        triples_by_elem[b].append((a, b, c))
        triples_by_elem[c].append((a, b, c))
    
    def can_add_fast(x, S):
        """Faster check using precomputed triples."""
        for (a, b, c) in triples_by_elem[x]:
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            elif x == c:
                if a in S and b in S:
                    return False
        return True
    
    # Local search with swaps
    for iteration in range(100000):
        if time.time() > deadline - 5:
            break
        
        if not excluded:
            break
            
        a = random.choice(excluded)
        
        if can_add_fast(a, S):
            S.add(a)
            excluded.remove(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # Try swapping: remove one element from S and add a
            if S:
                # Try multiple random removals to find a good swap
                found = False
                for _ in range(10):
                    if time.time() > deadline - 3:
                        break
                    b = random.choice(list(S))
                    S.remove(b)
                    excluded.append(b)
                    
                    if can_add_fast(a, S):
                        S.add(a)
                        excluded.remove(a)
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                        found = True
                        break
                    else:
                        S.add(b)
                        excluded.remove(b)
    
    return sorted(best_S)
