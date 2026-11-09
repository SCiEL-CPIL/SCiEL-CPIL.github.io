"""Rewritten program: greedy + local search for large reciprocal-Schur-free subset of [1,180]."""
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a,b,c) with distinct elements where 1/a = 1/b + 1/c
    # This means a = bc/(b+c). For a to be integer, (b+c) must divide bc.
    # Equivalently: 1/a = 1/b + 1/c  <=>  c = ab/(a-b) for a > b, or similar.
    # Let's just brute force: for each pair (b,c) with b < c, check if a = bc/(b+c) is an integer in [1,N]
    # and a != b and a != c.
    
    forbidden = set()  # set of frozensets {a,b,c}
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    forbidden.add(triple)
    
    # Also need to consider the case where a < b or a > c etc. But since we iterate b<c and compute a,
    # a could be less than b, between, or greater than c. The frozenset handles all permutations.
    # But wait: 1/a = 1/b + 1/c. If we fix b,c and solve for a, we get a = bc/(b+c).
    # Since 1/b + 1/c > 1/c (assuming b>0), we have 1/a > 1/c so a < c.
    # Similarly a < b. So a is always the smallest of the three.
    # Let's verify: a = bc/(b+c). Since b,c >= 1, b+c > b so a = bc/(b+c) < c.
    # And b+c > c so a < b. Yes, a is always the minimum.
    
    # Greedy approach: start with all elements, then iteratively remove elements that are in the most forbidden triples
    # Alternatively: greedy add - start empty, add elements one by one if they don't create a forbidden triple
    
    # Let's try a greedy construction: process elements in some order, add if it doesn't form a forbidden triple with existing elements.
    # Better: use a heuristic based on degree in the hypergraph of forbidden triples.
    
    # Precompute for each element, the list of forbidden triples containing it
    triples_by_element = {i: [] for i in range(1, N+1)}
    for triple in forbidden:
        for x in triple:
            triples_by_element[x].append(triple)
    
    # Strategy: greedy removal. Start with all N elements. While there's a forbidden triple, remove the element in that triple with the highest "degree" (number of forbidden triples it participates in, considering only currently present elements).
    
    def is_valid(S):
        """Check if set S has no forbidden triple."""
        S_set = set(S)
        for triple in forbidden:
            if all(x in S_set for x in triple):
                return False
        return True
    
    # Greedy: start with all elements, remove elements to break all forbidden triples
    S = set(range(1, N+1))
    
    # Build active forbidden triples
    active_triples = list(forbidden)
    
    # Iteratively remove elements
    while True:
        # Find a forbidden triple that is still fully in S
        found = None
        for t in active_triples:
            if all(x in S for x in t):
                found = t
                break
        if found is None:
            break
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Among the three elements in found, pick the one with the highest number of other active forbidden triples containing it (within S)
        best_elem = None
        best_count = -1
        for elem in found:
            count = 0
            for t2 in active_triples:
                if elem in t2 and all(x in S for x in t2):
                    count += 1
            if count > best_count:
                best_count = count
                best_elem = elem
        S.discard(best_elem)
    
    # Local search: try to add back removed elements
    removed = set(range(1, N+1)) - S
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 15:
        improved = False
        for elem in list(removed):
            if time.time() - start_time > DEADLINE - 15:
                break
            # Check if adding elem creates any forbidden triple
            S_test = S | {elem}
            valid = True
            for t in forbidden:
                if elem in t and all(x in S_test for x in t):
                    valid = False
                    break
            if valid:
                S = S_test
                removed.discard(elem)
                improved = True
    
    # Try swapping: remove one, add another
    if time.time() - start_time < DEADLINE - 20:
        for _ in range(3):  # a few rounds
            if time.time() - start_time > DEADLINE - 20:
                break
            improved = False
            for keep in list(S):
                if time.time() - start_time > DEADLINE - 20:
                    break
                for add in list(removed):
                    if keep == add:
                        continue
                    S_test = (S - {keep}) | {add}
                    valid = True
                    for t in forbidden:
                        if all(x in S_test for x in t):
                            valid = False
                            break
                    if valid and len(S_test) >= len(S):
                        S = S_test
                        removed = set(range(1, N+1)) - S
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    result = sorted(S)
    if not result:
        result = [91]  # fallback
    
    # Final validation
    result_set = set(result)
    for t in forbidden:
        if all(x in result_set for x in t):
            # Shouldn't happen, but if it does, remove one element
            for x in t:
                if x in result_set:
                    result_set.discard(x)
                    break
            result = sorted(result_set)
            break
    
    return result
