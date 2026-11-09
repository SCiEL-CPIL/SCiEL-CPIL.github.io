"""Rewritten program: Multi-start greedy + local search for large reciprocal-Schur-free subset of [1,180]."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    N = 180
    # Precompute all forbidden triples (a,b,c) with distinct elements where 1/a = 1/b + 1/c
    # a = bc/(b+c). a is the minimum.
    forbidden = set()
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = frozenset([a, b, c])
                    forbidden.add(triple)
    
    # Precompute for each element, the list of forbidden triples containing it
    triples_by_element = {i: [] for i in range(1, N+1)}
    for triple in forbidden:
        for x in triple:
            triples_by_element[x].append(triple)
            
    # Calculate degree (number of forbidden triples) for each element
    degrees = {i: len(triples_by_element[i]) for i in range(1, N+1)}

    def can_add(x, S):
        """Check if x can be added to set S without forming a forbidden triple."""
        # S is a set of integers
        for t in triples_by_element[x]:
            # t is a frozenset of 3 elements, one of which is x
            # Check if the other two are in S
            # We can just check if all elements of t except x are in S
            # Since t is small, this is fast.
            if all(y in S for y in t if y != x):
                return False
        return True

    def greedy_construction(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, time_limit):
        """Improve S using 1-flip (add) and 2-flip (swap) moves."""
        S = set(S)
        removed = set(range(1, N+1)) - S
        
        # Phase 1: Add as many as possible
        while True:
            improved = False
            for x in list(removed):
                if time.time() > time_limit:
                    return S
                if can_add(x, S):
                    S.add(x)
                    removed.discard(x)
                    improved = True
            if not improved:
                break
        
        # Phase 2: Swap (remove one, add one) - no size gain, but might enable future gains? 
        # Actually, for Max Independent Set, simple 1-flip add is usually sufficient if we start with a good greedy.
        # However, we can try 2-flip swap to escape local minima if we allow net 0 change? 
        # No, we want to maximize size. So we look for moves that increase size.
        # 1-flip add increases size by 1.
        # 2-flip swap (remove 1, add 1) keeps size same.
        # 3-flip swap (remove 1, add 2) increases size by 1.
        
        # Let's try 3-flip swaps: remove x in S, add y, z in removed such that {x, y, z} is not forbidden and {y, z, w} is not forbidden for any w in S\{x}.
        # This is expensive. 
        # Given the small N=180, we can try a simpler approach: 
        # Iterate through removed elements y, z. If can_add(y, S) and can_add(z, S-{y})... no.
        # Let's stick to 1-flip add and maybe some random restarts if we get stuck.
        
        return S

    best_S = set()
    
    # 1. Ascending degree
    order1 = sorted(range(1, N+1), key=lambda x: degrees[x])
    S1 = greedy_construction(order1)
    S1 = local_search(S1, DEADLINE - 10)
    if len(S1) > len(best_S):
        best_S = S1
        
    # 2. Descending degree
    order2 = sorted(range(1, N+1), key=lambda x: -degrees[x])
    S2 = greedy_construction(order2)
    S2 = local_search(S2, DEADLINE - 10)
    if len(S2) > len(best_S):
        best_S = S2
        
    # 3. Ascending index
    order3 = list(range(1, N+1))
    S3 = greedy_construction(order3)
    S3 = local_search(S3, DEADLINE - 10)
    if len(S3) > len(best_S):
        best_S = S3
        
    # 4. Descending index
    order4 = list(range(N, 0, -1))
    S4 = greedy_construction(order4)
    S4 = local_search(S4, DEADLINE - 10)
    if len(S4) > len(best_S):
        best_S = S4

    # 5. Random shuffles
    remaining_time = DEADLINE - time.time()
    if remaining_time > 5:
        max_iterations = 50
        for i in range(max_iterations):
            if time.time() > DEADLINE - 5:
                break
            order = list(range(1, N+1))
            random.shuffle(order)
            S_rand = greedy_construction(order)
            S_rand = local_search(S_rand, DEADLINE - 5)
            if len(S_rand) > len(best_S):
                best_S = S_rand
                # If we found a very good one, we might spend more time on it?
                # For now, just keep updating.

    # Final Polish: Try to add any remaining elements to best_S
    final_S = set(best_S)
    removed = set(range(1, N+1)) - final_S
    for x in list(removed):
        if can_add(x, final_S):
            final_S.add(x)
            removed.discard(x)
            
    # Final Validation
    result = sorted(final_S)
    result_set = set(result)
    for t in forbidden:
        if all(x in result_set for x in t):
            # Remove one element to fix
            for x in t:
                if x in result_set:
                    result_set.discard(x)
                    break
            result = sorted(result_set)
            break
            
    if not result:
        result = [91]
        
    return result
