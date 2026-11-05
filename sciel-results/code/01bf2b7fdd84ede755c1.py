import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 180
    
    # Precompute forbidden triples structure
    # We need to quickly check if adding 'a' to set S creates a triple.
    # Since a < b < c is required for distinct elements in the equation 1/a = 1/b + 1/c,
    # if we add 'a', it must be the smallest.
    # So we check if there exist b, c in S such that b > a, c > a and 1/a = 1/b + 1/c.
    # This is equivalent to bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2.
    
    # For a fixed 'a', the pairs (b, c) that form a forbidden triple with 'a' are determined by divisors of a^2.
    # Let x = b-a, y = c-a. xy = a^2.
    # b = a+x, c = a+y.
    # We need b, c in S.
    
    # Precompute for each a, the list of (b, c) pairs that would form a triple.
    # Actually, we can just compute this on the fly or precompute a set of "forbidden pairs for a".
    # forbidden_pairs[a] = set of (b, c) with b < c, b > a, c > a, such that {a, b, c} is forbidden.
    
    forbidden_pairs = [set() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # x < y, xy = a^2
        # x ranges from 1 to a-1 (since x != y and x*y=a^2 implies x != a)
        for x in range(1, a):
            if a_sq % x == 0:
                y = a_sq // x
                if x == y:
                    continue
                b = a + x
                c = a + y
                if b <= N and c <= N:
                    # b < c is guaranteed since x < y
                    forbidden_pairs[a].add((b, c))
    
    def is_valid_addition(a, S):
        # Check if adding a to S creates a forbidden triple.
        # Since we build by adding smaller numbers to a base of larger numbers,
        # and we check all a from 1 to N, we must ensure a is the smallest in any new triple.
        # If we maintain S such that we only add a if it doesn't form a triple with existing elements,
        # and we process a in increasing order, then any triple involving a must have a as the min.
        # However, if we do random restarts, we might add elements in any order.
        # To be safe, we check if there exist b, c in S such that {a, b, c} is forbidden.
        # This requires checking if a is the min, or if a is b or c.
        # If a is b: 1/min = 1/a + 1/c. min < a.
        # If a is c: 1/min = 1/b + 1/a. min < b < a.
        
        # For simplicity and speed in the greedy phase where we add small to large:
        # We assume the base set contains only large numbers.
        # Then any new small number 'a' can only be the 'a' in the equation.
        # So we only check if there are b, c in S such that (b, c) in forbidden_pairs[a].
        
        # But to support general local search, we should check all roles.
        # However, the "Greedy Addition to Large Base" strategy is so strong that we stick to it.
        # Base S = {91..180}.
        # Candidates = {1..90}.
        # If we add a from Candidates, a < 91 <= all elements in S.
        # So a is always the smallest.
        # We just need to check if any (b, c) in forbidden_pairs[a] is a subset of S.
        
        for b, c in forbidden_pairs[a]:
            if b in S and c in S:
                return False
        return True

    best_size = 0
    best_set = set()
    
    # Strategy:
    # 1. Start with base set {91..180}.
    # 2. Try to add elements from {1..90} in various orders.
    # 3. Use local search to improve.
    
    base_set = set(range(91, 181))
    
    # Initial greedy pass
    # Try adding 1..90 in increasing order
    S = set(base_set)
    for a in range(1, 91):
        if is_valid_addition(a, S):
            S.add(a)
            
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
        
    # Multiple random restarts with different orders of adding 1..90
    candidates = list(range(1, 91))
    
    for _ in range(200):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(candidates)
        S = set(base_set)
        for a in candidates:
            if is_valid_addition(a, S):
                S.add(a)
                
        # Local Search: Try to swap or add/remove
        # Since we want to maximize size, we can try to "kick" by removing a random element
        # that is "expensive" (blocks many others) and re-adding candidates.
        
        # Simple improvement: Try to add any remaining candidate that was skipped.
        # In the greedy pass, if 'a' was skipped, it's because it conflicted with some b, c in S.
        # If we remove one of those b or c, maybe we can add 'a' and still keep size?
        # This is complex. Let's stick to the greedy which is already very good.
        
        # However, we can try a "repair" step.
        # If we have a valid set S, can we add more?
        # The greedy pass adds if valid. If not valid, it skips.
        # It's possible that skipping 'a' allowed adding 'b', but if we removed 'b' we could add 'a' and 'c'?
        # Given the density, simple greedy is likely near optimal for this structure.
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Final Polish: Try to add any number from 1..180 not in best_set
    # This handles cases where the base set choice was suboptimal or greedy missed something.
    # But base set {91..180} is structurally perfect for "small to large" addition.
    
    # One more check: What if we start with a different split?
    # E.g. {100..180} base? Size 81.
    # {91..180} is size 90.
    # Adding small numbers:
    # a=1: 1/1 = 1/b + 1/c => b=c=2. Not distinct. No triple with 1?
    # (b-1)(c-1)=1 => b-1=1, c-1=1 => b=2, c=2. Not distinct.
    # So 1 is always safe?
    # Check: 1/a = 1/b + 1/c. If a=1, 1 = 1/b + 1/c.
    # Since b,c >= 2, 1/b + 1/c <= 1/2 + 1/3 = 5/6 < 1.
    # So 1 can NEVER be part of a forbidden triple?
    # Wait. 1/b + 1/c = 1.
    # If b=2, 1/c = 1/2 => c=2. Not distinct.
    # If b=3, 1/c = 2/3 => c=1.5. No.
    # So yes, 1 is always safe to include.
    
    # a=2: (b-2)(c-2)=4.
    # Pairs (x,y) for xy=4: (1,4), (2,2).
    # (1,4) => b=3, c=6. Triple {2, 3, 6}.
    # (2,2) => b=4, c=4. Not distinct.
    # So 2 conflicts with (3,6).
    
    # The greedy algorithm will handle these.
    
    return list(best_set)
