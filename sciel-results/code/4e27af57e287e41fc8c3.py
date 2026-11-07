import time
import itertools

# EVOLVE-BLOCK-START

def _is_valid_fast(S):
    """Fast check for validity using partner sets."""
    S_set = set(S)
    # Precompute partners: for each x, set of y in S such that x+y in S
    partners = {x: set() for x in S}
    for x in S:
        for y in S:
            if x == y:
                continue
            if x + y in S_set:
                partners[x].add(y)
    
    # Check for violation:
    # Existence of a, b, c distinct in S such that a+b, a+c, b+c in S.
    # Equivalent to: Exists a < b with a+b in S, and exists c != a,b in S
    # such that c in partners[a] and c in partners[b].
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            if a + b in S_set:
                # Check if there is a common partner c not equal to a or b
                # Iterate over the smaller set for efficiency
                if len(partners[a]) < len(partners[b]):
                    for c in partners[a]:
                        if c != a and c != b and c in partners[b]:
                            return False
                else:
                    for c in partners[b]:
                        if c != a and c != b and c in partners[a]:
                            return False
    return True

def _can_add(S, x):
    """Check if adding x to S (x not in S) creates a violation."""
    S_set = set(S)
    # New triples must include x.
    # So we check triples (x, a, b) with a, b in S, a != b.
    # Sums: x+a, x+b, a+b.
    # All must be in S U {x}.
    # x+a > x => x+a in S.
    # x+b > x => x+b in S.
    # a+b can be in S or a+b == x.
    
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        if x + a not in S_set:
            continue
        for j in range(len(S_list)):
            if i == j:
                continue
            b = S_list[j]
            if x + b not in S_set:
                continue
            # Check a+b
            if (a + b) in S_set or (a + b == x):
                return False
    return True

def _can_remove_and_add(S, remove, add):
    """Check if S - {remove} + {add} is valid (add not in S, remove in S)."""
    new_S = set(S)
    new_S.discard(remove)
    new_S.add(add)
    return _valid_triples_check(new_S)

def construct():
    deadline = time.time() + 80  # safety margin for 90s limit
    
    # Start with a known valid set
    # {63..128} has 66 elements, valid since min sum = 63+64=127, but 127 not in set, 
    # actually 63+64=127 not in {63..128}, 63+65=128 which IS in set.
    # Wait: 63+65=128, 63+64=127 (not in set), 64+65=129 (not in set)
    # So triple (63,64,65): sums 127,128,129. Only 128 in set. Not all three. OK.
    # Triple (63,65,66): 128,129,131. Only 128. OK.
    # Actually for any a<b<c in {63..128}: a+b >= 63+64=127, a+c >= 63+65=128, b+c >= 64+65=129
    # So a+b could be 127 or 128, a+c could be 128 or more, b+c >= 129
    # For all three sums to be in {63..128}: need a+b<=128, a+c<=128, b+c<=128
    # b+c >= 129 > 128, so b+c is NEVER in the set. Thus no violation!
    # So {63..128} is valid.
    
    S = set(range(63, 129))  # 66 elements
    best = set(S)
    
    all_nums = set(range(1, 129))
    
    # Phase 1: Greedy addition
    while time.time() < deadline:
        added = False
        for x in sorted(all_nums - S):
            if time.time() > deadline:
                break
            if _can_add(S, x):
                S.add(x)
                if len(S) > len(best):
                    best = set(S)
                added = True
        if not added:
            break
    
    # Phase 2: Limited Local search
    # To avoid timeout, we limit the number of swap attempts.
    max_swaps = 200
    swaps_attempted = 0
    improved = True
    while improved and time.time() < deadline and swaps_attempted < max_swaps:
        improved = False
        S_list = list(S)
        missing = list(all_nums - S)
        
        # Try remove 1, add 2 (net +1)
        # We pick random pairs to avoid checking all
        import random
        random.seed(42) # Deterministic for reproducibility in this block?
        
        # Shuffle to explore different parts
        r1_candidates = S_list[:]
        random.shuffle(r1_candidates)
        
        for r1 in r1_candidates:
            if time.time() > deadline or swaps_attempted >= max_swaps:
                break
            for r2 in S_list:
                if r2 <= r1:
                    continue
                if time.time() > deadline or swaps_attempted >= max_swaps:
                    break
                
                a1_candidates = missing[:]
                random.shuffle(a1_candidates)
                
                for a1 in a1_candidates:
                    if time.time() > deadline or swaps_attempted >= max_swaps:
                        break
                    for a2 in missing:
                        if a2 <= a1:
                            continue
                        if time.time() > deadline or swaps_attempted >= max_swaps:
                            break
                        
                        swaps_attempted += 1
                        new_S = set(S)
                        new_S.discard(r1)
                        new_S.discard(r2)
                        new_S.add(a1)
                        new_S.add(a2)
                        if _is_valid_fast(new_S):
                            S = new_S
                            if len(S) > len(best):
                                best = set(S)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try single add after swaps
        for x in sorted(all_nums - S):
            if time.time() > deadline:
                break
            if _can_add(S, x):
                S.add(x)
                if len(S) > len(best):
                    best = set(S)
                improved = True
    
    # Return the best found
    result = sorted(best)
    return list(result)

# EVOLVE-BLOCK-END
