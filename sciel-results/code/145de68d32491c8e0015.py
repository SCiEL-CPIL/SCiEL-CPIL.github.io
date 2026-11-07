import time
import sys

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def _has_bad_triple(S):
    """Return True if S contains distinct a,b,c with a+b,a+c,b+c all in S."""
    L = sorted(S)
    n = len(L)
    Sset = set(L)
    for i in range(n):
        a = L[i]
        for j in range(i+1, n):
            b = L[j]
            s_ab = a + b
            if s_ab not in Sset:
                continue
            for k in range(j+1, n):
                c = L[k]
                if (a+c) in Sset and (b+c) in Sset:
                    return True
    return False

def _validate(S):
    return not _has_bad_triple(set(S))

def _can_add_fast(S, x, Sset):
    """
    Check if adding x to S creates a bad triple.
    S is a set of ints, x is not in S.
    A new bad triple must include x.
    The triple is {x, p, q} with p, q in S, p < q.
    The pair sums are x+p, x+q, p+q.
    All three must be in S ∪ {x}.
    Since x is new, x+p, x+q, p+q must be in S if they are not x.
    Note: x+p = x implies p=0 (impossible, elements >= 1).
    So x+p, x+q, p+q must be in S.
    """
    L = sorted(S)
    n = len(L)
    for i in range(n):
        p = L[i]
        # Optimization: if p + x > N, then p+x cannot be in S (since max element is N).
        # But wait, S is subset of 1..128. If p+x > 128, it's not in S.
        # Actually, we just check membership in Sset.
        for j in range(i+1, n):
            q = L[j]
            # Triple {x, p, q}
            # Sums: x+p, x+q, p+q
            if (x + p) in Sset and (x + q) in Sset and (p + q) in Sset:
                return False
    return True

def _backtrack_search(current_set, candidates, best_set, best_size, deadline):
    """
    Branch and bound backtracking.
    current_set: set of selected elements.
    candidates: list of remaining elements to consider (sorted).
    best_set: global best set found so far.
    best_size: global best size.
    deadline: time limit.
    """
    if time.time() > deadline:
        return best_set, best_size
    
    if not candidates:
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        return best_set, best_size
    
    # Upper bound: current size + remaining candidates
    if len(current_set) + len(candidates) <= best_size:
        return best_set, best_size
    
    # Heuristic: try to add the first candidate? Or branch on include/exclude?
    # To find MAX size, we should try to INCLUDE as many as possible.
    # Branching strategy:
    # 1. Try including candidates[0]. If valid, recurse with it included.
    # 2. Try excluding candidates[0]. Recurse with it excluded.
    
    x = candidates[0]
    rest = candidates[1:]
    
    # Branch 1: Include x
    if _can_add_fast(current_set, x, current_set):
        new_set = current_set | {x}
        best_set, best_size = _backtrack_search(new_set, rest, best_set, best_size, deadline)
    
    # Branch 2: Exclude x
    best_set, best_size = _backtrack_search(current_set, rest, best_set, best_size, deadline)
    
    return best_set, best_size

def _construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Initial good solution: Upper half [65, 128] is sum-free, so no sums are in the set.
    # Thus no triple can exist because a+b >= 65+66=131 > 128.
    # Size 64.
    initial_set = set(range(65, 129))
    best_set = initial_set
    best_size = len(initial_set)
    
    # We want to beat 64.
    # The problem is small enough (128 elements) that a smart backtracking might find larger sets.
    # However, full backtracking on 128 elements is too slow.
    # We need to reduce the search space.
    
    # Strategy: 
    # The constraint is local. 
    # If we pick a set of "small" numbers, their sums might collide.
    # If we pick a set of "large" numbers, sums go out of range.
    # A mixed set is tricky.
    
    # Let's try a different angle: 
    # Randomized Restart with Local Search (Simulated Annealing) was the previous approach.
    # The new approach is Systematic Search with Pruning.
    # But 2^128 is too big.
    
    # Key Insight: 
    # If we fix the set of elements <= K, the rest are determined? No.
    # But we can iterate over possible "cores" of small numbers.
    
    # Let's try to find a set of size > 64.
    # Suppose we include some small numbers.
    # Let S_small be the set of elements in 1..64.
    # Let S_large be the set of elements in 65..128.
    # For any a,b in S, a+b in S.
    # If a,b in S_large, a+b > 128, so a+b not in S. Thus no triple can be formed entirely by large numbers.
    # So triples must involve at least one small number.
    
    # If we take ALL large numbers (64 of them), we can only add small numbers if they don't form a triple with other small numbers or large numbers.
    # If a in S_small, b in S_small, c in S_small: a+b, a+c, b+c in S.
    # If a in S_small, b in S_large, c in S_large:
    #   a+b in S, a+c in S, b+c in S.
    #   b+c > 128, so b+c NOT in S. Thus this case is impossible.
    # If a in S_small, b in S_small, c in S_large:
    #   a+b in S, a+c in S, b+c in S.
    #   a+c and b+c are in S (could be small or large).
    #   a+b is in S.
    
    # So, if we fix S_large to be a subset of [65..128], and S_small to be a subset of [1..64],
    # The constraints are:
    # 1. No triple in S_small (internal).
    # 2. For any a,b in S_small and c in S_large: NOT (a+b in S and a+c in S and b+c in S).
    
    # This decomposition allows us to search S_small (size 64) more easily?
    # 2^64 is still huge.
    
    # However, we can use the fact that we want to MAXIMIZE size.
    # Max size of S_large is 64.
    # If we can add even 1 small number, we get 65.
    # Can we add 1 small number to [65..128]?
    # Let a be the small number.
    # We need to check if there exist b,c in S_small (just {a})? No, distinct a,b,c.
    # So we need at least 3 small numbers to form a triple within smalls.
    # Or 2 smalls and 1 large?
    # If S_small = {a}, no triple within smalls.
    # Triple with 1 small, 2 large: Impossible as shown above (b+c > 128).
    # Triple with 2 smalls, 1 large:
    #   a,b in S_small, c in S_large.
    #   Need a+b in S, a+c in S, b+c in S.
    #   If S_small has only 1 element, we can't pick distinct a,b from S_small.
    #   So if |S_small| <= 1, no triples are possible at all!
    #   Wait, if |S_small| = 1, say {a}.
    #   Can we have a triple a, c, d with c,d in S_large?
    #   Sums: a+c, a+d, c+d.
    #   c+d > 128, so c+d not in S. So no triple.
    #   Can we have a triple a, c, d with c in S_small, d in S_large?
    #   But c must be in S_small. If S_small={a}, c=a. Distinctness requires a!=c. So no.
    #   
    #   Therefore, if we take ALL large numbers [65..128] and ANY subset of small numbers of size <= 1, the set is VALID.
    #   Size = 64 + 1 = 65.
    #   
    #   Let's verify: S = {65, ..., 128} U {1}.
    #   Elements: 1, 65, 66, ..., 128.
    #   Check triples:
    #   - All large: sums > 128. OK.
    #   - 1, large1, large2: sums 1+l1, 1+l2, l1+l2. l1+l2 > 128. Not in S. OK.
    #   - 1, 1, large: Not distinct.
    #   - 1, small, small: Only one small.
    #   
    #   So size 65 is achievable.
    #   
    #   Can we add a second small number?
    #   S_small = {a, b}.
    #   Check triples within smalls: only 2 elements, no triple.
    #   Check triples with 2 smalls, 1 large:
    #   a, b in S_small, c in S_large.
    #   Need a+b in S, a+c in S, b+c in S.
    #   a+b is in 2..128.
    #   a+c and b+c are in 66..129 (if a,b >= 1, c >= 65).
    #   If a+b is in S, and a+c is in S, and b+c is in S, then we have a bad triple.
    #   
    #   We can choose S_large to AVOID c such that a+c and b+c are in S_large.
    #   If we remove c from S_large, the triple is broken.
    #   
    #   So the strategy is:
    #   1. Start with S_large = [65..128].
    #   2. Try to add small numbers.
    #   3. For each small number added, we might need to remove some large numbers to prevent triples.
    #   4. Net gain = (# smalls added) - (# larges removed).
    #   
    #   This is a combinatorial optimization.
    #   
    #   Let's implement a search that tries to add small numbers and removes conflicting large numbers.
    #   
    #   Algorithm:
    #   1. S_large = set(65..128)
    #   2. S_small = set()
    #   3. Iterate through small numbers 1..64 in random order.
    #   4. Try to add x to S_small.
    #   5. Check if adding x creates any bad triple.
    #      - Triples within S_small: check if x forms a triple with other smalls.
    #      - Triples with 2 smalls, 1 large: for each pair (x, y) in S_small (including x itself if y!=x? No, distinct), check if there exists c in S_large such that x+y, x+c, y+c are in S.
    #        If yes, we must remove c from S_large.
    #   6. If removing c is "worth it" (i.e., we can still maintain a large set), do it.
    #   
    #   Actually, simpler:
    #   For a fixed S_small, what is the maximum size of S_large?
    #   S_large must be a subset of [65..128] such that no c in S_large forms a bad triple with any pair (a,b) in S_small.
    #   Forbidden c: c is forbidden if (a+b) in S (which is S_small U S_large) AND (a+c) in S AND (b+c) in S.
    #   Note: a+b might be in S_small or S_large.
    #   a+c and b+c are > 65, so they must be in S_large.
    #   So c is forbidden if:
    #     (a+b) in S_small OR (a+b) in S_large
    #     AND (a+c) in S_large
    #     AND (b+c) in S_large
    #   
    #   This is a constraint on S_large.
    #   
    #   Let's try a greedy approach with backtracking on S_small.
    
    best_size = 64
    best_set = set(range(65, 129))
    
    # Try to add small numbers
    # We'll use a randomized local search on the "small" part, updating the "large" part to be maximal valid given the small part.
    
    def _max_large_set(S_small, S_large_initial):
        """
        Given S_small, find the largest subset of S_large_initial that is valid with S_small.
        We start with S_large_initial and remove elements that cause conflicts.
        """
        S_large = set(S_large_initial)
        S_small_set = set(S_small)
        
        # We need to ensure no triple (a,b,c) with a,b in S_small, c in S_large
        # And no triple within S_small (handled externally or here?)
        # And no triple within S_large (impossible)
        # And no triple with 1 small, 2 large (impossible)
        
        # So only constraint on S_large is from pairs in S_small.
        # For each pair (a,b) in S_small:
        #   c is invalid if (a+b) in S AND (a+c) in S AND (b+c) in S.
        #   (a+c) and (b+c) must be in S_large.
        #   (a+b) must be in S_small or S_large.
        
        # Iterate and remove conflicting c's.
        # Since removing c might make other c's valid? No, constraints are "c is bad if...".
        # If c is bad, we remove it. Removing c doesn't make other c's bad.
        # So we can just remove all c that are bad with respect to the CURRENT S_large.
        # But wait, (a+b) in S depends on S_large.
        # If a+b is in S_large, and we remove a+b, then the condition "(a+b) in S" fails, so c might become valid.
        # So we need to iterate until stable.
        
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for c in list(S_large):
                for a in S_small_set:
                    for b in S_small_set:
                        if a == b: continue
                        if a > b: continue # avoid duplicates
                        # Check triple {a, b, c}
                        sum_ab = a + b
                        sum_ac = a + c
                        sum_bc = b + c
                        if sum_ab in S_small_set or sum_ab in S_large:
                            if sum_ac in S_large and sum_bc in S_large:
                                to_remove.add(c)
            if to_remove:
                S_large -= to_remove
                changed = True
        
        return S_large

    # Randomized restarts
    import random
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Randomly select a subset of small numbers to try
        # Or start with empty small set and add greedily?
        # Let's try to build S_small greedily.
        S_small = set()
        S_large = set(range(65, 129))
        
        # Order small numbers randomly
        smalls = list(range(1, 65))
        rng.shuffle(smalls)
        
        for x in smalls:
            # Try adding x to S_small
            new_S_small = S_small | {x}
            
            # Check if new_S_small has internal triples
            # If yes, skip x? Or resolve?
            # Internal triples in S_small: a,b,c in S_small, a+b,a+c,b+c in S_small.
            # If S_small has an internal triple, it's invalid.
            # So we must avoid internal triples.
            
            # Fast check for internal triple with x
            valid_internal = True
            for a in S_small:
                for b in S_small:
                    if a == b: continue
                    if a > b: continue
                    # Triple {x, a, b}
                    if x + a in new_S_small and x + b in new_S_small and a + b in new_S_small:
                        valid_internal = False
                        break
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
            
            # Calculate new S_large
            new_S_large = _max_large_set(new_S_small, S_large)
            
            # Compare sizes
            if len(new_S_small) + len(new_S_large) > len(S_small) + len(S_large):
                S_small = new_S_small
                S_large = new_S_large
                # Update global best
                if len(S_small) + len(S_large) > best_size:
                    best_size = len(S_small) + len(S_large)
                    best_set = S_small | S_large
            
        # Local search on S_small?
        # Try swapping small numbers?
        # For now, just do a few random swaps
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            if not S_small:
                break
            # Pick a random small to remove
            to_remove = rng.choice(list(S_small))
            S_small.remove(to_remove)
            # Recalculate S_large
            S_large = _max_large_set(S_small, set(range(65, 129)))
            
            # Try to add a random small
            candidates = [x for x in range(1, 65) if x not in S_small]
            if candidates:
                x = rng.choice(candidates)
                new_S_small = S_small | {x}
                # Check internal
                valid_internal = True
                for a in S_small:
                    for b in S_small:
                        if a == b: continue
                        if a > b: continue
                        if x + a in new_S_small and x + b in new_S_small and a + b in new_S_small:
                            valid_internal = False
                            break
                    if not valid_internal:
                        break
                if valid_internal:
                    new_S_large = _max_large_set(new_S_small, set(range(65, 129)))
                    if len(new_S_small) + len(new_S_large) > len(S_small) + len(S_large):
                        S_small = new_S_small
                        S_large = new_S_large
                        if len(S_small) + len(S_large) > best_size:
                            best_size = len(S_small) + len(S_large)
                            best_set = S_small | S_large

    # Final validation
    if _validate(best_set):
        return list(best_set)
    else:
        return list(range(65, 129))

def construct():
    return _construct()
