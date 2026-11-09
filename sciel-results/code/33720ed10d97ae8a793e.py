"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# EVOLVE-BLOCK-END

# We will define the helper functions inside the EVOLVE-BLOCK to keep it self-contained 
# or outside? The prompt says "Keep the function names and signatures of the seed program."
# The seed program only has `construct`. I should put everything inside `construct` 
# or define helpers in the block. Defining helpers in the block is cleaner.

# EVOLVE-BLOCK-START
import time
import random

def _precompute_conflicts():
    """
    Precompute all forbidden triples (a, b, c) with a < b < c in 1..128
    such that a+b, a+c, b+c are all in 1..128.
    Note: The problem says "a+b, a+c and b+c are also selected". 
    It does NOT require a+b, a+c, b+c to be distinct from a,b,c or each other.
    It just requires them to be in the set S.
    """
    n = 128
    # A triple (a,b,c) is "bad" if a+b, a+c, b+c are all <= 128.
    # Actually, the condition is: a,b,c in S AND a+b in S AND a+c in S AND b+c in S.
    # So for a candidate set S, we check if any such triple exists.
    # To speed up search, we precompute the list of all "structurally possible" 
    # forbidden triples in the universe 1..128.
    # A triple (a,b,c) is a potential violator if a+b <= 128, a+c <= 128, b+c <= 128.
    # If any of these sums > 128, they can't be in S (since S subset 1..128), 
    # so the triple can never be a violator.
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if a + b > n:
                break # Since c > b, a+c and b+c will also be > n
            for c in range(b + 1, n + 1):
                if a + c > n:
                    break # b+c will also be > n
                if b + c > n:
                    break
                # If we reach here, a+b, a+c, b+c are all <= 128.
                # This triple (a,b,c) is a candidate for being a "forbidden triple"
                # if all 6 values (a,b,c,a+b,a+c,b+c) are in S.
                bad_triples.append((a, b, c))
    return bad_triples

def _is_valid_set(S, bad_triples):
    """Check if set S contains any forbidden triple."""
    S_set = set(S)
    for a, b, c in bad_triples:
        if a in S_set and b in S_set and c in S_set:
            # Check if sums are also in S
            if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                return False
    return True

def _greedy_construct(order, bad_triples):
    """
    Greedily construct a valid set by adding elements in the given order.
    An element is added if it doesn't create a new forbidden triple.
    """
    S = set()
    S_list = []
    # To check if adding x creates a forbidden triple, we need to check all pairs (y,z) in S
    # such that {x,y,z} is in bad_triples and sums are in S.
    # Optimized check:
    # When adding x, it can only form a new forbidden triple with pairs (y,z) already in S.
    # We can precompute for each x, the list of pairs (y,z) such that {x,y,z} is a bad triple.
    # Let's precompute this mapping for speed.
    
    # We will do this mapping inside the function or globally? 
    # Let's assume bad_triples is passed. We can build the map on the fly for each restart? 
    # No, build it once outside construct and pass it? 
    # Since construct is the entry point, I'll build it inside construct.
    
    return S_list

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    # Precompute bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if a + b > n:
                break
            for c in range(b + 1, n + 1):
                if a + c > n:
                    break
                if b + c > n:
                    break
                bad_triples.append((a, b, c))
    
    # Precompute conflict map: for each x, list of pairs (y,z) with y<z, y!=x, z!=x
    # such that (min, mid, max) is in bad_triples.
    # Actually, just list of (y,z) pairs that with x form a bad triple.
    # We need to check if y,z are in S AND sums are in S.
    # Sums depend on x,y,z.
    
    # To make greedy fast:
    # When considering adding x to S:
    # For each pair (y, z) in S:
    #   Let T = {x, y, z} sorted.
    #   If T is in bad_triple_set:
    #      Check if x+y, x+z, y+z are in S.
    #      If yes, conflict.
    
    # Precompute set of bad triples for O(1) lookup
    bad_triple_set = set(bad_triples)
    
    def try_add_greedy(x, S, S_set):
        """
        Returns True if x can be added to S without violating constraints.
        S is a list, S_set is a set.
        """
        # Check against all pairs in S
        # Optimization: Only check pairs (y,z) where y,z < x? No, x can be smaller.
        # Just iterate all pairs in S.
        # S size is at most 64-70, so pairs ~ 2000. 
        # 128 elements * 2000 pairs = 256k checks per full pass. 
        # 256k is fast in Python? Maybe ~0.1s.
        # 128 elements to add. 128 * 0.1s = 12.8s per greedy pass.
        # We have 85s. So we can do ~5-6 greedy passes.
        
        for i in range(len(S)):
            y = S[i]
            for j in range(i + 1, len(S)):
                z = S[j]
                # Form triple
                a, b, c = sorted((x, y, z))
                if (a, b, c) in bad_triple_set:
                    # Check sums
                    if (x + y) in S_set and (x + z) in S_set and (y + z) in S_set:
                        return False
        return True

    best_size = 0
    best_set = set()
    
    # Strategy: Multiple randomized greedy passes.
    # Order 1: High to Low
    # Order 2: Low to High
    # Order 3: Random Shuffles
    
    orders = [
        list(range(n, 0, -1)),
        list(range(1, n + 1))
    ]
    
    # Add some random orders
    for _ in range(5):
        rand_order = list(range(1, n + 1))
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        S = []
        S_set = set()
        
        for x in order:
            if x in S_set:
                continue
            if try_add_greedy(x, S, S_set):
                S.append(x)
                S_set.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S_set
            
            # Optional: Local Search on the best found so far
            # Try to add any remaining element? 
            # If try_add_greedy failed for x in the main loop, it might succeed 
            # if we remove some elements. But simple add-only is done.
            
            # Let's do a simple "fill" pass: try adding any x not in S_set
            # in any order, just in case the greedy order blocked it but 
            # a different subset of S would allow it? 
            # No, if x couldn't be added to S, it can't be added to any subset of S 
            # unless we remove elements that caused the conflict.
            # So simple fill won't help unless we remove.
            
            # Let's try a few swaps?
            # For each x not in S_set, find conflicts.
            # If removing 1 element allows adding x, size stays same.
            # If removing 0 allows adding x, size increases.
            # We already tried adding all x.
            
    return list(best_set)
# EVOLVE-BLOCK-END
