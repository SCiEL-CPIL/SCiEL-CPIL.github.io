"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    n = 128
    universe = list(range(1, n + 1))
    
    # Bitset representation: bit i set means (i+1) is in the set
    # Using a Python int as a bitset for fast membership
    
    def make_bitset(elements):
        bs = 0
        for e in elements:
            bs |= (1 << (e - 1))
        return bs
    
    def bitset_contains(bs, x):
        return (bs >> (x - 1)) & 1
    
    def bitset_add(bs, x):
        return bs | (1 << (x - 1))
    
    def bitset_remove(bs, x):
        return bs & ~(1 << (x - 1))
    
    def bitset_count(bs):
        return bin(bs).count('1')
    
    def bitset_to_set(bs):
        return set(i + 1 for i in range(n) if (bs >> i) & 1)
    
    # Check if adding x to set S (represented by bitset bs and set S) creates a violation
    # A violation: distinct a,b,c in S∪{x} such that a+b, a+c, b+c all in S∪{x}
    # When adding x, new violations must involve x. So check pairs (a,b) in S:
    # if a+b in S∪{x}, x+a in S∪{x}, x+b in S∪{x}, and a,b,x distinct
    def creates_violation(bs, S, x):
        for a in S:
            if a == x:
                continue
            for b in S:
                if b == x or b == a:
                    continue
                # Check if (a, b, x) is a violation
                # Need: a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
                sum_ab = a + b
                sum_ax = a + x
                sum_bx = b + x
                # All three sums must be in S∪{x}
                if (bitset_contains(bs, sum_ab) or sum_ab == x) and \
                   (bitset_contains(bs, sum_ax) or sum_ax == x) and \
                   (bitset_contains(bs, sum_bx) or sum_bx == x):
                    return True
        return False
    
    # Greedy construction: try adding elements in different orders
    def greedy(order):
        bs = 0
        S = set()
        for x in order:
            if not creates_violation(bs, S, x):
                bs = bitset_add(bs, x)
                S.add(x)
        return bs, S
    
    # Local search: try to improve by removing one element and adding another
    def local_search(bs, S, max_iters=10000):
        S_list = list(S)
        random.shuffle(S_list)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try removing a random element and adding a random non-element
            if not S_list:
                break
            a = random.choice(S_list)
            candidates = [x for x in universe if x not in S]
            if not candidates:
                break
            b = random.choice(candidates)
            
            # Remove a
            bs_new = bitset_remove(bs, a)
            S_new = S - {a}
            
            # Try to add b
            if not creates_violation(bs_new, S_new, b):
                bs_new = bitset_add(bs_new, b)
                S_new.add(b)
                if len(S_new) > len(S):
                    bs = bs_new
                    S = S_new
                    S_list = list(S)
                    random.shuffle(S_list)
                # Even if same size, accept with some probability for diversity
                # But for now, only accept improvements
        return bs, S
    
    # Try multiple random orders for greedy
    best_bs = 0
    best_S = set()
    
    # First, try a natural order
    order = universe[:]
    bs, S = greedy(order)
    best_bs = bs
    best_S = S
    
    # Try reversed
    order = universe[::-1]
    bs, S = greedy(order)
    if len(S) > len(best_S):
        best_bs = bs
        best_S = S
    
    # Try random orders
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        order = universe[:]
        random.shuffle(order)
        bs, S = greedy(order)
        if len(S) > len(best_S):
            best_bs = bs
            best_S = S
    
    # Local search to improve
    if time.time() < deadline - 5:
        best_bs, best_S = local_search(best_bs, best_S, max_iters=50000)
    
    # Final check and return
    result = sorted(best_S)
    # Verify validity
    S_set = set(result)
    valid = True
    for a, b, c in combinations(result, 3):
        if (a + b in S_set) and (a + c in S_set) and (b + c in S_set):
            valid = False
            break
    
    if not valid:
        # Fallback to a safe set: just the upper half
        return list(range(n//2 + 1, n + 1))
    
    return result

# EVOLVE-BLOCK-END
