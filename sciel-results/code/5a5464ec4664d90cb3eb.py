import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} is a 3-AP
# and a, b != x
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        # Case 1: x is first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        
        # Case 2: x is middle term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        # Case 3: x is last term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

def can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    """
    for a, b in ap_pairs[x]:
        # Check if both a and b are in the bitset
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def greedy_construct(order):
    """
    Greedy construction: add elements in the given order if they don't create a 3-AP.
    Returns (bitset, score) where score is the exact sum of reciprocals as a Fraction.
    """
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

# Precompute suffix sums of reciprocals for bounding in B&B
suffix_sums = [0.0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sums[i] = suffix_sums[i + 1] + 1.0 / i

# Precompute conflict bitsets: conflicts[x] is a bitset of elements that form a 3-AP with x
# (i.e., there exists a third element such that all three form a 3-AP)
conflicts = [0] * (N + 1)
for x in range(1, N + 1):
    mask = 0
    for a, b in ap_pairs[x]:
        mask |= (1 << (a - 1))
        mask |= (1 << (b - 1))
    conflicts[x] = mask

def can_add_fast(bitset, x):
    """Fast conflict check using precomputed conflict bitset."""
    return (bitset & conflicts[x]) == 0

def local_search(bitset, score, deadline, max_rounds=200):
    """
    Enhanced Local Search:
    1. Greedy Add
    2. 1-Swap
    3. Perturbation (Random Removal) if stuck
    """
    improved = True
    rounds = 0
    stuck_rounds = 0
    
    while time.time() < deadline and rounds < max_rounds:
        improved = False
        rounds += 1
        
        # Phase 1: Try to add any element that can be added
        for x in range(1, N + 1):
            if not (bitset & (1 << (x - 1))):
                if can_add_fast(bitset, x):
                    bitset |= (1 << (x - 1))
                    score += recip_frac[x]
                    improved = True
        
        # Phase 2: Try single swaps
        elements_in_set = [x for x in range(1, N + 1) if bitset & (1 << (x - 1))]
        random.shuffle(elements_in_set) # Randomize order to escape local optima
        
        for y in elements_in_set:
            if time.time() > deadline:
                return bitset, score
            # Try removing y
            trial_bitset = bitset & ~(1 << (y - 1))
            trial_score = score - recip_frac[y]
            
            # Try adding any x not in the set
            # We prioritize smaller x (higher reciprocal)
            for x in range(1, N + 1):
                if x == y: continue
                if trial_bitset & (1 << (x - 1)):
                    continue
                if can_add_fast(trial_bitset, x):
                    new_score = trial_score + recip_frac[x]
                    if new_score > score:
                        bitset = trial_bitset | (1 << (x - 1))
                        score = new_score
                        improved = True
                        break
        
        if not improved:
            stuck_rounds += 1
            # If stuck for a few rounds, perturb
            if stuck_rounds > 5:
                # Remove a random element
                if elements_in_set:
                    y = random.choice(elements_in_set)
                    bitset &= ~(1 << (y - 1))
                    score -= recip_frac[y]
                    stuck_rounds = 0
                    # Immediately try to add back or others
                    improved = True 
        else:
            stuck_rounds = 0
            
    return bitset, score

# Precompute reciprocal Fractions for speed
recip_frac = [Fraction(0, 1)] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)

def branch_and_bound(deadline, init_bitset=0, init_score=None):
    """
    Branch and Bound search to maximize the reciprocal sum.
    """
    if init_score is None:
        init_score = Fraction(0, 1)
        
    best_score = init_score
    best_bitset = init_bitset
    best_score_float = float(init_score)
    
    state = {
        'best_score': best_score,
        'best_bitset': best_bitset,
        'best_score_float': best_score_float,
        'deadline': deadline
    }
    
    def compute_compatible_bound(x, bitset, score_float):
        bound = 0.0
        for i in range(x, N + 1):
            if (bitset & conflicts[i]) == 0:
                bound += 1.0 / i
        return score_float + bound
    
    def dfs(x, bitset, score, score_float):
        if time.time() > state['deadline']:
            return
        
        if score_float + suffix_sums[x] > state['best_score_float'] + 1e-15:
            tight_bound = compute_compatible_bound(x, bitset, score_float)
            if tight_bound <= state['best_score_float'] + 1e-15:
                return
        
        if x > N:
            if score > state['best_score']:
                state['best_score'] = score
                state['best_bitset'] = bitset
                state['best_score_float'] = score_float
            return
        
        if can_add_fast(bitset, x):
            new_bitset = bitset | (1 << (x - 1))
            new_score = score + recip_frac[x]
            new_score_float = score_float + 1.0 / x
            dfs(x + 1, new_bitset, new_score, new_score_float)
        
        dfs(x + 1, bitset, score, score_float)

    # Start DFS from x=1
    # Note: We don't do initial greedy here because construct() does it.
    # We just start the search with the provided incumbent.
    dfs(1, 0, Fraction(0, 1), 0.0)
    
    return state['best_bitset'], state['best_score']

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bitset = 0
    best_score = Fraction(0, 1)
    best_score_float = 0.0
    
    # Multi-start initialization
    # 1. Sorted (Reciprocal) order
    order_recip = list(range(1, N + 1))
    b, s = greedy_construct(order_recip)
    b, s = local_search(b, s, deadline, max_rounds=50)
    if float(s) > best_score_float:
        best_bitset = b
        best_score = s
        best_score_float = float(s)
        
    # 2. Random orders
    for _ in range(50):
        if time.time() > deadline - 10: break # Save time for B&B
        order = list(range(1, N + 1))
        random.shuffle(order)
        b, s = greedy_construct(order)
        b, s = local_search(b, s, deadline, max_rounds=20)
        if float(s) > best_score_float:
            best_bitset = b
            best_score = s
            best_score_float = float(s)
            
    # Run Branch and Bound with the best incumbent
    # Pass a slightly earlier deadline to ensure we return in time
    bb_deadline = deadline - 2.0
    if time.time() < bb_deadline:
        # Modify state in branch_and_bound to use current best
        # We need to pass the current best to branch_and_bound
        # The current branch_and_bound does its own init. 
        # Let's modify branch_and_bound to accept an initial solution?
        # Or just let it run. It might find something better.
        # However, branch_and_bound is slow. 
        # Let's just run it if there is significant time left.
        best_bitset, best_score = branch_and_bound(bb_deadline, best_bitset, best_score)
        
    # Final polish
    if time.time() < deadline:
        best_bitset, best_score = local_search(best_bitset, best_score, deadline, max_rounds=100)
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
