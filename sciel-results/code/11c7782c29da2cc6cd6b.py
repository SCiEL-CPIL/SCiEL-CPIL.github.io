"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    triples_set = set()
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    triples_set.add((a, b, c))
                    
    if time.time() > deadline - 10:
        # Fallback to simple greedy if precomputation is too slow (unlikely for 128)
        selected = []
        for x in range(1, 129):
            if all(not (lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected

    # Build conflict map: for each x, which pairs (u, v) with u < v form a bad triple with x?
    conflict = {i: [] for i in range(1, n + 1)}
    for a, b, c in triples_set:
        # (a, b, c) is a bad triple
        # For x=c, the conflicting pair is (a, b)
        # For x=b, the conflicting pair is (a, c)
        # For x=a, the conflicting pair is (b, c)
        conflict[c].append((a, b))
        conflict[b].append((a, c))
        conflict[a].append((b, c))

    # Convert conflict lists to sets for faster lookup if needed, but lists are fine for small sizes
    # Let's also precompute a boolean matrix for "is this pair a conflict for x"? 
    # No, the list of pairs is the most direct.

    def is_valid(S, x):
        """Check if x can be added to set S without forming a bad triple."""
        for u, v in conflict[x]:
            if u in S and v in S:
                return False
        return True

    def greedy_solve(order):
        S = set()
        for x in order:
            if is_valid(S, x):
                S.add(x)
        return S

    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with various orderings
    # Ordering 1: Ascending conflict count
    conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
    order_asc_conf = sorted(numbers, key=lambda x: conflict_count[x])
    S1 = greedy_solve(order_asc_conf)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Ordering 2: Descending conflict count
    order_desc_conf = sorted(numbers, key=lambda x: -conflict_count[x])
    S2 = greedy_solve(order_desc_conf)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2

    # Ordering 3: Ascending value
    S3 = greedy_solve(numbers)
    if len(S3) > best_size:
        best_size = len(S3)
        best_set = S3

    # Ordering 4: Descending value
    S4 = greedy_solve(sorted(numbers, reverse=True))
    if len(S4) > best_size:
        best_size = len(S4)
        best_set = S4

    # Strategy 2: Local Search (1-swap and add)
    # Start from the best greedy solution
    current_set = set(best_set)
    
    # Try to add any element not in the set
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in numbers:
            if x not in current_set and is_valid(current_set, x):
                current_set.add(x)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
    
    # Try 1-swap: remove one element, try to add another, see if we can grow
    for _ in range(100):  # Limit iterations for local search
        if time.time() > deadline - 5:
            break
        
        trial_set = set(current_set)
        found_swap = False
        
        # Pick a random element to remove? Or iterate?
        # Let's iterate to be deterministic for now, or randomize for diversity?
        # Randomized local search is better.
        import random
        random.seed(int(time.time() * 1000) % 10000)
        
        elements = list(current_set)
        if not elements:
            break
            
        out_elem = random.choice(elements)
        trial_set.remove(out_elem)
        
        # Try to add any element not in trial_set
        for in_elem in numbers:
            if in_elem in trial_set:
                continue
            if is_valid(trial_set, in_elem):
                # Swap found
                trial_set.add(in_elem)
                
                # Now try to add more
                local_improved = True
                while local_improved:
                    local_improved = False
                    for x in numbers:
                        if x not in trial_set and is_valid(trial_set, x):
                            trial_set.add(x)
                            local_improved = True
                
                if len(trial_set) > len(current_set):
                    current_set = trial_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    found_swap = True
                    break
        
        if not found_swap:
            # If no improving swap found, we might be stuck in a local optimum.
            # We could try restarting from a different greedy solution, but for now,
            # let's just break if we've done a few iterations without improvement.
            # Actually, the loop above runs 100 times. If no improvement, it just keeps trying.
            # To avoid infinite loop on no improvement, we should track best size and break if no change for X iterations.
            # But 100 iterations is small, so it's fine.
            pass

    return sorted(list(best_set))

# EVOLVE-BLOCK-END
