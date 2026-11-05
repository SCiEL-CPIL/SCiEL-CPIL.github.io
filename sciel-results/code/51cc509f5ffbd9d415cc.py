import random
import time

N = 512
# Precompute AP checks
# ap_checks[i] contains triples (a, b, c) such that {i, a, b, c} is a 4-AP
# i can be 1st, 2nd, 3rd, or 4th term.
ap_checks = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a = start
        b = start + d
        c = start + 2 * d
        e = start + 3 * d
        # 4-AP: a, b, c, e
        # If we add a, need b, c, e in S
        ap_checks[a].append((b, c, e))
        # If we add b, need a, c, e in S
        ap_checks[b].append((a, c, e))
        # If we add c, need a, b, e in S
        ap_checks[c].append((a, b, e))
        # If we add e, need a, b, c in S
        ap_checks[e].append((a, b, c))

def can_add(i, in_set):
    for a, b, c in ap_checks[i]:
        if in_set[a] and in_set[b] and in_set[c]:
            return False
    return True

def greedy_construction(order):
    in_set = [False] * (N + 1)
    size = 0
    for i in order:
        if can_add(i, in_set):
            in_set[i] = True
            size += 1
    return in_set, size

def local_search(in_set):
    # Returns improved in_set and size
    size = sum(in_set[1:])
    improved = True
    while improved:
        improved = False
        # 1. Try to add any missing element
        for i in range(1, N + 1):
            if not in_set[i]:
                if can_add(i, in_set):
                    in_set[i] = True
                    size += 1
                    improved = True
        # 2. Try 1-swaps if no adds were made
        if not improved:
            # Get lists of in and out
            in_list = [i for i in range(1, N + 1) if in_set[i]]
            out_list = [i for i in range(1, N + 1) if not in_set[i]]
            
            # To speed up, we can just check if a swap allows an immediate add
            # This is expensive. Let's just do a limited number of random swaps or all swaps if N is small.
            # N=512, |S|~150, |S^c|~360. 150*360 = 54000 swaps.
            # For each swap, we check if y fits in S\x.
            # Then we check if any z fits in S' = S\x+{y}.
            # This is too slow for a full loop.
            
            # Alternative: Randomized Swap Search
            # Pick random x in S, random y not in S.
            # If y fits in S\x, do the swap.
            # Then try to add any z.
            
            # Let's stick to just the "Add" phase for the polish, 
            # and rely on multiple restarts for the main optimization.
            # The "Add" phase is very fast.
            pass
            
    return in_set, size

def construct():
    start_time = time.time()
    best_in_set = None
    best_size = 0
    
    # 1. Deterministic Greedy (Natural, Reverse)
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        in_set, size = greedy_construction(order)
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # 2. Randomized Greedy
    random.seed(42)
    while time.time() - start_time < 55: # Leave 5s margin
        order = list(range(1, N + 1))
        random.shuffle(order)
        in_set, size = greedy_construction(order)
        
        # Polish with local search (Add phase only)
        in_set, size = local_search(in_set)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # Convert to sorted list
    if best_in_set is None:
        # Fallback
        return [1]
        
    return [i for i in range(1, N + 1) if best_in_set[i]]
