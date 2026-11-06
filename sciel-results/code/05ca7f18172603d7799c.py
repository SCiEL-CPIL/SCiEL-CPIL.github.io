import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Find all bad triples and build conflict lists
    # conflicts[x] is a list of tuples (a, b) such that {x, a, b} is a bad triple
    # Note: a < b, and a,b != x
    conflicts = [[] for _ in range(N+1)]
    
    # To speed up, we can iterate a < b < c
    # If gcd(a,b) == gcd(b,c) == gcd(c,a), then (a,b,c) is bad.
    # This adds (b,c) to conflicts[a], (a,c) to conflicts[b], (a,b) to conflicts[c]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        # Bad triple found
                        conflicts[a].append((b, c))
                        conflicts[b].append((a, c))
                        conflicts[c].append((a, b))

    def is_valid_add(S_set, x):
        # Check if x can be added to S_set without creating a bad triple
        # A bad triple is formed if there exist a, b in S_set such that (a,b) in conflicts[x]
        for a, b in conflicts[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if is_valid_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def local_search(S, S_set):
        # Returns best set found in this local search run
        best_S = list(S)
        best_S_set = set(S_set)
        best_size = len(S)
        
        # We use a loop that tries to add, then swap, then perturb
        # To avoid infinite loops, we limit steps or use a timer
        
        steps = 0
        max_steps = 5000 # Adjust based on speed
        
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            # Phase 1: Try to add an element
            added = False
            # Try elements not in S
            # Random order for diversity? Or sorted by conflict count?
            # Let's try random order for additions to escape local minima
            candidates = [x for x in range(1, N+1) if x not in S_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(S_set, x):
                    S.append(x)
                    S_set.add(x)
                    added = True
                    # Update best if size increased
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                        best_S_set = set(S_set)
                    break # Add one at a time, then re-evaluate
            
            if added:
                continue # Restart loop, try to add more
            
            # Phase 2: Try to swap
            # If we can't add, try swapping u in S with v not in S
            # This keeps size constant but changes composition
            # We do this to potentially unlock additions
            
            # Heuristic: Pick a random u in S to remove
            # Then try to find a v not in S that can be added to S \ {u}
            
            # To make this efficient, we can limit the number of swap attempts per "stuck" state
            swap_made = False
            # Try a few random swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                if not S:
                    break
                u = random.choice(S)
                S.remove(u)
                S_set.remove(u)
                
                # Now try to add some v
                candidates_v = [x for x in range(1, N+1) if x not in S_set]
                random.shuffle(candidates_v)
                
                found_v = None
                for v in candidates_v:
                    if is_valid_add(S_set, v):
                        found_v = v
                        break
                
                if found_v is not None:
                    S.append(found_v)
                    S_set.add(found_v)
                    swap_made = True
                    # We don't update best_size here because size didn't change
                    # But we continue local search from this new state
                else:
                    # Revert swap
                    S.append(u)
                    S_set.add(u)
                    break # If we can't add anything after removing u, revert and try another u? 
                    # Actually, if we revert, we are back to the same state. 
                    # So we should try a different u.
            
            if not swap_made:
                # If no swap was made (or all reverted), we are stuck.
                # Perturb: randomly remove a few elements to break the local optimum
                if len(S) > 5:
                    # Remove 1-3 random elements
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if S:
                            u = random.choice(S)
                            S.remove(u)
                            S_set.remove(u)
                else:
                    # If set is very small, restart greedy?
                    # For now, just break and rely on outer restarts
                    break

        return best_S, best_S_set, best_size

    best_size_overall = 0
    best_set_overall = []
    
    # Run multiple restarts with different greedy orderings
    # Ordering strategies:
    # 1. Random
    # 2. Sorted by number of conflicts (fewest first)
    # 3. Sorted by value
    
    # Precompute conflict counts for ordering
    conflict_counts = [0] * (N+1)
    for i in range(1, N+1):
        conflict_counts[i] = len(conflicts[i])
    
    order_by_conflicts = sorted(range(1, N+1), key=lambda x: (conflict_counts[x], x))
    
    num_restarts = 50 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose an ordering
        if i % 3 == 0:
            order = list(range(1, N+1))
            random.shuffle(order)
        elif i % 3 == 1:
            order = list(order_by_conflicts)
        else:
            order = list(range(1, N+1))
            # Sort by value, maybe reverse?
            if i % 6 == 2:
                order.reverse()
        
        S, S_set = greedy_construct(order)
        S, S_set, size = local_search(S, S_set)
        
        if size > best_size_overall:
            best_size_overall = size
            best_set_overall = S
            
        # Early exit if we reached theoretical max?
        # Max is N=128, but likely much lower.
        # If we find a very large set, we can stop.
        if best_size_overall > 100: # Just a guess for early stop
             pass

    return best_set_overall
