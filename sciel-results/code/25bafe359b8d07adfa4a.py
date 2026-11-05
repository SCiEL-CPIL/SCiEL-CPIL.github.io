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

def local_search(in_set, deadline=None):
    """
    Directed local search with systematic swap escape.
    Phase 1: Add any element that fits.
    Phase 2: Systematic 1-swap search - try removing x and adding y,
             then check if any z can be added to gain net +1.
    """
    size = sum(in_set[1:])
    improved = True
    while improved:
        improved = False
        if deadline and time.time() > deadline:
            break
            
        # Phase 1: Try to add any missing element
        for i in range(1, N + 1):
            if deadline and time.time() > deadline:
                break
            if not in_set[i]:
                if can_add(i, in_set):
                    in_set[i] = True
                    size += 1
                    improved = True
                    # After adding, restart the add phase since other elements may now be blocked
                    break
        
        if improved:
            continue
            
        if deadline and time.time() > deadline:
            break
            
        # Phase 2: Systematic swap search
        # For each x in S, try removing it and adding y not in S.
        # Then check if we can add another z to gain net +1.
        # To be efficient, we only do this when we're stuck.
        
        in_list = [i for i in range(1, N + 1) if in_set[i]]
        out_list = [i for i in range(1, N + 1) if not in_set[i]]
        
        # Try systematic swaps: for each x in in_list, for each y in out_list
        # Check if y fits in S\x, then if any z fits in S\x+{y}
        # This is O(|S| * |S^c| * (cost of can_add)) which could be slow.
        # Optimization: only try swaps where y is "close" to fitting.
        
        swap_found = False
        for x in in_list:
            if deadline and time.time() > deadline:
                break
            if swap_found:
                break
            # Temporarily remove x
            in_set[x] = False
            size -= 1
            
            # Try adding each y
            for y in out_list:
                if deadline and time.time() > deadline:
                    break
                if swap_found:
                    break
                if can_add(y, in_set):
                    # y fits in S\x
                    in_set[y] = True
                    size += 1
                    # Now try to add any z to gain net +1
                    for z in out_list:
                        if z == y:
                            continue
                        if deadline and time.time() > deadline:
                            break
                        if can_add(z, in_set):
                            # We can add z, net gain of +1
                            in_set[z] = True
                            size += 1
                            swap_found = True
                            break
                    if not swap_found:
                        # Undo y addition, keep x removed? No, undo both.
                        in_set[y] = False
                        size -= 1
                    else:
                        # Keep y and z, x is removed. Net +1.
                        out_list.remove(y)
                        out_list.remove(z)
                        in_list.remove(x)
                        in_list.append(y)
                        in_list.append(z)
                        break
            
            # Restore x if swap wasn't found
            if not swap_found:
                in_set[x] = True
                size += 1
                break
        
        if swap_found:
            improved = True
    
    return in_set, size

def construct():
    start_time = time.time()
    best_in_set = None
    best_size = 0
    deadline = start_time + 55  # Leave 5s margin
    
    # 1. Deterministic Greedy (Natural, Reverse)
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        in_set, size = greedy_construction(order)
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # 2. Polish the best deterministic result
    if best_in_set:
        in_set, size = local_search(best_in_set[:], deadline)
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # 3. Randomized Greedy with local search
    random.seed(42)
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        in_set, size = greedy_construction(order)
        
        # Polish with local search
        in_set, size = local_search(in_set, deadline)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            # Update deadline slightly if we're doing well? No, keep fixed.
            
    # Convert to sorted list
    if best_in_set is None:
        # Fallback
        return [1]
        
    return [i for i in range(1, N + 1) if best_in_set[i]]
