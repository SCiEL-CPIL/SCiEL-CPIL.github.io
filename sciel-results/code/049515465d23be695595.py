import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin (limit is 60s)
    
    N = 128
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute Conflict Lists
    # conflict[x] is a list of pairs (a, b) such that (x, a, b) is a bad triple.
    # i.e., gcd(x,a) == gcd(x,b) == gcd(a,b)
    conflict = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_mat[a][b]
                g_ac = gcd_mat[a][c]
                g_bc = gcd_mat[b][c]
                if g_ab == g_ac and g_ab == g_bc:
                    # Triple (a, b, c) is bad
                    # For x=a, the other two are b, c
                    conflict[a].append((b, c))
                    # For x=b, the other two are a, c
                    conflict[b].append((a, c))
                    # For x=c, the other two are a, b
                    conflict[c].append((a, b))

    def can_add(x, in_set):
        """
        Check if x can be added to the set represented by in_set (list of bools).
        Uses precomputed conflict lists for O(1) per pair check.
        """
        for a, b in conflict[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def greedy_build(order, in_set=None):
        """Build set greedily following given order."""
        if in_set is None:
            in_set = [False] * (N + 1)
        mask = 0
        for x in order:
            if not in_set[x]:
                if can_add(x, in_set):
                    in_set[x] = True
                    mask |= (1 << x)
        return mask, in_set

    def local_search(mask, in_set, max_iters=500):
        """
        Improve the set via additions and kicks (removals followed by re-optimization).
        Returns the best mask found.
        """
        best_mask = mask
        best_size = bin(mask).count('1')
        
        iters = 0
        while iters < max_iters and time.time() < deadline:
            iters += 1
            improved = False
            
            # 1. Try to add any element
            # We iterate through all possible additions to ensure we reach a local maximum
            for x in range(1, N + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        mask |= (1 << x)
                        best_mask = mask
                        best_size += 1
                        improved = True
                        break # Restart scan to catch new additions enabled by this one
            
            if improved:
                continue

            # 2. Local maximum reached. Try kicks.
            # Get list of elements in set
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            if not current_elements:
                break
            
            # Try a few random kicks to escape local optimum
            found_better = False
            # Number of kicks can be tuned; 10-20 is often good
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Randomly pick an element to remove
                u = random.choice(current_elements)
                
                # Save state
                old_in_set = in_set[:]
                old_mask = mask
                old_size = best_size
                
                # Remove u
                in_set[u] = False
                mask ^= (1 << u)
                
                # Greedily add back as many as possible
                add_count = 0
                # Iterate multiple times to allow cascading additions
                changed = True
                while changed:
                    changed = False
                    for x in range(1, N + 1):
                        if not in_set[x]:
                            if can_add(x, in_set):
                                in_set[x] = True
                                mask |= (1 << x)
                                add_count += 1
                                changed = True
                                break
                                
                new_size = bin(mask).count('1')
                
                # If we added more than we removed (1), we improved
                if new_size > old_size:
                    best_mask = mask
                    best_size = new_size
                    found_better = True
                    # Update current_elements for next kick iteration
                    current_elements = [i for i in range(1, N + 1) if in_set[i]]
                    break
                else:
                    # Revert
                    in_set = old_in_set
                    mask = old_mask
                    best_size = old_size
            
            if not found_better:
                # If we couldn't find an improvement with random kicks,
                # try a more systematic kick: remove the element that is in the most conflicts?
                # Or just break to allow outer loop to restart.
                # Breaking is safer to avoid getting stuck in a loop that doesn't improve.
                break
                
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy with natural order
    mask, in_set = greedy_build(range(1, N + 1))
    m = local_search(mask, in_set)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
        
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline:
        mask, in_set = greedy_build(range(N, 0, -1))
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy 3: Iterated Local Search / Random Restarts
    # We use the remaining time to try random initializations and local search
    while time.time() < deadline - 2:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        mask, in_set = greedy_build(random_order)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')
            
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
