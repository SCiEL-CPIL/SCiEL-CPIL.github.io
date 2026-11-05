import time
import random

def construct():
    n = 512
    # Precompute all 4-term arithmetic progressions
    # ap_list[x] contains tuples of the other 3 elements for each AP containing x
    ap_list = [[] for _ in range(n + 1)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            a = start
            b = start + d
            c = start + 2 * d
            e = start + 3 * d
            # Store the other 3 elements for each member
            ap_list[a].append((b, c, e))
            ap_list[b].append((a, c, e))
            ap_list[c].append((a, b, e))
            ap_list[e].append((a, b, c))

    def is_valid(x, S):
        """Check if x can be added to S without forming a 4-AP."""
        for other_3 in ap_list[x]:
            if all(elem in S for elem in other_3):
                return False
        return True

    # Greedy Initialization
    S = set()
    for x in range(1, n + 1):
        if is_valid(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Local Search
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Add all valid elements
        while True:
            added_any = False
            # Iterate over a static list to avoid modification during iteration issues, 
            # though we are only adding to S, so checking x not in S is fine.
            for x in range(1, n + 1):
                if x not in S and is_valid(x, S):
                    S.add(x)
                    added_any = True
            if not added_any:
                break
            improved = True
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            if time.time() > deadline:
                return sorted(best_S)

        # Phase 2: Try 1-2 swaps (remove 1, add k>=2)
        # To speed up, we can just check if any swap improves the set.
        # We iterate through y in S.
        for y in list(S):
            S_temp = S - {y}
            gain = 0
            new_elements = []
            # Check all x not in S_temp
            for x in range(1, n + 1):
                if x not in S_temp and is_valid(x, S_temp):
                    gain += 1
                    new_elements.append(x)
            
            # We removed 1 element (y). We add 'gain' elements.
            # Net change = gain - 1.
            # We want net change > 0, so gain >= 2.
            if gain >= 2:
                S = S_temp | set(new_elements)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break # Restart Phase 1 with new S
        
        if improved:
            continue

        # Phase 3: Perturbation to escape local optimum
        # Remove a few random elements
        if len(S) > 10:
            # Remove 5% of the set, min 5 elements
            k = max(5, len(S) // 20)
            to_remove = random.sample(list(S), k)
            for y in to_remove:
                S.remove(y)
        else:
            # If set is small, just restart with greedy
            S = set()
            for x in range(1, n + 1):
                if is_valid(x, S):
                    S.add(x)

    return sorted(best_S)
