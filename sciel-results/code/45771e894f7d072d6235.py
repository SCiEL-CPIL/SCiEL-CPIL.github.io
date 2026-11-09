import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute bad triple relationships
    # bad_triplets[x][y] is a set of u such that (x, u, y) is a bad triple
    # We only need this for x != u != y != x
    bad_triplets = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    
    # Compute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            for u in range(y + 1, N + 1):
                g_xy = gcd_table[x][y]
                g_xu = gcd_table[x][u]
                g_yu = gcd_table[y][u]
                if g_xy == g_xu == g_yu:
                    # (x, y, u) is a bad triple
                    bad_triplets[x][y].add(u)
                    bad_triplets[x][u].add(y)
                    bad_triplets[y][x].add(u)
                    bad_triplets[y][u].add(x)
                    bad_triplets[u][x].add(y)
                    bad_triplets[u][y].add(x)

    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def score_number(n):
        pf = prime_factors(n)
        if not pf:
            return 0
        return max(pf)

    best_size = 0
    best_set = set()
    
    # Try multiple initializations
    initializations = [
        sorted(elements, key=score_number, reverse=True),
        sorted(elements, key=lambda x: len(prime_factors(x))),
        sorted(elements, key=lambda x: x),
        list(elements)
    ]
    
    # Add some random shuffles
    random.seed(42)
    for _ in range(5):
        shuffled = elements.copy()
        random.shuffle(shuffled)
        initializations.append(shuffled)

    for order in initializations:
        if time.time() > deadline:
            break
            
        s = set()
        conflict_count = [0] * (N + 1)
        
        # Greedy Build
        for x in order:
            if conflict_count[x] == 0:
                s.add(x)
                # Update conflict counts for all other y
                for y in range(1, N + 1):
                    if y == x or y in s:
                        continue
                    # How many u in s such that (y, u, x) is bad?
                    # bad_triplets[y][x] contains all u.
                    # We need to count how many are in s.
                    count = 0
                    for u in bad_triplets[y][x]:
                        if u in s:
                            count += 1
                    conflict_count[y] += count
        
        current_size = len(s)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Add
            added_any = False
            for x in range(1, N + 1):
                if x not in s and conflict_count[x] == 0:
                    s.add(x)
                    # Update conflict counts
                    for y in range(1, N + 1):
                        if y == x or y in s:
                            continue
                        count = 0
                        for u in bad_triplets[y][x]:
                            if u in s:
                                count += 1
                        conflict_count[y] += count
                    added_any = True
                    improved = True
                    break # Restart loop to check for more adds
            
            if not added_any:
                # Phase 2: 1-for-2 Swap
                found_swap = False
                # Shuffle s for randomness in swap search
                s_list = list(s)
                random.shuffle(s_list)
                
                for u in s_list:
                    if time.time() > deadline:
                        break
                    
                    # Temporarily remove u
                    s.remove(u)
                    
                    # Find addable candidates
                    candidates = []
                    for v in range(1, N + 1):
                        if v not in s and conflict_count[v] == 0:
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        # Pick two candidates
                        v1, v2 = candidates[0], candidates[1]
                        # Add v1
                        s.add(v1)
                        # Update conflict counts for adding v1
                        for y in range(1, N + 1):
                            if y == v1 or y in s:
                                continue
                            count = 0
                            for w in bad_triplets[y][v1]:
                                if w in s:
                                    count += 1
                            conflict_count[y] += count
                        
                        # Add v2
                        # Note: conflict_count[v2] was 0 before adding v1.
                        # But adding v1 might have increased it.
                        # We must check if v2 is still addable.
                        if conflict_count[v2] == 0:
                            s.add(v2)
                            # Update conflict counts for adding v2
                            for y in range(1, N + 1):
                                if y == v2 or y in s:
                                    continue
                                count = 0
                                for w in bad_triplets[y][v2]:
                                    if w in s:
                                        count += 1
                                conflict_count[y] += count
                            
                            # Success! Size increased by 1 (removed 1, added 2)
                            improved = True
                            found_swap = True
                            break
                        else:
                            # v2 is not addable anymore, revert v1?
                            # Or try a different pair?
                            # For simplicity, revert v1 and try next u
                            # This is expensive. Let's just revert and move on.
                            pass
                    else:
                        # Try 1-for-1 swap (neutral move) to escape local minima
                        if len(candidates) == 1:
                            v1 = candidates[0]
                            # Check if this swap is "different" from current
                            # It's always different since u is removed and v1 added
                            # Accept with some probability or always?
                            # Let's accept to explore
                            s.add(v1)
                            for y in range(1, N + 1):
                                if y == v1 or y in s:
                                    continue
                                count = 0
                                for w in bad_triplets[y][v1]:
                                    if w in s:
                                        count += 1
                                conflict_count[y] += count
                            improved = True
                            found_swap = True
                            break
                            
                    # Revert u if no successful swap found in this iteration
                    if not found_swap:
                        s.add(u)
                        # Revert conflict counts
                        for y in range(1, N + 1):
                            if y == u or y in s:
                                continue
                            count = 0
                            for w in bad_triplets[y][u]:
                                if w in s:
                                    count += 1
                            conflict_count[y] -= count

                if found_swap:
                    continue

        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()

    return list(best_set)
