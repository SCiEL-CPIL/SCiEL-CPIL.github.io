import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 512
    # 1. Build compatibility graph
    # adj[i] is a set of j such that lcm(i, j) <= 512
    # We only care about i, j in 1..512
    # To save memory and time, we can just check on the fly in the search?
    # No, precomputing is faster for repeated checks.
    # 512 * 512 = 262k checks. ~0.1s
    
    # Actually, we can just use a list of lists or sets.
    # Let's use a bitset or just a list of bools?
    # A set of neighbors is good for intersection operations.
    
    neighbors = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            # Optimization: if i * j > 512 * max(i, j) is not right.
            # lcm(i, j) = i * j // gcd(i, j)
            # If i * j > 512 * gcd(i, j), then lcm > 512.
            # gcd(i, j) <= min(i, j).
            # So if i * j > 512 * min(i, j), i.e., max(i, j) > 512, impossible.
            # But i, j <= 512.
            # Just compute.
            g = math.gcd(i, j)
            if i * j // g <= 512:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # 2. Greedy Initial Solution
    # Sort candidates by degree (descending) or value (ascending)?
    # Small numbers have higher degree.
    # Let's try to build a maximal clique.
    # Greedy: pick node with highest degree in current candidate set.
    
    S = set()
    candidates = set(range(1, N + 1))
    
    while candidates:
        # Pick node with max degree in candidates
        # To speed up, we can just pick the smallest number?
        # Smallest numbers are most compatible.
        # Let's try picking smallest number first.
        v = min(candidates)
        S.add(v)
        # Remove v and all non-neighbors of v from candidates
        new_candidates = set()
        for u in candidates:
            if u == v:
                continue
            if u in neighbors[v]:
                new_candidates.add(u)
        candidates = new_candidates
        
    current_best = S
    best_size = len(S)
    
    # 3. Local Search
    # Try to add nodes. If not possible, try swaps.
    # To keep it simple and fast, we'll do a few iterations of:
    # a) Try to add any node from 1..512 not in S.
    # b) If no add, try to swap u in S with v not in S.
    
    # Precompute "incompatible with S" for speed?
    # Just check on the fly.
    
    def is_compatible(v, S_set):
        for u in S_set:
            if v not in neighbors[u]:
                return False
        return True
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add
        for v in range(1, N + 1):
            if v in current_best:
                continue
            if is_compatible(v, current_best):
                current_best.add(v)
                improved = True
                break  # Restart loop with new S
                
        if not improved:
            # Try to swap
            # For each u in S, for each v not in S:
            # If v compatible with S \ {u}, swap.
            # To limit complexity, only try swaps if S is "stuck".
            # Number of swaps to check: |S| * (512 - |S|) ~ 24 * 488 ~ 12000.
            # Each check is O(|S|). 12000 * 24 = 288k ops. Fast.
            
            found_swap = False
            S_list = list(current_best)
            for u in S_list:
                S_minus_u = current_best - {u}
                for v in range(1, N + 1):
                    if v in current_best:
                        continue
                    if is_compatible(v, S_minus_u):
                        # Perform swap
                        current_best.remove(u)
                        current_best.add(v)
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
            if found_swap:
                # Loop back to try to add more
                continue
            else:
                # No swap found, break
                break
                
    return list(current_best)
