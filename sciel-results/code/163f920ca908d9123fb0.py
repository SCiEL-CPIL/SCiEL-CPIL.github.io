"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Efficient approach using precomputed adjacency bitmasks:
- Precompute for each number 1..512 a bitmask of all numbers it's compatible with
- Use bit operations for fast compatibility checks
- Multiple greedy strategies + local search to find large cliques
"""

def construct():
    import math
    import time
    import random

    t0 = time.time()
    deadline = t0 + 85  # 85s safety margin (limit is 90s)
    LIMIT = 512
    N = LIMIT

    def time_left():
        return deadline - time.time()

    # Precompute adjacency using bitmasks
    adj = [0] * (N + 1)  # 1-indexed, adj[i] is bitmask
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = math.gcd(i, j)
            if (i // g) * j <= LIMIT:
                mask |= (1 << (j - 1))
        adj[i] = mask

    def popcount(mask):
        return bin(mask).count('1')

    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask

    def mask_to_set(mask):
        S = []
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            S.append(idx + 1)
            m ^= lb
        return S

    def compatible_with_all(x, S_mask):
        return (adj[x] & S_mask) == S_mask

    def divisors_of(n):
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)

    # Fast greedy: pick node with max degree in remaining, add to clique
    def greedy_max_degree(order=None):
        if order is None:
            order = list(range(1, N + 1))
        remaining = set_to_mask(order)
        clique = []
        clique_mask = 0
        while remaining and time_left() > 5:
            best_node = None
            best_deg = -1
            m = remaining
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                node = idx + 1
                deg = popcount(adj[node] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
                m ^= lb
            if best_node is None:
                break
            clique.append(best_node)
            clique_mask |= (1 << (best_node - 1))
            remaining &= adj[best_node]
        return clique

    def greedy_by_order(order):
        clique = []
        clique_mask = 0
        for node in order:
            if time_left() < 5:
                break
            if compatible_with_all(node, clique_mask):
                clique.append(node)
                clique_mask |= (1 << (node - 1))
        return clique

    def local_add(clique):
        """Add all compatible nodes greedily."""
        S = list(clique)
        S_set = set(S)
        S_mask = set_to_mask(S)
        changed = True
        while changed and time_left() > 10:
            changed = False
            for x in range(1, N + 1):
                if time_left() < 10:
                    break
                if x not in S_set and compatible_with_all(x, S_mask):
                    S.append(x)
                    S_set.add(x)
                    S_mask |= (1 << (x - 1))
                    changed = True
        return S

    def local_swap(clique):
        """Try 1-for-2 swaps: remove one node, try to add two new nodes."""
        S = list(clique)
        S_set = set(S)
        S_mask = set_to_mask(S)
        improved = True
        while improved and time_left() > 10:
            improved = False
            for i in range(len(S)):
                if time_left() < 10:
                    break
                removed = S[i]
                S_trial = S[:i] + S[i+1:]
                S_trial_set = S_set - {removed}
                S_trial_mask = S_mask & ~(1 << (removed - 1))
                
                # Try to add as many nodes as possible
                added = []
                for x in range(1, N + 1):
                    if time_left() < 10:
                        break
                    if x not in S_trial_set and x != removed and compatible_with_all(x, S_trial_mask):
                        added.append(x)
                        S_trial_set.add(x)
                        S_trial_mask |= (1 << (x - 1))
                
                if len(added) >= 2:
                    S = S_trial + added
                    S_set = S_trial_set
                    S_mask = S_trial_mask
                    improved = True
                    break
        return S

    # Phase 1: Start with divisors of highly composite numbers
    div_counts = []
    for n in range(1, N + 1):
        d = len(divisors_of(n))
        div_counts.append((d, n))
    div_counts.sort(reverse=True)

    best_clique = []
    best_size = 0

    for d, n in div_counts[:20]:
        if time_left() < 20:
            break
        base = divisors_of(n)
        S = local_add(base)
        S = local_swap(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 2: Greedy by degree ordering
    if time_left() > 15:
        degrees = [(popcount(adj[i]), i) for i in range(1, N + 1)]
        degrees.sort(reverse=True)
        order = [i for _, i in degrees]
        
        S = greedy_by_order(order)
        S = local_add(S)
        S = local_swap(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 3: Max-degree greedy
    if time_left() > 15:
        S = greedy_max_degree()
        S = local_add(S)
        S = local_swap(S)
        if len(S) > best_size:
            best_size = len(S)
            best_clique = S[:]

    # Phase 4: Random restarts
    if time_left() > 20:
        random.seed(42)
        for restart in range(40):
            if time_left() < 10:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = greedy_by_order(order)
            S = local_add(S)
            S = local_swap(S)
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S[:]

    # Phase 5: Aggressive final polish on best clique
    if time_left() > 15 and best_clique:
        S = list(best_clique)
        S_set = set(S)
        S_mask = set_to_mask(S)
        
        # Multiple passes of add + swap
        for _ in range(3):
            if time_left() < 10:
                break
            S = local_add(S)
            S_set = set(S)
            S_mask = set_to_mask(S)
            S = local_swap(S)
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S[:]

    # Final validation
    result = sorted(set(best_clique)) if best_clique else divisors_of(480)
    
    valid = []
    valid_mask = 0
    for a in result:
        if compatible_with_all(a, valid_mask):
            valid.append(a)
            valid_mask |= (1 << (a - 1))
    
    if len(valid) < 2:
        valid = divisors_of(480)
    
    return valid
