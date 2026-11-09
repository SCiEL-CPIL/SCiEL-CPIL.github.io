import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit (evaluator kills at 90s)
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute bad triples and edge weights
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    edge_weight = [[0.0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    conflict[a][b] |= (1 << c)
                    conflict[b][a] |= (1 << c)
                    conflict[a][c] |= (1 << b)
                    conflict[c][a] |= (1 << b)
                    conflict[b][c] |= (1 << a)
                    conflict[c][b] |= (1 << a)
                    # Weight edges by frequency of bad triples they participate in
                    edge_weight[a][b] += 1.0
                    edge_weight[b][a] += 1.0
                    edge_weight[a][c] += 1.0
                    edge_weight[c][a] += 1.0
                    edge_weight[b][c] += 1.0
                    edge_weight[c][b] += 1.0
    
    # Spectral relaxation: build weighted adjacency matrix and compute top eigenvector
    # using power iteration. The eigenvector gives a "score" for each element;
    # elements with higher scores are "safer" to include.
    
    # Build weighted adjacency as list of (neighbor, weight) for each node
    adj = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if edge_weight[a][b] > 0:
                adj[a].append((b, edge_weight[a][b]))
                adj[b].append((a, edge_weight[b][a]))
    
    # Power iteration for top eigenvector of the weighted adjacency matrix
    # We want the eigenvector corresponding to the most negative eigenvalue
    # (since we want to minimize conflicts), or equivalently, we can use
    # the top eigenvector of (max_weight * I - W) where W is the adjacency.
    
    # Actually, for maximum independent set in a graph, we want elements that
    # have few conflicts. Let's compute a "conflict score" for each element.
    
    # Compute degree-weighted conflict score
    conflict_score = [0.0] * (N + 1)
    for a in range(1, N + 1):
        for b, w in adj[a]:
            conflict_score[a] += w
    
    # Use power iteration on a transformed matrix to get a ranking
    # T = D - W where D is diagonal of degrees, W is weighted adjacency
    # The top eigenvector of T gives elements that are "well-separated"
    
    # Simpler approach: use the conflict_score to order elements,
    # then do greedy with multiple restarts using different tie-breaking
    
    # Also try: weighted greedy where we pick element that maximizes
    # (benefit - cost) where benefit=1 and cost=sum of edge weights to already-selected
    
    def weighted_greedy(order):
        S = []
        S_mask = 0
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            # Check if x is compatible with S
            compatible = True
            for s in S:
                if conflict[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S
    
    # Try several greedy orderings
    best_set = []
    best_size = 0
    
    # Natural order
    S = weighted_greedy(list(range(1, N + 1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Reverse order
    S = weighted_greedy(list(range(N, 0, -1)))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by conflict score (ascending - prefer low-conflict elements)
    order = sorted(range(1, N + 1), key=lambda x: conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by conflict score (descending)
    order = sorted(range(1, N + 1), key=lambda x: -conflict_score[x])
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Sort by number of prime factors
    def num_prime_factors(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order = sorted(range(1, N + 1), key=num_prime_factors)
    S = weighted_greedy(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Spectral ordering: use power iteration on (D - W) to get ranking
    # v_{k+1} = (D - W) v_k, normalize
    # Elements with high v values are "well-separated"
    
    # Initialize with random-ish vector
    import random
    random.seed(42)
    v = [0.0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = random.random()
    
    # Normalize
    norm = math.sqrt(sum(v[i]**2 for i in range(1, N + 1)))
    if norm > 0:
        for i in range(1, N + 1):
            v[i] /= norm
    
    # Power iteration
    for _ in range(100):
        if time.time() - start_time > deadline - 5:
            break
        new_v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            # (D - W) v at position i = D[i]*v[i] - sum_j W[i][j]*v[j]
            new_v[i] = conflict_score[i] * v[i]
            for j, w in adj[i]:
                new_v[i] -= w * v[j]
        
        # Normalize
        norm = math.sqrt(sum(new_v[i]**2 for i in range(1, N + 1)))
        if norm > 1e-10:
            for i in range(1, N + 1):
                new_v[i] /= norm
        v = new_v
    
    # Sort by spectral score (descending - prefer high spectral value)
    spectral_order = sorted(range(1, N + 1), key=lambda x: -v[x])
    S = weighted_greedy(spectral_order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Also try ascending spectral order
    spectral_order_asc = sorted(range(1, N + 1), key=lambda x: v[x])
    S = weighted_greedy(spectral_order_asc)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    # Branch and bound
    def solve(selected, available):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > deadline:
            return
        
        # Update best if we have a larger set
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
        
        if not available:
            return
        
        # Upper bound: can't exceed current best
        if len(selected) + available.bit_count() <= best_size:
            return
        
        # Find all available elements
        avail_list = []
        temp = available
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            avail_list.append(x)
            temp ^= low
        
        if not avail_list:
            return
        
        # Pick element with highest conflict degree in available
        # For each x, count weighted conflicts within available
        best_x = avail_list[0]
        best_conflict = -1
        for x in avail_list:
            c = 0.0
            for y, w in adj[x]:
                if (available >> y) & 1:
                    c += w
            if c > best_conflict:
                best_conflict = c
                best_x = x
        x = best_x
        
        # Branch 1: Include x
        new_selected = selected + [x]
        new_avail = available & ~(1 << x)
        for s in selected:
            new_avail &= ~conflict[s][x]
        
        solve(new_selected, new_avail)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(selected, available & ~(1 << x))
    
    # Start search
    solve([], full_mask)
    
    return best_set
