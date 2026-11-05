import time
import math
from itertools import permutations

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses a TSP heuristic on the cycle structure of permutations.
    """
    start_time = time.time()
    deadline = 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312134213241321431214231412312"

    # For n >= 5, use the heuristic
    symbols = list(range(1, n + 1))
    all_perms = list(permutations(symbols))
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    
    # Identify cycles (based on left shift)
    # shift(p) = (p[1], p[2], ..., p[n-1], p[0])
    visited = set()
    cycles = []
    
    for p in all_perms:
        if p in visited:
            continue
        cycle = []
        curr = p
        while curr not in visited:
            visited.add(curr)
            cycle.append(curr)
            curr = (curr[1:] + curr[:1])
        cycles.append(cycle)
        
    K = len(cycles)
    
    # Precompute overlap costs between cycles
    # W[i][j] = min cost to go from any node in cycle i to any node in cycle j
    # Cost = n - overlap
    
    def get_overlap(p, q):
        # Max k such that p[n-k:] == q[:k]
        max_k = min(n, n)
        for k in range(max_k, -1, -1):
            if k == 0:
                return 0
            if p[n-k:] == q[:k]:
                return k
        return 0

    # We only need min cost.
    # W[i][j] is min over u in C_i, v in C_j of (n - overlap(u, v))
    
    # To speed up, we can just compute this matrix.
    # K is 720 for n=7. 720x720 matrix.
    # 720*720*7*7 operations approx 25 million. Fast enough.
    
    W = [[0] * K for _ in range(K)]
    
    for i in range(K):
        for j in range(K):
            if i == j:
                W[i][j] = 1 # Intra-cycle cost is 1 (shift)
                continue
            min_cost = n
            for u in cycles[i]:
                for v in cycles[j]:
                    ov = get_overlap(u, v)
                    cost = n - ov
                    if cost < min_cost:
                        min_cost = cost
                        if min_cost == 2: # Can't get lower than 2 for distinct cycles
                            break
                if min_cost == 2:
                    break
            W[i][j] = min_cost
            
    if time.time() > deadline:
        # Fallback to classic if too slow (unlikely for n=7)
        return _classic_construct(n)

    # Solve TSP on K nodes with weights W
    # Use Nearest Neighbor + 2-Opt
    
    # Start at 0
    tour = [0]
    unvisited = set(range(1, K))
    
    while unvisited:
        curr = tour[-1]
        # Find nearest neighbor
        best_node = None
        best_dist = float('inf')
        for node in unvisited:
            if W[curr][node] < best_dist:
                best_dist = W[curr][node]
                best_node = node
        tour.append(best_node)
        unvisited.remove(best_node)
        
    # 2-Opt optimization
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(K - 1):
            for j in range(i + 1, K):
                # Reverse segment i+1 to j
                # Check if it improves
                # Tour: ... A (i) B (i+1) ... C (j) D (j+1) ...
                # New: ... A D ... C B ...
                # If i+1 == j, no change
                if j == i + 1:
                    continue
                
                node_a = tour[i]
                node_b = tour[i + 1]
                node_c = tour[j]
                node_d = tour[(j + 1) % K]
                
                old_cost = W[node_a][node_b] + W[node_c][node_d]
                new_cost = W[node_a][node_d] + W[node_c][node_b]
                
                if new_cost < old_cost:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
                    if time.time() > deadline - 5:
                        return _classic_construct(n)
                        
    # Reconstruct the superpermutation
    # For each cycle in the tour, we need to pick entry/exit nodes.
    # We established: Enter u, exit prev(u).
    # We need to minimize sum D(prev(u_i), u_{i+1}).
    # This is a chain dependency.
    # We can greedily pick u_i given u_{i+1}?
    # Or just pick u_i to minimize D(prev(u_i), u_{i+1}) for the best u_{i+1}?
    # Actually, we can just iterate.
    
    word = []
    
    # Pick starting node for first cycle
    # To minimize first jump, we can just pick any, or optimize later.
    # Let's just pick the first node in the cycle.
    
    current_perm = None
    
    for idx, c_idx in enumerate(tour):
        cycle = cycles[c_idx]
        
        if idx == 0:
            # Start with first node of cycle
            u = cycle[0]
            word.extend(u)
            current_perm = u
            # We must visit all nodes in this cycle.
            # We enter at u. We must exit at prev(u).
            # So we traverse the cycle starting at u, ending at prev(u).
            # This means we append n-1 symbols.
            # The sequence of perms is u, shift(u), ..., prev(u).
            for k in range(1, n):
                current_perm = (current_perm[1:] + current_perm[:1])
                word.append(current_perm[-1])
            # Now current_perm is prev(u).
        else:
            # We are at current_perm (which is exit of previous cycle).
            # We want to jump to an entry node u in this cycle.
            # We want to minimize cost D(current_perm, u).
            # Note: The cost to jump is n - overlap(current_perm, u).
            # We pick u that minimizes this.
            best_u = None
            best_cost = float('inf')
            for u in cycle:
                ov = get_overlap(current_perm, u)
                cost = n - ov
                if cost < best_cost:
                    best_cost = cost
                    best_u = u
            
            u = best_u
            
            # Append symbols to reach u
            # current_perm is P. u is Q.
            # We append Q[n-overlap:]
            ov = get_overlap(current_perm, u)
            to_append = u[n - ov:]
            word.extend(to_append)
            current_perm = u
            
            # Now traverse the cycle from u to prev(u)
            for k in range(1, n):
                current_perm = (current_perm[1:] + current_perm[:1])
                word.append(current_perm[-1])
                
    return "".join(map(str, word))

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
    return "".join(map(str, word))
