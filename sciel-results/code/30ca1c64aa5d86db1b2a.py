import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    # Identify valid nodes (no self-conflict: 2*v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)

    n = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}

    # Build conflict bitmask for each node
    # conflicts_mask[i] is a bitmask where bit j is set if node i conflicts with node j
    conflicts_mask = [0] * n
    for i in range(n):
        vi = valid_nodes[i]
        for j in range(i + 1, n):
            vj = valid_nodes[j]
            if vi + vj in squares:
                conflicts_mask[i] |= (1 << j)
                conflicts_mask[j] |= (1 << i)

    # Full mask of all nodes
    full_mask = (1 << n) - 1

    # Precompute degree of each node (number of conflicts)
    degrees = [bin(conflicts_mask[i]).count('1') for i in range(n)]

    # Branch and bound for maximum independent set
    # We use a recursive approach with bitmasks for efficiency
    # State: (available_mask, current_size)
    # We want to maximize current_size + |MIS(available_mask)|

    best_size = 0
    best_mask = 0

    # Upper bound: for a set of available nodes, the maximum independent set size
    # is at most the number of available nodes
    # Better upper bound: greedy coloring bound (like in maximum clique algorithms)
    # For independent set, we can use: |MIS(S)| <= |S| - (minimum number of edges needed to cover S)
    # Simpler: just use |S| as upper bound, or a greedy bound

    # Let's use a simpler but effective approach:
    # Order nodes by degree (highest first) for better pruning
    # Actually, for MIS, we want to pick low-degree nodes first in greedy,
    # but for branch and bound, we branch on high-degree nodes first

    # Convert to list of (degree, node_index) sorted by degree descending
    node_order = sorted(range(n), key=lambda i: degrees[i], reverse=True)

    # For the B&B, we'll work with the original indexing but use the order for branching
    # Actually, let me just do a clean B&B with bitmasks

    def mis_size_upper_bound(available_mask):
        """Upper bound on MIS size for the available set.
        Use greedy: repeatedly pick a node, remove it and its neighbors.
        This gives a lower bound on the number of "independent" picks needed,
        which is an upper bound on... wait, no.
        
        Actually, a simple upper bound is just popcount(available_mask).
        A better one: the size of a maximal independent set found greedily is a lower bound on MIS,
        but we need an upper bound for pruning.
        
        Upper bound: |S| (trivial)
        Or: For each node v in S, MIS(S) <= MIS(S \ {v} \ N(v)) + 1
        This is the recurrence itself.
        
        Let's just use popcount as the upper bound for simplicity, 
        or a greedy coloring-like bound.
        
        Actually, a good upper bound for MIS: 
        The minimum vertex cover size + ... no.
        
        Let's use: upper bound = number of nodes in available_mask.
        This is loose but simple. We can improve with a greedy bound later.
        """
        return bin(available_mask).count('1')

    # Better upper bound using a greedy heuristic:
    # Greedily build an independent set from available_mask.
    # The size of this greedy set is a LOWER bound on MIS, not upper.
    # For upper bound, we can use: 
    #   MIS(S) <= |S| - floor(E(S)/d_max) ... not standard.
    
    # Let's use a different approach: 
    # Upper bound = size of maximum independent set in a relaxed problem.
    # Simplest effective bound: just use popcount.

    # Actually, let me implement a proper branch and bound with a good upper bound.
    # Upper bound: greedy coloring bound adapted for independent set.
    # For maximum clique, we color the graph and use the max color as upper bound.
    # For maximum independent set, we can color the COMPLEMENT graph.
    # But that's complex. Let's just use popcount for now and see if it's fast enough.

    # Let me try a different strategy: since n is ~200+, and the graph is sparse,
    # a well-optimized B&B should work. Let's implement it properly.

    # Recursive B&B with memoization won't work well due to large state space.
    # Instead, use a clean recursive B&B with good pruning.

    # To make it faster, let's use an iterative deepening or just a clean recursion.
    # Given the time limit, let's try a clean B&B with popcount upper bound.

    # Actually, for n~200, even with popcount bound, B&B might be too slow in Python.
    # Let's use a hybrid: 
    # 1. Get a good initial solution with greedy
    # 2. Use B&B with the initial solution size as lower bound for pruning
    # 3. Use a better upper bound

    # Step 1: Greedy to get initial solution
    def greedy_mis(available_mask):
        """Greedily find an independent set."""
        S = 0
        avail = available_mask
        while avail:
            # Pick node with minimum degree in avail
            best_node = -1
            best_deg = float('inf')
            temp = avail
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                # Count neighbors in avail
                nb_in_avail = conflicts_mask[idx] & avail
                deg = bin(nb_in_avail).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_node = idx
                    if deg == 0:
                        break
                temp ^= lb
            if best_node == -1:
                break
            S |= (1 << best_node)
            # Remove best_node and its neighbors
            avail &= ~(1 << best_node)
            avail &= ~conflicts_mask[best_node]
        return S

    initial_solution = greedy_mis(full_mask)
    initial_size = bin(initial_solution).count('1')
    best_size = initial_size
    best_mask = initial_solution

    # Step 2: B&B with pruning
    # We branch on nodes in a specific order.
    # For each node, we either include it or exclude it.
    # If we include it, we remove it and its neighbors.
    # If we exclude it, we just remove it.
    
    # To make this efficient, we process nodes in order of decreasing degree.
    # We use a recursive function with the available mask and current set.

    # For upper bound, let's use a greedy bound:
    # The maximum independent set size in available_mask is at most
    # the size of a maximum matching in the conflict graph restricted to available_mask? No.
    
    # Simple but effective: upper bound = popcount(available_mask)
    # With the initial solution giving a good lower bound, this should prune well.

    # Let's also add: if current_size + upper_bound <= best_size, prune.

    # To avoid deep recursion, we can use an iterative approach or limit depth.
    # With n~200, recursion depth could be 200, which is fine in Python.

    # But 2^200 is way too many states. We need good pruning.
    # The popcount bound is weak. Let's use a better one.

    # Better upper bound: 
    # For any set S, MIS(S) <= |S| - floor(|E(S)| / (d_max - 1)) ... not standard.
    
    # Let's use: MIS(S) <= greedy_max_independent_set_size(S) ... no, that's a lower bound.
    
    # Actually, for pruning, we need an UPPER bound. 
    # The trivial upper bound is |S|. 
    # A better one: if we can find a vertex cover of size k in S, then MIS(S) <= |S| - k.
    # Finding minimum vertex cover is hard, but we can find a maximal matching as a lower bound on vertex cover.
    # So: MIS(S) <= |S| - |max_matching(S)|
    # But finding max matching is also non-trivial.
    
    # Let's just use |S| as upper bound and hope the lower bound from initial solution is good enough.
    # With initial_size ~ 85, and |S| starting at ~200, we need to prune when current_size + |S| <= 85,
    # i.e., |S| <= 85 - current_size. This means we can only explore states where |S| is small.
    # This should be very effective!

    # Let's implement the B&B:
    def branch_and_bound(available_mask, current_size, current_mask):
        nonlocal best_size, best_mask

        if time.time() > deadline:
            return

        # Upper bound
        ub = bin(available_mask).count('1')
        if current_size + ub <= best_size:
            return

        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return

        # Pick a node to branch on
        # Choose node with highest degree in available_mask for better pruning
        best_node = -1
        best_deg = -1
        temp = available_mask
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            nb_in_avail = conflicts_mask[idx] & available_mask
            deg = bin(nb_in_avail).count('1')
            if deg > best_deg:
                best_deg = deg
                best_node = idx
            temp ^= lb

        if best_node == -1:
            return

        node_bit = 1 << best_node

        # Option 1: Include best_node
        # New available: remove best_node and its neighbors
        new_avail_include = available_mask & ~node_bit & ~conflicts_mask[best_node]
        branch_and_bound(new_avail_include, current_size + 1, current_mask | node_bit)

        # Option 2: Exclude best_node
        new_avail_exclude = available_mask & ~node_bit
        branch_and_bound(new_avail_exclude, current_size, current_mask)

    # Run B&B
    branch_and_bound(full_mask, 0, 0)

    # Convert best_mask back to list of values
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(valid_nodes[i])

    # Safety: verify the result
    result_set = set(result)
    for a in result:
        for b in result:
            if a <= b:
                if a + b in squares:
                    # This should not happen, but if it does, fall back to initial
                    return [v for v in valid_nodes if initial_solution & (1 << node_to_idx[v])]
    
    return result if result else [valid_nodes[0]]
