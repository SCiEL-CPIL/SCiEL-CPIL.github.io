import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using a graph-based branch-and-bound algorithm
    with dynamic node selection, Moon-Moser clique cover bound, and
    structural pruning.
    """
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Build the graph: nodes are 1..511, edges connect non-coprime pairs
    # We want maximum weight independent set = maximum weight set of pairwise coprime numbers
    
    # Sort nodes by score descending for better pruning
    nodes = list(range(1, N))
    nodes.sort(key=lambda x: -scores[x])
    n = len(nodes)
    
    # Map original value to sorted index
    val_to_idx = {nodes[i]: i for i in range(n)}
    
    # Build adjacency: adj[i] is a bitmask of nodes that conflict with node i
    # Two nodes conflict if they share a prime factor (not coprime)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (factor_masks[nodes[i]] & factor_masks[nodes[j]]) != 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Precompute degree (number of conflicts)
    degree = [adj[i].bit_count() for i in range(n)]
    
    # Moon-Moser clique cover bound:
    # Partition nodes into cliques (groups where every pair conflicts).
    # For an independent set, we can pick at most one node from each clique.
    # Upper bound = sum of max weight in each clique.
    # 
    # A natural clique cover: group by smallest prime factor (SPF).
    # All numbers with the same SPF form a clique (they all share that prime).
    # Numbers that are prime themselves form singleton cliques.
    # Number 1 has no prime factors - it's coprime to everything, so it's in its own group.
    
    # Build SPF-based groups
    spf_groups = {}  # spf -> list of sorted indices
    for i in range(n):
        val = nodes[i]
        if val == 1:
            spf = -1  # special group for 1
        else:
            # Find smallest prime factor
            temp = val
            spf = temp
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    spf = p
                    break
            if spf == temp and temp > 1:
                spf = temp  # prime itself
        if spf not in spf_groups:
            spf_groups[spf] = []
        spf_groups[spf].append(i)
    
    # Precompute max weight for each spf group
    spf_group_max = {}
    for spf, group in spf_groups.items():
        max_w = max(scores[nodes[i]] for i in group)
        spf_group_max[spf] = max_w
    
    # Better bound: use the actual available set
    # For a given set of available nodes, the bound is:
    # sum over each spf group of max weight among available nodes in that group
    
    # Precompute for each node its spf group
    node_spf = [0] * n
    for i in range(n):
        val = nodes[i]
        if val == 1:
            node_spf[i] = -1
        else:
            temp = val
            spf = temp
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    spf = p
                    break
            if spf == temp and temp > 1:
                spf = temp
            node_spf[i] = spf
    
    # Best solution tracking
    best_weight = 0.0
    best_solution = []
    
    # Greedy initialization
    def greedy_init():
        chosen = []
        chosen_mask = 0
        chosen_idx_set = set()
        for i in range(n):
            if (adj[i] & chosen_mask) == 0:
                chosen.append(nodes[i])
                chosen_idx_set.add(i)
                # Build mask of chosen indices for conflict checking
                for j in range(n):
                    if (adj[i] >> j) & 1:
                        chosen_mask |= (1 << j)
        return chosen, sum(scores[v] for v in chosen)
    
    greedy_sol, greedy_weight = greedy_init()
    if greedy_weight > best_weight:
        best_weight = greedy_weight
        best_solution = greedy_sol[:]
    
    # Branch and bound with dynamic node selection
    # State: available set (bitmask), current weight, current solution
    # Use iterative approach with stack to avoid recursion depth issues
    
    # For the bound, we need to efficiently compute the sum of max weights per spf group
    # among available nodes
    
    # Precompute: for each spf group, sort nodes by weight descending
    # This allows quick lookup of max available weight in a group
    
    # Actually, let's use a different approach for the bound:
    # The Moon-Moser bound using clique cover by spf groups.
    # We precompute for each group the nodes sorted by weight.
    
    # For efficiency, let's use a simpler but effective bound:
    # Upper bound = sum of weights of all available nodes / 1 (trivial, too loose)
    # Better: use the spf group structure
    
    # Let's implement a recursive B&B with careful pruning
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute for each spf group, the list of (weight, index) sorted by weight desc
    spf_group_sorted = {}
    for spf, group in spf_groups.items():
        sorted_group = sorted(group, key=lambda i: -scores[nodes[i]])
        spf_group_sorted[spf] = sorted_group
    
    def compute_bound(available_mask):
        """Compute upper bound using spf clique cover."""
        bound = 0.0
        for spf, sorted_group in spf_group_sorted.items():
            max_w = 0.0
            for i in sorted_group:
                if (available_mask >> i) & 1:
                    max_w = scores[nodes[i]]
                    break
            bound += max_w
        return bound
    
    # For faster bound computation, precompute group membership
    # And use incremental updates
    
    # Alternative: use a tighter bound based on the actual graph structure
    # Greedy coloring bound (Tomita's algorithm for maximum clique)
    
    # Let's implement Tomita-style B&B for maximum weight independent set
    # This uses a greedy coloring of the complement graph (which is the original graph)
    # to get an upper bound on the maximum independent set size/weight
    
    # Actually, for weighted case, we can use:
    # 1. Sort available nodes by weight
    # 2. Compute a greedy coloring of the conflict graph among available nodes
    # 3. The bound is the sum of the maximum weight in each color class
    #    (since each color class is an independent set in the conflict graph,
    #     i.e., a clique in the original graph... wait, no)
    
    # Let me reconsider. For maximum weight independent set:
    # - We have a graph G where edges represent conflicts
    # - We want to find a maximum weight set of vertices with no edges between them
    # - This is the maximum weight independent set problem
    # 
    # Tomita's algorithm is for maximum clique. The complement of our problem:
    # - Independent set in G = Clique in complement of G
    # - So we can use maximum clique algorithms on the complement graph
    #
    # But the complement graph has ~511*(510)/2 - edges_of_G edges, which could be large.
    #
    # Alternative approach: Use the spf group bound which is simpler and effective.
    
    # Let's implement a clean B&B:
    
    best_weight_val = best_weight
    best_solution_list = best_solution[:]
    
    # Track best solution indices
    best_solution_indices = [val_to_idx[v] for v in best_solution]
    
    def branch_and_bound(available_mask, current_weight, chosen_indices):
        nonlocal best_weight_val, best_solution_list, best_solution_indices
        
        if time.time() > deadline:
            return
        
        # Update best if current solution is better
        if current_weight > best_weight_val + 1e-12:
            best_weight_val = current_weight
            best_solution_list = [nodes[i] for i in chosen_indices]
            best_solution_indices = chosen_indices[:]
        
        if available_mask == 0:
            return
        
        # Compute upper bound
        ub = compute_bound(available_mask)
        if current_weight + ub <= best_weight_val + 1e-12:
            return
        
        # Select next node to branch on
        # Strategy: pick the node with highest weight among available
        # This tends to find good solutions early and prune more
        
        # Find available nodes
        avail_nodes = []
        m = available_mask
        i = 0
        while m:
            if m & 1:
                avail_nodes.append(i)
            m >>= 1
            i += 1
        
        if not avail_nodes:
            return
        
        # Sort by weight descending
        avail_nodes.sort(key=lambda i: -scores[nodes[i]])
        
        # Branch on the highest weight node
        v = avail_nodes[0]
        
        # Branch 1: Include v
        new_available = available_mask & ~(1 << v)
        # Remove all neighbors of v
        new_available &= ~adj[v]
        
        branch_and_bound(new_available, current_weight + scores[nodes[v]], 
                        chosen_indices + [v])
        
        # Branch 2: Exclude v
        new_available = available_mask & ~(1 << v)
        branch_and_bound(new_available, current_weight, chosen_indices)
    
    # Initial available mask: all nodes
    initial_mask = (1 << n) - 1
    branch_and_bound(initial_mask, 0.0, [])
    
    # Final local search to polish
    if time.time() < deadline - 5:
        # Try to improve by swapping
        chosen_set = set(best_solution_list)
        chosen_mask = 0
        for v in best_solution_list:
            chosen_mask |= factor_masks[v]
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for v in list(best_solution_list):
                # Try removing v and adding a better node
                new_mask = 0
                for u in best_solution_list:
                    if u != v:
                        new_mask |= factor_masks[u]
                
                best_b = -1
                best_gain = 0.0
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if (factor_masks[b] & new_mask) == 0:
                        gain = scores[b] - scores[v]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                
                if best_b != -1 and best_gain > 1e-12:
                    best_solution_list.remove(v)
                    best_solution_list.append(best_b)
                    chosen_set.discard(v)
                    chosen_set.add(best_b)
                    improved = True
        
        # Update best
        final_weight = sum(scores[v] for v in best_solution_list)
        if final_weight > best_weight_val:
            best_weight_val = final_weight
    
    return best_solution_list
