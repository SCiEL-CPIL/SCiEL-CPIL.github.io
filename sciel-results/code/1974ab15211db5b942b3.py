"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0

    # Moduli range 2..128
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # 1. Precompute GCD table
    gcd_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # 2. Construct the Conflict Graph
    # Nodes represent (modulus_idx, residue)
    # Total nodes = sum(m for m in moduli) ~ 8000
    # We build adjacency bitsets for fast intersection checks.
    # Node ID = offset[i] + r
    
    offsets = [0] * n
    total_nodes = 0
    for i in range(n):
        offsets[i] = total_nodes
        total_nodes += moduli[i]
        
    # adj[i] is a list of bitsets (integers) representing neighbors of node i
    # To save memory and time, we only store adjacencies for i < j initially? 
    # No, for MIS we need random access.
    # 8000 nodes * 8000 bits = 8M bits = 1MB. Very manageable.
    adj = [0] * total_nodes
    
    # We also need to know which modulus a node belongs to
    node_mod_idx = [0] * total_nodes
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        for r in range(m):
            node_mod_idx[start + r] = i

    # Build adjacency
    # Two nodes (i, r1) and (j, r2) are connected (conflict) if:
    # 1. i == j (same modulus, different residues) -> Always conflict
    # 2. i != j and (r1 - r2) % gcd(m_i, m_j) == 0 -> Conflict
    
    # Optimization: Group by modulus.
    # For each modulus i, all residues form a clique.
    # For each pair (i, j), we determine which residues conflict.
    
    # Initialize adj with self-cliques (same modulus)
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        # All nodes in [start, start+m) are connected to each other
        # We can represent this later or just add edges.
        # Adding 8000*8000 edges is too slow if done naively.
        # Let's build adj using bit operations.
        
    # Efficient Adjacency Construction
    # For each modulus i, the nodes are a clique.
    # For each pair (i, j) with i < j:
    # g = gcd(m_i, m_j)
    # r1 in [0, m_i), r2 in [0, m_j)
    # Conflict if (r1 - r2) % g == 0
    # This means r1 % g == r2 % g.
    
    # We can build the adj list for each node.
    # Since total nodes is ~8000, we can use Python integers as bitsets.
    
    # Step 2a: Add intra-modulus cliques
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        # Create a mask for this modulus
        mask = (1 << m) - 1
        # Shift to position
        # Actually, it's easier to just iterate pairs for inter-modulus and 
        # handle intra-modulus separately or via a helper.
        
    # Let's restart the adjacency construction with a more efficient method.
    # We will store for each node u, a bitset of its neighbors.
    
    # Intra-modulus: All residues of m_i are neighbors.
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        # The bitset for these nodes relative to their start is (1<<m)-1
        # But we need global bitset.
        # Let's just add edges.
        # Since m is small (<=128), intra-modulus edges are few.
        for r1 in range(m):
            u = start + r1
            for r2 in range(r1+1, m):
                v = start + r2
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Inter-modulus:
    for i in range(n):
        mi = moduli[i]
        start_i = offsets[i]
        for j in range(i+1, n):
            mj = moduli[j]
            start_j = offsets[j]
            g = gcd_table[i][j]
            
            # Group residues of i by r % g
            # Group residues of j by r % g
            # If r_i % g == r_j % g, they conflict.
            
            # To avoid O(mi * mj) loop for every pair (which is ~8000*8000/2 * avg cost),
            # we can optimize.
            # Sum of mi*mj over all pairs is roughly (sum mi)^2 / 2 ~ 32M.
            # 32M operations in Python is ~3-5 seconds. This is acceptable.
            
            # We can precompute groups for each modulus i once?
            # No, g depends on j.
            
            # Let's just do the double loop but optimized.
            # For each r_i in 0..mi-1:
            #   target_residue = r_i % g
            #   For each r_j in 0..mj-1 where r_j % g == target_residue:
            #       add edge
            
            # Precompute residues of j grouped by mod g?
            # Doing this for every pair (i,j) is expensive.
            # Alternative: Iterate r_i, then iterate r_j step by g?
            # No, r_j % g == r_i % g.
            
            # Let's stick to the simple double loop but break early if possible?
            # Or use numpy?
            # Let's try the simple loop first, it might be fast enough.
            
            for r_i in range(mi):
                u = start_i + r_i
                target = r_i % g
                # Find all r_j such that r_j % g == target
                # r_j = target, target+g, target+2g, ... < mj
                first = target
                if first >= mj: continue
                for r_j in range(first, mj, g):
                    v = start_j + r_j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    # 3. Maximum Independent Set (MIS) Solver
    # We want to select a subset of nodes such that no two are adjacent.
    # And we must select at most one node per modulus (which is guaranteed by intra-modulus cliques).
    # So any Independent Set is a valid solution.
    # We want to maximize the size of the Independent Set.
    
    # Algorithm: Branch and Bound.
    # We maintain a set of candidates.
    # We order nodes by degree (descending) to prune better.
    
    # Degrees
    degrees = [0] * total_nodes
    for u in range(total_nodes):
        # Count bits
        degrees[u] = bin(adj[u]).count('1')
        
    # Order nodes by degree
    ordered_nodes = sorted(range(total_nodes), key=lambda x: -degrees[x])
    
    # Map original index to position in ordered list? 
    # Actually, we can just work with the original indices but pick the next node 
    # from the candidate set based on degree.
    
    # To make it faster, we can use a recursive solver that takes a bitset of candidates.
    # However, Python integers can be large.
    # Total nodes ~8000. Integer size 1KB. Bit operations are fast in C.
    
    # Upper Bound: 
    # A simple upper bound is the number of candidates remaining.
    # A better bound is the size of a maximum clique in the complement graph (which is the independence number of the subgraph).
    # Or simply: Greedy coloring of the subgraph induced by candidates. 
    # The chromatic number is an upper bound on the independence number? No, Independence Number <= Chromatic Number is FALSE. 
    # Independence Number * Chromatic Number >= N. So Independence Number <= N / Min_Chromatic? No.
    # Actually, a valid upper bound for MIS is the size of a Maximum Clique in the *complement* graph? No, that's MIS itself.
    # Standard bound: Greedy coloring of the *candidate graph*. The number of colors used is an upper bound on the size of a Maximum Independent Set?
    # Wait. In a graph G, the size of a Maximum Independent Set (alpha(G)) is related to the Chromatic Number (chi(G)).
    # alpha(G) * chi(G) >= |V|. So alpha(G) >= |V|/chi(G). This is a lower bound.
    # We need an upper bound.
    # The size of a Maximum Independent Set is at most the number of vertices.
    # A tighter upper bound: Greedy coloring gives a partition into cliques? No, coloring partitions into independent sets.
    # If we color the graph with k colors, we can pick at most 1 vertex from each color class? No, a color class is an independent set.
    # So we can pick ALL vertices from a color class.
    # This logic is reversed.
    
    # Let's use a simpler bound: 
    # The maximum number of vertices we can pick is bounded by the number of moduli remaining?
    # Yes! Since we can pick at most 1 node per modulus, the upper bound is simply the number of distinct moduli present in the candidate set.
    # This is a very strong bound because the "clique" constraint (one per modulus) is the dominant one.
    
    # Let's verify:
    # If we have candidates C, the max independent set size is <= number of moduli represented in C.
    # This is correct because any independent set contains at most one node from each modulus clique.
    
    # So, the algorithm:
    # 1. Start with all nodes as candidates.
    # 2. Pick a node u from candidates (e.g., highest degree).
    # 3. Branch 1: Include u. Remove u and all neighbors of u from candidates.
    # 4. Branch 2: Exclude u. Remove u from candidates.
    # 5. Upper bound: Count distinct moduli in candidates.
    
    # To make this efficient, we need to quickly count distinct moduli in a bitset.
    # We can precompute for each modulus i, a bitset mask_i containing all nodes of modulus i.
    # Then, for a candidate bitset C, the number of moduli is the count of i such that (C & mask_i) != 0.
    # There are 127 moduli. Checking 127 bitsets is fast.
    
    mod_masks = [0] * n
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        # Create mask
        mask = 0
        for r in range(m):
            mask |= (1 << (start + r))
        mod_masks[i] = mask
        
    # Precompute list of (mod_idx, mask) for fast iteration
    mod_info = [(i, mod_masks[i]) for i in range(n)]
    
    def count_moduli(candidates):
        count = 0
        for i, mask in mod_info:
            if candidates & mask:
                count += 1
        return count
        
    # State
    best_size = 0
    best_set = 0 # Bitset of the solution
    
    # To speed up, we can use a recursive function with memoization? 
    # State space is huge. No memoization.
    # We need good pruning.
    
    # We will use a stack-based or recursive search.
    # Recursion depth up to 127 (max moduli). Python default is 1000, so OK.
    
    # Optimization: 
    # When we include a node u, we remove N(u).
    # When we exclude u, we just remove u.
    # We should always pick the node that leads to the best pruning.
    # Heuristic: Pick the node with the highest degree in the current subgraph?
    # Or the node from the modulus that has the most remaining candidates?
    
    # Let's try: Pick the node u in C that maximizes the number of neighbors in C.
    # This is expensive to compute each time.
    # Alternative: Pick the node from the modulus that has the "most constrained" neighbors?
    
    # Let's stick to a simple heuristic: Pick the first node in C (in some order) or the one with highest global degree.
    # Global degree is static.
    
    # To make it faster, we can pre-sort nodes by degree.
    # In the candidate set C, we pick the node with the highest global degree.
    
    # Implementation of finding the highest degree node in C:
    # Iterate through ordered_nodes, check if in C.
    # This is O(N) per step. N=8000. Depth=127. 1M ops. OK.
    
    def find_max_degree_node(candidates):
        for u in ordered_nodes:
            if candidates & (1 << u):
                return u
        return -1
        
    def solve(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        # Pruning
        if current_size + count_moduli(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Pick a node
        u = find_max_degree_node(candidates)
        if u == -1:
            return
            
        # Branch 1: Include u
        # New candidates = candidates & ~(adj[u] | (1 << u))
        # Note: adj[u] contains neighbors. We must remove u and neighbors.
        new_candidates_inc = candidates & ~(adj[u] | (1 << u))
        solve(new_candidates_inc, current_set | (1 << u), current_size + 1)
        
        # Branch 2: Exclude u
        new_candidates_exc = candidates & ~(1 << u)
        solve(new_candidates_exc, current_set, current_size)
        
    # Initial call
    all_nodes = (1 << total_nodes) - 1
    solve(all_nodes, 0, 0)
    
    # Extract solution from best_set
    solution = []
    for i in range(n):
        m = moduli[i]
        start = offsets[i]
        for r in range(m):
            if best_set & (1 << (start + r)):
                solution.append([r, m])
                break # Only one per modulus
        
    if not solution:
        return [[1, 2]]
        
    return solution

# EVOLVE-BLOCK-END
