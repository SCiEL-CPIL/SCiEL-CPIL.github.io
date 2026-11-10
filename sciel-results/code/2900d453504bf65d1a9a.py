"""
Strategy: Exact Maximum Clique via Branch and Bound (Tomita's algorithm)
on the compatibility graph of residue classes.
Nodes: (residue, modulus) pairs for moduli 2..128.
Edges: Two nodes are connected if their residue classes are disjoint.
Goal: Find the maximum clique.
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    # 90s limit, use 80s to be safe
    deadline = start_time + 80.0
    MAX_MOD = 128

    # 1. Generate all possible nodes (residue, modulus)
    # Node index = (m - 2) * m + r  (since m ranges 2..128, r ranges 0..m-1)
    # Total nodes = sum_{m=2}^{128} m = 128*129/2 - 1 = 8256 - 1 = 8255
    nodes = []
    node_info = [] # (r, m)
    node_to_idx = {}
    
    idx = 0
    for m in range(2, MAX_MOD + 1):
        for r in range(m):
            node_info.append((r, m))
            node_to_idx[(r, m)] = idx
            idx += 1
            
    n_nodes = idx
    
    # 2. Build Adjacency Bitsets
    # adj[i] is a Python int representing a bitset of neighbors of node i
    # We also need to handle the constraint that moduli must be distinct.
    # This can be handled by NOT adding edges between nodes with the same modulus.
    
    adj = [0] * n_nodes
    
    # Optimization: Precompute gcd for moduli pairs
    moduli = list(range(2, MAX_MOD + 1))
    gcd_table = {}
    for i in range(len(moduli)):
        for j in range(i, len(moduli)):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[(moduli[i], moduli[j])] = g
            gcd_table[(moduli[j], moduli[i])] = g
            
    # Build graph
    # Two nodes (r1, m1) and (r2, m2) are compatible (disjoint classes) iff:
    # 1. m1 != m2 (distinct moduli constraint)
    # 2. (r1 - r2) % gcd(m1, m2) != 0 (disjointness condition)
    
    # To speed up, we can iterate over pairs.
    # 8255 nodes -> ~34 million pairs. This might be slow in Python loops.
    # Let's try to optimize.
    
    # Alternative: Iterate by modulus groups.
    # For each pair of moduli (m1, m2), determine which residues are compatible.
    
    # Let's store nodes grouped by modulus for easier access
    nodes_by_mod = {m: [] for m in range(2, MAX_MOD + 1)}
    for i, (r, m) in enumerate(node_info):
        nodes_by_mod[m].append(i)
        
    # Precompute compatibility between residue pairs for each modulus pair?
    # No, just build the bitsets directly.
    
    # To make it faster, we can use the fact that if m1 and m2 are fixed,
    # the compatibility depends only on r1 and r2 via gcd.
    
    # Let's iterate over all pairs of nodes.
    # 8255^2 / 2 = 34M iterations. In Python, this takes ~10-20s.
    # We can potentially prune or use numpy?
    # Let's try a slightly optimized pure python loop first, with time checks.
    
    # Actually, we can be smarter.
    # For a fixed m1, and a fixed m2, the condition (r1 - r2) % g != 0
    # means r2 is NOT congruent to r1 mod g.
    
    # Let's just do the O(N^2) construction but efficiently.
    
    # We'll use lists of ints for the bitsets.
    
    # Optimization:
    # For each node i (r1, m1), we want to find all nodes j (r2, m2) such that
    # m2 != m1 and (r1 - r2) % gcd(m1, m2) != 0.
    
    # Let's precompute for each modulus m, the list of indices.
    
    # We will iterate i from 0 to N-1.
    # For each i, we iterate over all m2 != m1.
    # For each m2, we iterate over all r2 in 0..m2-1.
    # This is still O(N^2) effectively.
    
    # Let's try to use numpy for the bitset construction if possible?
    # Or just accept the 10s overhead?
    # 8255 nodes.
    # Sum of m for m=2..128 is 8255.
    # Pairs: sum_{m1} sum_{m2 != m1} m1*m2.
    # This is (sum m)^2 - sum m^2.
    # (8255)^2 - sum_{m=2}^{128} m^2.
    # sum m^2 approx 128^3/3 = 699050.
    # 8255^2 = 68,145,025.
    # Total pairs ~ 67.4 million.
    # Python loop of 67M simple operations is ~5-10 seconds. Acceptable.
    
    # We will build adj[i] as an integer.
    
    # To speed up, we can precompute the "incompatible" residues for each (m1, m2, r1) pair?
    # No, let's just do the direct check.
    
    # Let's try to vectorize the inner loop using numpy?
    # We can't easily build bitsets with numpy in a single shot for variable lengths.
    
    # Let's stick to Python but optimize the inner loop.
    
    # Precompute gcd for all modulus pairs
    # We already have gcd_table.
    
    # Let's iterate over nodes.
    # For speed, we can precompute the bitset of all nodes for each modulus.
    # mask_mod[m] = bitset of all nodes with modulus m.
    
    mask_mod = [0] * (MAX_MOD + 1)
    for m in range(2, MAX_MOD + 1):
        mask = 0
        for idx in nodes_by_mod[m]:
            mask |= (1 << idx)
        mask_mod[m] = mask
        
    # For each node i (r1, m1):
    # Start with all nodes except those with modulus m1.
    # candidate_mask = all_nodes_mask ^ mask_mod[m1]
    # Then for each other modulus m2, we need to remove residues r2 such that (r1-r2)%g==0.
    
    all_nodes_mask = (1 << n_nodes) - 1
    
    for i in range(n_nodes):
        r1, m1 = node_info[i]
        # Start with all nodes
        adj_i = all_nodes_mask
        # Remove self and same modulus
        adj_i &= ~mask_mod[m1]
        
        # For each other modulus m2
        for m2 in range(2, MAX_MOD + 1):
            if m2 == m1:
                continue
            g = gcd_table[(m1, m2)]
            # We need to remove nodes (r2, m2) where r2 % g == r1 % g
            # Let's iterate through nodes of m2 and clear bits if incompatible
            # This is O(m2) per m2.
            # Total work per node i is sum_{m2 != m1} m2 = 8255 - m1.
            # Total work O(N^2).
            
            # Optimization: Precompute for each (m, g, r_mod_g) the bitset of residues?
            # Since g is determined by m1, m2, we can precompute masks for each m2 and each possible g?
            # No, g varies.
            
            # Let's just loop.
            # To make it faster, we can precompute for each m2, a list of (idx, r2)
            # and check (r1 - r2) % g != 0.
            
            # Actually, we can precompute for each modulus m2, and for each divisor g of m2 (or any g),
            # the bitsets of residues congruent to k mod g.
            # But g can be any divisor of gcd(m1, m2).
            
            # Let's just do the loop. It's 67M iterations.
            # In Python, 67M iterations of `if (r1 - r2) % g == 0: mask &= ~(1<<idx)`
            # might be too slow (10-20s).
            
            # Let's try to optimize the inner loop.
            # We can precompute for each m2, a dictionary:
            # residue_masks[m2][g][k] = bitset of nodes in m2 with residue r such that r % g == k
            # Then for node (r1, m1), and modulus m2, g = gcd(m1, m2), k = r1 % g.
            # incompatible_mask = residue_masks[m2][g][k]
            # adj_i &= ~incompatible_mask
            
            # This reduces the inner loop to O(1) bit operations!
            # Precomputation cost:
            # For each m2 (2..128), for each possible g (divisors of m2? or just 1..m2?),
            # for each k (0..g-1), build the mask.
            # Number of divisors is small. But g is gcd(m1, m2), so g divides m2.
            # So we only need g that are divisors of m2.
            
            # Let's precompute residue_masks.
            # residue_masks[m][g][k]
            
            # This precomputation is fast.
            # Then the main loop is O(N * 127) bit operations. Very fast.
            
            pass # Placeholder, we will implement precomputation below
            
    # --- Precomputation for Fast Bitset Construction ---
    
    # residue_masks[m][g] is a list of length g, where residue_masks[m][g][k] is the bitset of nodes in modulus m with residue r such that r % g == k.
    # We only need g that are divisors of m.
    # Actually, gcd(m1, m2) always divides m2. So for a fixed m2, g is a divisor of m2.
    
    residue_masks = {}
    for m in range(2, MAX_MOD + 1):
        residue_masks[m] = {}
        # Find divisors of m
        divisors = []
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                divisors.append(d)
                if d != m // d:
                    divisors.append(m // d)
        for g in divisors:
            masks = [0] * g
            for idx in nodes_by_mod[m]:
                r = node_info[idx][0]
                k = r % g
                masks[k] |= (1 << idx)
            residue_masks[m][g] = masks
            
    # --- Build Adjacency List ---
    
    for i in range(n_nodes):
        r1, m1 = node_info[i]
        adj_i = all_nodes_mask & ~mask_mod[m1] # Exclude same modulus
        
        for m2 in range(2, MAX_MOD + 1):
            if m2 == m1:
                continue
            g = gcd_table[(m1, m2)]
            k = r1 % g
            # Get the mask of incompatible residues in m2
            # Note: g must be a divisor of m2, which it is because g = gcd(m1, m2).
            # So residue_masks[m2][g] exists.
            incompatible_mask = residue_masks[m2][g][k]
            adj_i &= ~incompatible_mask
            
        adj[i] = adj_i
        
    # 3. Maximum Clique Algorithm (Tomita's Branch and Bound)
    # We want to find the max clique in the graph defined by adj.
    # Since we want distinct moduli, the graph already enforces that (no edges between same modulus).
    # A clique in this graph corresponds to a set of nodes (residue classes) that are pairwise disjoint.
    # We want to maximize the size of such a set.
    
    best_clique_size = 0
    best_clique_nodes = []
    
    # Order nodes by degree (descending) for better pruning?
    # Tomita's algorithm works well with a specific ordering.
    # Let's sort nodes by degree.
    
    degrees = [bin(adj[i]).count('1') for i in range(n_nodes)]
    sorted_nodes = sorted(range(n_nodes), key=lambda i: degrees[i], reverse=True)
    
    # Remap indices? Or just use the sorted order in the algorithm?
    # It's easier to remap to 0..N-1 in the new order.
    # Let's create a new adjacency list based on the sorted order.
    
    # map old_idx -> new_idx
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_nodes)}
    
    new_adj = [0] * n_nodes
    for i in range(n_nodes):
        old_i = sorted_nodes[i]
        mask = adj[old_i]
        new_mask = 0
        # Iterate through bits set in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            j_old = lsb.bit_length() - 1
            j_new = old_to_new[j_old]
            new_mask |= (1 << j_new)
            m ^= lsb
        new_adj[i] = new_mask
        
    # Now run Max Clique on new_adj
    
    # Tomita's Algorithm (MCQ)
    # R: current clique (list of nodes)
    # P: candidate set (bitset)
    # color_class: greedy coloring of P to provide upper bound
    
    def max_clique(R, P, color_class, depth):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline:
            return
            
        # If P is empty, update best
        if P == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_nodes = R[:]
            return
            
        # Greedy coloring of P
        # We need to assign colors to nodes in P such that adjacent nodes have different colors.
        # The color of a node is an upper bound on the clique size it can form.
        # We process nodes in some order (e.g., by degree in P, or just index order).
        
        # Standard MCQ uses a specific coloring procedure.
        # Let's implement a simple greedy coloring.
        # We want to find an upper bound.
        
        # Actually, Tomita's algorithm uses a specific way to compute the color class.
        # Let's use a simplified version:
        # 1. Order nodes in P.
        # 2. Assign colors greedily.
        # 3. Use the max color as the bound.
        
        # To be efficient, we should order P by degree in P.
        # But recalculating degrees is expensive.
        # Let's just use the index order for now, or precomputed global degree.
        
        # Let's try a standard greedy coloring on the subgraph induced by P.
        
        # Extract nodes in P
        nodes_in_P = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes_in_P.append(idx)
            m ^= lsb
            
        # Order by degree in P (approximate by global degree for speed, or exact?)
        # Let's use global degree for speed.
        # nodes_in_P.sort(key=lambda x: degrees[sorted_nodes[x]], reverse=True)
        
        colors = [0] * n_nodes # Reuse array? No, local is safer.
        color_class = [] # List of (color, bitset_of_nodes_with_this_color)
        
        # Greedy coloring
        # We want to assign the smallest possible color.
        # But for the bound, we want to maximize the number of colors used?
        # No, the max color assigned is the bound.
        # To get a tight bound, we should try to use fewer colors?
        # Actually, in MCQ, we assign colors such that nodes with higher color are processed first?
        
        # Let's use the standard MCQ coloring:
        # 1. Initialize P' = P.
        # 2. While P' not empty:
        #    a. Pick a node u from P' (e.g., first one).
        #    b. Assign current color to u.
        #    c. Remove u and its neighbors from P'.
        #    d. Increment color.
        
        # This gives a valid coloring. The number of colors is the bound.
        
        P_prime = P
        color = 0
        # We store the nodes assigned to each color to facilitate the branching step
        # Actually, we just need the max color.
        # But for the branching, we process nodes in reverse color order.
        
        # Let's store the nodes in order of color assignment.
        # Or just use the bitset P and the color values.
        
        # Simple approach:
        # Assign color to each node in P.
        # color_node[i] = color assigned to node i.
        # Then sort nodes in P by color_node[i] descending.
        # Process them in that order.
        
        color_node = {}
        P_curr = P
        current_color = 0
        
        while P_curr:
            # Pick a node. Let's pick the one with highest degree in P_curr?
            # Or just the lowest index?
            # Let's pick lowest index for determinism and speed.
            lsb = P_curr & -P_curr
            u = lsb.bit_length() - 1
            
            # Assign current_color to u
            color_node[u] = current_color
            
            # Remove u and its neighbors from P_curr
            # Neighbors of u in P_curr: adj[u] & P_curr
            # Note: adj[u] includes u? No, usually simple graph.
            # Our adj[u] does not include u.
            P_curr &= ~(adj[u] | (1 << u)) # Remove u and neighbors
            
            current_color += 1
            
        max_color = current_color
        
        # If max_color + len(R) <= best_clique_size, prune
        if len(R) + max_color <= best_clique_size:
            return
            
        # Branch: Process nodes in descending order of color
        # Get list of nodes in P sorted by color descending
        nodes_to_process = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes_to_process.append((color_node[idx], idx))
            m ^= lsb
            
        nodes_to_process.sort(reverse=True) # Sort by color desc, then idx desc
        
        for c, u in nodes_to_process:
            if time.time() > deadline:
                return
            # Prune based on color
            if len(R) + c <= best_clique_size:
                break # Since colors are non-increasing, we can stop
                
            # Branch on u
            R.append(u)
            
            # New P: neighbors of u in P
            new_P = P & adj[u]
            
            max_clique(R, new_P, None, depth + 1)
            
            R.pop()
            
            # Remove u from P for subsequent branches
            P &= ~(1 << u)
            
    # Initialize
    # Start with empty clique, P = all nodes
    # But we need to order the initial P.
    # Tomita's algorithm usually starts with P = all nodes, ordered by degree.
    # We already sorted nodes by degree in `sorted_nodes`.
    # So P = (1 << n_nodes) - 1.
    
    # We need to handle the coloring in the initial call.
    # My implementation of max_clique does the coloring internally.
    
    # One issue: Recursion depth. Max clique size is likely small (< 50?).
    # Python recursion limit is 1000. Should be fine.
    
    # Let's run it.
    # To avoid recursion limit issues, we might need to increase it or use iterative DFS.
    # But let's try recursion first.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Initial P is all nodes.
    # We need to ensure the coloring inside max_clique is efficient.
    # The coloring loop `while P_curr:` iterates max_color times.
    # Inside, it does bit operations.
    
    # Let's call it.
    # R = []
    # P = (1 << n_nodes) - 1
    
    # Wait, the coloring in max_clique is done from scratch every time?
    # Yes, in my implementation above. This is O(N^2) per node visited.
    # This might be too slow if the search tree is large.
    # However, for N=8255, the clique number is small.
    # The branching factor is reduced by the bound.
    
    # Let's try it.
    
    # Optimization: Pass the coloring down?
    # Standard MCQ precomputes the color class for P and passes it.
    # My implementation recomputes it.
    # Let's stick to this for now, it's simpler.
    
    # Actually, recomputing coloring is expensive.
    # Let's optimize the coloring step.
    # Instead of picking lowest index, we can pick nodes more strategically?
    # For now, let's just run it and see.
    
    # One potential issue: The graph is dense or sparse?
    # Disjointness condition: (r1-r2)%g != 0.
    # If g=1, always disjoint.
    # If g is large, less likely to be disjoint.
    # Most moduli have small gcds with others.
    # So the graph is likely dense.
    # In dense graphs, max clique is large?
    # But we have the distinct modulus constraint.
    # Max possible clique size is 127 (one per modulus).
    
    # Let's run the search.
    
    # To speed up, we can use a heuristic to order the nodes in P for coloring.
    # Greedy coloring is better if we process nodes with high degree first.
    
    # Let's modify the coloring part to use degree order.
    # But we don't have degrees in P readily available.
    # We can approximate with global degrees.
    
    # Let's just run the basic version.
    
    # Note: The `color_class` parameter in the signature is unused in my current impl.
    # I'll remove it or use it.
    
    # Let's refine the max_clique function to be more efficient.
    
    def max_clique_optimized(R, P):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_nodes = R[:]
            return
            
        # Coloring
        P_curr = P
        current_color = 0
        color_node = {}
        
        # To speed up coloring, we can use a list of nodes in P sorted by global degree.
        # Let's precompute this list once? No, P changes.
        # Let's just extract nodes and sort by global degree.
        
        nodes_in_P = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes_in_P.append(idx)
            m ^= lsb
            
        # Sort by global degree descending
        # degrees is global degree
        nodes_in_P.sort(key=lambda x: degrees[x], reverse=True)
        
        # Greedy Coloring
        # We assign colors to nodes in nodes_in_P order.
        # But we must ensure that if two nodes are adjacent, they have different colors.
        # Standard greedy:
        # for u in nodes_in_P:
        #    find smallest color c such that no neighbor of u in P has color c.
        #    assign c to u.
        
        # This is O(N * MaxColor).
        
        # Let's implement this.
        # We need to know the colors of neighbors.
        
        # Actually, the "remove neighbors" method I used before is a valid coloring algorithm
        # (it's a greedy coloring where each "color class" is an independent set).
        # The number of colors is the bound.
        # Let's stick to the "remove neighbors" method but pick nodes by degree?
        
        # If we pick nodes by degree, the coloring might be tighter?
        # Let's try picking the node with highest degree in P_curr.
        
        P_prime = P
        current_color = 0
        
        while P_prime:
            # Find node with highest degree in P_prime
            # This is expensive.
            # Let's just pick the first one (lowest index) for now.
            lsb = P_prime & -P_prime
            u = lsb.bit_length() - 1
            
            color_node[u] = current_color
            
            # Remove u and neighbors
            P_prime &= ~(new_adj[u] | (1 << u))
            
            current_color += 1
            
        max_color = current_color
        
        if len(R) + max_color <= best_clique_size:
            return
            
        # Branch
        # Process nodes in descending color order
        nodes_to_process = []
        m = P
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes_to_process.append((color_node[idx], idx))
            m ^= lsb
            
        nodes_to_process.sort(reverse=True)
        
        for c, u in nodes_to_process:
            if time.time() > deadline:
                return
            if len(R) + c <= best_clique_size:
                break
                
            R.append(u)
            new_P = P & new_adj[u]
            
            max_clique_optimized(R, new_P)
            
            R.pop()
            P &= ~(1 << u)
            
    # Start search
    # Initial P is all nodes.
    # We should order the initial nodes by degree to help the first coloring?
    # The algorithm sorts nodes_in_P inside, so it handles it.
    
    # Let's run.
    max_clique_optimized([], (1 << n_nodes) - 1)
    
    # 4. Construct Result
    if best_clique_nodes:
        result = []
        for idx in best_clique_nodes:
            r, m = node_info[idx]
            result.append([r, m])
        return result
    else:
        # Fallback
        return [[1, 2]]

# EVOLVE-BLOCK-END
