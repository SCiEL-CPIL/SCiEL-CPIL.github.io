import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_edges = 0
    best_graph = None
    
    def count_edges(G):
        return sum(G[i].bit_count() for i in range(n)) // 2
    
    def has_k33_fast(G):
        """Check if graph contains K3,3. Returns True if found."""
        for i in range(n):
            for j in range(i+1, n):
                common_ij = G[i] & G[j]
                # For each pair k,l in common_ij (k<l), check if they share >=1 common neighbor
                # Actually we need: exist i,j,k in A and a,b,c in B s.t. all cross edges exist
                # Equivalent: exist triple {i,j,k} with |N(i)∩N(j)∩N(k)| >= 3
                pass
        # More efficient: iterate over triples
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                return True
        return False
    
    def find_k33(G):
        """Find a K3,3 in G. Returns list of 6 edges forming K3,3 or None."""
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                bits = []
                m = mask
                while len(bits) < 3 and m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    m ^= lsb
                return [(i, l) for l in bits] + [(j, l) for l in bits] + [(k, l) for l in bits]
        return None
    
    def is_add_valid(G, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        A new K3,3 must use edge (u,v). So u in part A, v in part B (or vice versa).
        Part A: {u} ∪ (N(v) \ {u}), Part B: {v} ∪ (N(u) \ {v})
        We need to check if there exist a1,a2 in A\{u} and b1,b2 in B\{v}
        such that all four cross-edges exist.
        This is equivalent to: the bipartite graph between A\{u} and B\{v} contains a K2,2.
        """
        A = G[v] & ~(1 << u)  # neighbors of v except u
        B = G[u] & ~(1 << v)  # neighbors of u except v
        
        # Check all pairs in A and all pairs in B for K2,2
        # For each pair (a1, a2) in A, check if they share >= 2 common neighbors in B
        # But we only care about neighbors in B.
        # More efficient: for each pair a1,a2 in A, compute common = G[a1] & G[a2] & B
        # If bit_count(common) >= 2, then K2,2 exists.
        
        # Iterate over pairs in A
        A_list = []
        m = A
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            A_list.append(idx)
            m ^= lsb
        
        for i in range(len(A_list)):
            a1 = A_list[i]
            for j in range(i+1, len(A_list)):
                a2 = A_list[j]
                common = G[a1] & G[a2] & B
                if common.bit_count() >= 2:
                    return False
        return True
    
    def remove_one_k33_edge(G):
        """Find a K3,3 and remove one random edge from it. Returns True if removed."""
        k33 = find_k33(G)
        if k33 is None:
            return False
        u, v = random.choice(k33)
        G[u] &= ~(1 << v)
        G[v] &= ~(1 << u)
        return True
    
    def greedy_add_edges(G, rng):
        """Try to add all possible edges greedily."""
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        added_any = True
        while added_any:
            added_any = False
            rng.shuffle(missing)
            for u, v in missing:
                if not (G[u] & (1 << v)):
                    if is_add_valid(G, u, v):
                        G[u] |= (1 << v)
                        G[v] |= (1 << u)
                        added_any = True
                        missing = [e for e in missing if e != (u, v)]
                        if not missing:
                            break
    
    def local_search_swap(G, rng):
        """Try edge swaps: remove edge (u,v), add edge (x,y) if valid.
        Returns number of successful swaps."""
        swaps = 0
        all_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if G[i] & (1 << j):
                    all_edges.append((i, j))
        
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        if not all_edges or not missing:
            return 0
            
        rng.shuffle(all_edges)
        rng.shuffle(missing)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            # Remove edge (u,v)
            G[u] &= ~(1 << v)
            G[v] &= ~(1 << u)
            
            # Try to add a missing edge
            found = False
            for x, y in missing:
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if G[x] & (1 << y):
                    continue
                if is_add_valid(G, x, y):
                    G[x] |= (1 << y)
                    G[y] |= (1 << x)
                    found = True
                    swaps += 1
                    break
            
            if not found:
                # Put edge back
                G[u] |= (1 << v)
                G[v] |= (1 << u)
        
        return swaps
    
    # Multiple random restarts
    for seed in range(5000):
        if time.time() > deadline:
            break
            
        rng = random.Random(seed)
        
        # Initialize: start with random graph at varying densities
        p = rng.uniform(0.3, 0.7)
        G = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
        
        # Phase 1: Remove edges to make K3,3-free
        max_removals = 200
        removals = 0
        while removals < max_removals:
            if time.time() > deadline:
                break
            if not remove_one_k33_edge(G):
                break
            removals += 1
        
        # Phase 2: Greedy add edges
        greedy_add_edges(G, rng)
        
        # Phase 3: Local search with swaps
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() > deadline:
                    break
                local_search_swap(G, rng)
        
        # Count edges
        ec = count_edges(G)
        if ec > best_edges:
            best_edges = ec
            best_graph = G[:]
    
    # If no valid graph found, return a simple valid one
    if best_graph is None:
        # Return a star graph (valid, K3,3-free)
        edges = [[0, i] for i in range(1, n)]
        return edges
    
    edges = []
    for u in range(n):
        mask = best_graph[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
    return edges
