# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80  # 80s safety margin
    
    N = 512
    
    # Precompute lcm compatibility
    # compatible[i] is a set of j where lcm(i,j) <= N
    from math import gcd
    
    # Build adjacency: adj[i] is list of j > i where lcm(i,j) <= N
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum clique via branch and bound (Bron-Kerbosch with pivot)
    # Use bitsets for efficiency
    # Represent each node's neighbors as a Python int bitmask (512 bits)
    n = N
    adj_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in adj[i]:
            mask |= (1 << (j - 1))
        adj_bits[i] = mask
    
    best_clique = []
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        if not P:
            return
        # Upper bound: |R| + |P|
        if len(R) + popcount(P) <= best_size:
            return
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        P_union_X = P | X
        if P_union_X:
            # Find pivot with max neighbors in P
            best_u = -1
            best_count = -1
            # Iterate through bits in P_union_X
            temp = P_union_X
            while temp:
                lb = temp & (-temp)
                u = lb.bit_length()  # 1-indexed position
                temp ^= lb
                count = popcount(P & adj_bits[u])
                if count > best_count:
                    best_count = count
                    best_u = u
                if time.time() > deadline:
                    return
            # Candidates: P \ N(u)
            candidates = P & ~adj_bits[best_u]
        else:
            candidates = P
        
        temp = candidates
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length()
            temp ^= lb
            bron_kerbosch(R | lb, P & adj_bits[v], X & adj_bits[v])
            P ^= lb
            X |= lb
            if time.time() > deadline:
                return
    
    # Initialize with all nodes
    all_nodes = (1 << n) - 1
    bron_kerbosch(0, all_nodes, 0)
    
    if best_clique:
        return best_clique
    else:
        return [1]
# EVOLVE-BLOCK-END
