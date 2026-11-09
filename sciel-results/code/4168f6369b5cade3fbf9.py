import time
import random
import math

DEADLINE = time.time() + 270

def _popcount(x):
    return bin(x).count('1')

def _count_k4(adj, n):
    """Count K4 in graph represented by adj bitsets."""
    total = 0
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                common = ai & adj[j]
                # Count edges in common
                # For each k in common, count neighbors in common
                # Sum degrees in common subgraph / 2
                deg_sum = 0
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    deg_sum += _popcount(adj[k] & common)
                total += deg_sum // 2
    return total // 6

def _count_ind4(adj, n):
    """Count independent sets of size 4."""
    # Independent set in G is K4 in complement
    # Complement adj: ~adj[i] & ((1 << n) - 1)
    # But we can just count K4 in complement
    # To avoid creating complement list every time, we can pass a flag or compute inline
    # For efficiency in SA, we might want to maintain complement or just compute cost directly
    
    # Let's just compute K4 in complement
    total = 0
    mask_all = (1 << n) - 1
    for i in range(n):
        ai = adj[i]
        non_ai = mask_all ^ ai
        for j in range(i + 1, n):
            if not ((ai >> j) & 1):
                common = non_ai & (mask_all ^ adj[j])
                deg_sum = 0
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    # Neighbors of k in common (which are non-neighbors in G)
                    # k is in common, so k is non-neighbor of i and j
                    # We need non-neighbors of k that are also in common
                    non_ak = mask_all ^ adj[k]
                    deg_sum += _popcount(non_ak & common)
                total += deg_sum // 2
    return total // 6

def _cost(adj, n):
    return _count_k4(adj, n) + _count_ind4(adj, n)

def _sa_search(n, max_iters=5000, initial_temp=1.0, cool_rate=0.99):
    """Simulated Annealing to find (5,5)-free graph on n vertices."""
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_cost = _cost(adj, n)
    if current_cost == 0:
        return adj
    
    best_adj = adj[:]
    best_cost = current_cost
    temp = initial_temp
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Flip edge
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        new_cost = _cost(adj, n)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj[:]
                if best_cost == 0:
                    return best_adj
        else:
            # Metropolis criterion
            if random.random() < math.exp(-(new_cost - current_cost) / temp):
                current_cost = new_cost
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        temp *= cool_rate
        if temp < 0.01:
            temp = 0.01
            
    return best_adj

def construct():
    # Try to find larger graphs
    for n in range(43, 37, -1):
        if time.time() > DEADLINE:
            break
        
        # Try multiple restarts for this n
        found = False
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() > DEADLINE:
                break
            adj = _sa_search(n, max_iters=2000)
            if _cost(adj, n) == 0:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return n, edges
        
    # Fallback: Paley 37
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if adj[i] & (1 << j):
                edges.append((i, j))
    return q, edges
