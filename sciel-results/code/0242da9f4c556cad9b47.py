"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _has_k5_fast(adj_bits, n):
    """Check K5 using bitmasks. adj_bits[i] is a bitmask of neighbors of i."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_bits[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj_bits[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj_bits[c] >> d) & 1:
                        continue
                    # Check if a,b,c,d form K4 and find 5th vertex
                    # All four must be mutually connected
                    if not (adj_bits[a] >> c) & 1: continue
                    if not (adj_bits[a] >> d) & 1: continue
                    if not (adj_bits[b] >> d) & 1: continue
                    # Now find e > d connected to all of a,b,c,d
                    common = adj_bits[a] & adj_bits[b] & adj_bits[c] & adj_bits[d]
                    common &= (1 << (d+1)) - 1  # only vertices <= d, but we need e > d
                    # Actually we need e such that e > d and connected to all
                    # Let's check vertices from d+1 to n-1
                    for e in range(d+1, n):
                        if (common >> e) & 1:
                            return True
    return False

def _has_indep5_fast(adj_bits, n):
    """Check independent set of size 5 using bitmasks."""
    non_adj_bits = []
    for i in range(n):
        # Non-neighbors of i (excluding i itself)
        all_mask = (1 << n) - 1
        all_mask ^= (1 << i)
        non_adj_bits.append(all_mask ^ adj_bits[i])
    
    for a in range(n):
        for b in range(a+1, n):
            if (adj_bits[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if (adj_bits[b] >> c) & 1:
                    continue
                if (adj_bits[a] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if (adj_bits[a] >> d) & 1:
                        continue
                    if (adj_bits[b] >> d) & 1:
                        continue
                    if (adj_bits[c] >> d) & 1:
                        continue
                    # Find e > d not connected to any of a,b,c,d
                    common_non = non_adj_bits[a] & non_adj_bits[b] & non_adj_bits[c] & non_adj_bits[d]
                    for e in range(d+1, n):
                        if (common_non >> e) & 1:
                            return True
    return False

def _is_valid_fast(adj_bits, n):
    """Check no K5 and no independent 5-set using bitmasks."""
    if _has_k5_fast(adj_bits, n):
        return False
    if _has_indep5_fast(adj_bits, n):
        return False
    return True

def _adj_to_bits(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    bits = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _bits_to_adj(bits, n):
    """Convert bitmask list to adjacency matrix."""
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if (bits[i] >> j) & 1:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _paley_graph(q):
    """Construct Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[False] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _try_add_vertex_fast(adj_bits, n, deadline):
    """Try to add a new vertex. Return new adj_bits if valid, else None."""
    all_mask = (1 << n) - 1
    
    # Try random patterns
    for attempt in range(500):
        if time.time() > deadline:
            return None
        new_bits = adj_bits[:]
        new_bit = 0
        for i in range(n):
            if random.random() < 0.5:
                new_bit |= (1 << i)
                new_bits[i] |= (1 << n)
        new_bits.append(new_bit)
        
        if _is_valid_fast(new_bits, n + 1):
            return new_bits
    
    # Try greedy: start with all connections
    new_bits = adj_bits[:]
    new_bit = all_mask
    for i in range(n):
        new_bits[i] |= (1 << n)
    new_bits.append(new_bit)
    
    if _is_valid_fast(new_bits, n + 1):
        return new_bits
    
    # Remove edges one by one
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            return None
        new_bit ^= (1 << i)
        new_bits[i] ^= (1 << n)
        new_bits[n] = new_bit
        if _is_valid_fast(new_bits, n + 1):
            return new_bits
    
    # Try the other way: start with no edges, add one by one
    new_bits = adj_bits[:]
    new_bit = 0
    new_bits.append(new_bit)
    
    if _is_valid_fast(new_bits, n + 1):
        return new_bits
    
    order = list(range(n))
    random.shuffle(order)
    for i in order:
        if time.time() > deadline:
            return None
        new_bit |= (1 << i)
        new_bits[i] |= (1 << n)
        new_bits[n] = new_bit
        if _is_valid_fast(new_bits, n + 1):
            return new_bits
    
    return None

def _try_flip_edge_fast(adj_bits, n, deadline):
    """Try flipping a random edge. Return new adj_bits if valid, else None."""
    if n < 2:
        return None
    i, j = random.sample(range(n), 2)
    new_bits = adj_bits[:]
    if (adj_bits[i] >> j) & 1:
        new_bits[i] &= ~(1 << j)
        new_bits[j] &= ~(1 << i)
    else:
        new_bits[i] |= (1 << j)
        new_bits[j] |= (1 << i)
    if _is_valid_fast(new_bits, n):
        return new_bits
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s to leave safety margin
    
    # Start with Paley graph on 37 vertices
    adj = _paley_graph(37)
    n = 37
    adj_bits = _adj_to_bits(adj, n)
    best_n = n
    best_adj_bits = adj_bits[:]
    
    random.seed(42)
    
    # Phase 1: Try to add vertices
    while n < 46 and time.time() < deadline:
        new_bits = _try_add_vertex_fast(adj_bits, n, deadline)
        if new_bits is not None:
            adj_bits = new_bits
            n += 1
            best_n = n
            best_adj_bits = adj_bits[:]
        else:
            # Try some edge flips to make room, then retry
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge_fast(adj_bits, n, deadline)
                if flipped is not None:
                    adj_bits = flipped
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 2: Continue trying to add vertices with more aggressive search
    while n < 46 and time.time() < deadline:
        added = False
        for trial in range(1000):
            if time.time() > deadline:
                break
            new_bits = adj_bits[:]
            new_bit = 0
            for i in range(n):
                if random.random() < 0.45:
                    new_bit |= (1 << i)
                    new_bits[i] |= (1 << n)
            new_bits.append(new_bit)
            
            if _is_valid_fast(new_bits, n + 1):
                adj_bits = new_bits
                n += 1
                best_n = n
                best_adj_bits = adj_bits[:]
                added = True
                break
        
        if not added:
            # Try edge modifications
            progress = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                flipped = _try_flip_edge_fast(adj_bits, n, deadline)
                if flipped is not None:
                    adj_bits = flipped
                    progress = True
            if not progress:
                break
    
    # Return the best valid graph found
    if best_n >= n:
        adj_bits = best_adj_bits
        n = best_n
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append((i, j))
    
    return n, edges

# EVOLVE-BLOCK-END
