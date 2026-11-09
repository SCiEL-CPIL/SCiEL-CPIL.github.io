import time
import random
import numpy as np

N = 20
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge_fast(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    adj is a list of int bitmasks.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Candidates for B-side (neighbors of u, excluding v)
    b_mask = nu & ~(1 << v)
    # Candidates for A-side (neighbors of v, excluding u)
    a_mask = nv & ~(1 << u)
    
    # Need at least 2 in each to form K3,3 with u,v
    if _popcount(b_mask) < 2 or _popcount(a_mask) < 2:
        return True
    
    # We need to check if there exist a1, a2 in A and b1, b2 in B
    # such that a1-b1, a1-b2, a2-b1, a2-b2 all exist.
    # This is a K2,2 in the bipartite graph (A, B, E_existing).
    
    # Iterate over pairs in the smaller set to reduce complexity
    # Convert masks to lists of vertices
    a_list = [i for i in range(N) if (a_mask >> i) & 1]
    b_list = [i for i in range(N) if (b_mask >> i) & 1]
    
    if len(a_list) < len(b_list):
        # Iterate pairs in a_list
        for i in range(len(a_list)):
            a1 = a_list[i]
            adj_a1 = adj[a1]
            for j in range(i + 1, len(a_list)):
                a2 = a_list[j]
                adj_a2 = adj[a2]
                # Common neighbors of a1 and a2 within b_mask
                common = adj_a1 & adj_a2 & b_mask
                if _popcount(common) >= 2:
                    return False
    else:
        # Iterate pairs in b_list
        for i in range(len(b_list)):
            b1 = b_list[i]
            adj_b1 = adj[b1]
            for j in range(i + 1, len(b_list)):
                b2 = b_list[j]
                adj_b2 = adj[b2]
                # Common neighbors of b1 and b2 within a_mask
                common = adj_b1 & adj_b2 & a_mask
                if _popcount(common) >= 2:
                    return False
    return True

def _edge_count(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj, time_limit):
    """
    Greedily add all possible edges.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
    return adj

def _fast_local_search(adj, time_limit):
    """
    Fast local search: try removing one edge and adding one or two edges.
    Much faster than the original because we limit how many edges we try to add.
    """
    start_time = time.time()
    current_count = _edge_count(adj)
    best_adj = adj[:]
    best_count = current_count
    
    # Precompute all edges
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Get current missing edges
        missing = [(u, v) for u, v in all_pairs if not ((adj[u] >> v) & 1)]
        existing = [(u, v) for u, v in all_pairs if (adj[u] >> v) & 1]
        
        # Try 1-remove, 2-add swap (net +1)
        for u, v in existing:
            if time.time() - start_time > time_limit:
                return best_adj
            
            # Remove edge (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            added = 0
            new_edges = []
            
            # Try to add up to 2 missing edges
            for u2, v2 in missing:
                if added >= 2:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _can_add_edge_fast(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    added += 1
                    new_edges.append((u2, v2))
            
            if added >= 2:
                # Net gain = added - 1 >= 1
                current_count += (added - 1)
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
            else:
                # Revert
                for eu, ev in new_edges:
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Also try 2-remove, 3-add swaps for bigger jumps
        if time.time() - start_time < time_limit:
            for i in range(len(existing)):
                if time.time() - start_time > time_limit:
                    return best_adj
                u1, v1 = existing[i]
                for j in range(i + 1, len(existing)):
                    if time.time() - start_time > time_limit:
                        return best_adj
                    u2, v2 = existing[j]
                    
                    # Remove both edges
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    
                    added = 0
                    new_edges = []
                    
                    for u3, v3 in missing:
                        if added >= 3:
                            break
                        if (adj[u3] >> v3) & 1:
                            continue
                        if _can_add_edge_fast(adj, u3, v3):
                            adj[u3] |= (1 << v3)
                            adj[v3] |= (1 << u3)
                            added += 1
                            new_edges.append((u3, v3))
                    
                    if added >= 3:
                        current_count += (added - 2)
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                    else:
                        # Revert
                        for eu, ev in new_edges:
                            adj[eu] &= ~(1 << ev)
                            adj[ev] &= ~(1 << eu)
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
        
        if not improved:
            break
            
    return best_adj

def _random_start(density):
    """Generate a random graph with given edge probability, filtered for K3,3."""
    adj = [0] * N
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < density:
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
    return adj

def _algebraic_start():
    """
    Generate a dense, symmetric starting graph using a structured approach.
    Uses a block-circulant design to create a highly symmetric K3,3-free graph.
    """
    adj = [0] * N
    
    # Use a 5x4 block structure with circulant properties
    # Each vertex is (block, position) where block in 0..3, position in 0..4
    # This gives 5 blocks of 4 vertices each
    
    # Define connection patterns using modular arithmetic
    # For blocks i and j, connect vertex (i, a) to (j, b) if (b - a) mod 4 is in S_{i,j}
    
    # We'll use a symmetric pattern to ensure K3,3-freedom
    # Key insight: limit the number of common neighbors between any pair
    
    # Start with a base pattern and verify
    for i in range(5):
        for j in range(5):
            if i == j:
                # Within block: use a cycle or sparse pattern
                for a in range(4):
                    # Connect to next vertex in cycle
                    b = (a + 1) % 4
                    u = i * 4 + a
                    v = i * 4 + b
                    if u < v:
                        if _can_add_edge_fast(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            else:
                # Between blocks: use a circulant pattern
                # Connect based on offset
                offset = (j - i) % 5
                # Use different patterns for different offsets
                if offset == 1 or offset == 4:
                    # Dense connection
                    for a in range(4):
                        for b in range(4):
                            # Connect if (b - a) mod 4 is 0 or 1
                            if (b - a) % 4 in [0, 1]:
                                u = i * 4 + a
                                v = j * 4 + b
                                if u < v:
                                    if _can_add_edge_fast(adj, u, v):
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                elif offset == 2 or offset == 3:
                    # Medium connection
                    for a in range(4):
                        for b in range(4):
                            # Connect if (b - a) mod 4 is 0
                            if (b - a) % 4 == 0:
                                u = i * 4 + a
                                v = j * 4 + b
                                if u < v:
                                    if _can_add_edge_fast(adj, u, v):
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
    
    return adj

def _improved_local_search(adj, time_limit):
    """
    Wrapper that calls _fast_local_search.
    """
    return _fast_local_search(adj, time_limit)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: Algebraic construction
    adj = _algebraic_start()
    adj = _greedy_fill(adj, min(5.0, deadline - time.time() - 10))
    current_count = _edge_count(adj)
    if current_count > best_count:
        best_adj = adj[:]
        best_count = current_count
    
    # Strategy 2: Fast local search on algebraic start
    remaining = deadline - time.time() - 5
    if remaining > 2:
        adj = best_adj[:]
        adj = _fast_local_search(adj, remaining * 0.4)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
    
    # Strategy 3: Random starts with varied densities
    densities = [0.6, 0.7, 0.8, 0.5, 0.65, 0.75]
    random.shuffle(densities)
    
    for density in densities:
        if time.time() > deadline - 10:
            break
            
        adj = _random_start(density)
        
        # Greedy fill
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        adj = _greedy_fill(adj, min(remaining * 0.2, 2.0))
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
        
        # Fast local search
        remaining = deadline - time.time() - 3
        if remaining < 1:
            break
        adj = _fast_local_search(adj, remaining * 0.5)
        
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count
            
    # Final polish: local search on best found
    remaining = deadline - time.time() - 1
    if remaining > 2:
        adj = best_adj[:]
        adj = _fast_local_search(adj, remaining)
        current_count = _edge_count(adj)
        if current_count > best_count:
            best_adj = adj[:]
            best_count = current_count

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
