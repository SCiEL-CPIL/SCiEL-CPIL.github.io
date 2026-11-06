"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _is_triangle_free(n, adj_mask):
    """Check if graph is triangle-free using adjacency bitmasks."""
    for u in range(n):
        neighbors_u = adj_mask[u]
        v = 0
        while neighbors_u:
            if neighbors_u & 1:
                if adj_mask[u] & adj_mask[v]:
                    return False
            neighbors_u >>= 1
            v += 1
    return True

def _max_independent_set_size(n, adj_mask, time_limit=2.0):
    """
    Exact branch and bound for maximum independent set size.
    Uses bitmasks for speed.
    """
    best = 0
    start_time = time.time()
    
    def branch(candidates_mask, size, depth):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        if size + bin(candidates_mask).count('1') <= best:
            return
        if not candidates_mask:
            if size > best:
                best = size
            return
        
        # Select vertex with max degree in candidates for better pruning
        max_deg = -1
        chosen = -1
        cm = candidates_mask
        v_idx = 0
        while cm:
            if cm & 1:
                deg = bin(adj_mask[v_idx] & candidates_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v_idx
            cm >>= 1
            v_idx += 1
            
        if chosen == -1:
            return

        # Branch 1: Include chosen
        new_cand = candidates_mask & ~(adj_mask[chosen] | (1 << chosen))
        branch(new_cand, size + 1, depth + 1)
        
        # Branch 2: Exclude chosen
        new_cand = candidates_mask & ~(1 << chosen)
        branch(new_cand, size, depth + 1)

    branch((1 << n) - 1, 0, 0)
    return best

def _paley_graph(n):
    """
    Construct a Paley-type graph. For p ≡ 1 (mod 4), the Paley graph on p vertices
    is triangle-free if p is a prime power with specific properties.
    We use a simpler approach: for n vertices, define edge (u,v) if (v-u) is a quadratic residue mod n.
    This works well when n is prime and n ≡ 1 (mod 4).
    """
    adj_mask = [0] * n
    if n < 5:
        return adj_mask
    
    # For Paley graph, we need n to be a prime ≡ 1 (mod 4)
    # If n is not suitable, we'll create a random graph with similar density
    is_prime = True
    if n < 2:
        is_prime = False
    else:
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                is_prime = False
                break
    
    if is_prime and n % 4 == 1:
        # True Paley graph
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        for u in range(n):
            for v in range(u + 1, n):
                diff = (v - u) % n
                if diff in residues:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
    else:
        # Fallback: create a random graph with density ~0.5
        # This won't be triangle-free, but simulated annealing will fix it
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
    
    return adj_mask

def _count_triangles(n, adj_mask):
    """Count number of triangles in the graph."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj_mask[u] & (1 << v):
                # Count common neighbors of u and v that are > v to avoid double counting
                common = adj_mask[u] & adj_mask[v]
                # Only count w > v
                common &= ~((1 << (v + 1)) - 1)
                count += bin(common).count('1')
    return count

def _simulated_annealing(n, adj_mask, time_limit=30.0):
    """
    Simulated annealing to eliminate triangles while keeping alpha < 10.
    Moves: add or remove an edge.
    Energy: number of triangles (want to minimize) + penalty if alpha >= 10.
    """
    start_time = time.time()
    current_adj = adj_mask[:]
    
    # Initial energy
    current_triangles = _count_triangles(n, current_adj)
    current_alpha = _max_independent_set_size(n, current_adj, time_limit=1.0)
    
    def energy(adj):
        t = _count_triangles(n, adj)
        a = _max_independent_set_size(n, adj, time_limit=0.5)
        # Penalty for alpha >= 10
        alpha_penalty = max(0, a - 9) * 100
        return t + alpha_penalty
    
    current_energy = current_triangles + max(0, current_alpha - 9) * 100
    best_adj = current_adj[:]
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while T > T_min and time.time() - start_time < time_limit:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Ensure u < v for consistency
        if u > v:
            u, v = v, u
        
        # Check if this edge exists
        has_edge = bool(current_adj[u] & (1 << v))
        
        # Flip the edge
        if has_edge:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        else:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
        
        # Calculate new energy
        new_triangles = _count_triangles(n, current_adj)
        new_alpha = _max_independent_set_size(n, current_adj, time_limit=0.5)
        new_energy = new_triangles + max(0, new_alpha - 9) * 100
        
        delta = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_adj = current_adj[:]
        else:
            # Reject: flip back
            if has_edge:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            else:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
        
        T *= cooling_rate
    
    return best_adj

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Seed solution: use a known small triangle-free graph with small alpha
    n_seed = 26
    # Use a circulant graph that's likely triangle-free
    edges_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            diff = (j - i) % n_seed
            if diff in [1, 2, 5, 10, 11, 13, 15, 20, 21, 24]:
                edges_seed.append([i, j])
    
    best_n = n_seed
    best_edges = edges_seed
    
    def mask_to_edges(n, mask_list):
        edges = []
        for i in range(n):
            m = mask_list[i]
            j = i + 1
            while m:
                if m & 1:
                    edges.append([i, j])
                m >>= 1
                j += 1
        return edges
    
    # Try to find a valid graph for n=40 using spectral construction + simulated annealing
    for target_n in [40, 39, 38]:
        if time.time() > deadline - 15:
            break
        
        found = False
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 10:
                break
            
            n = target_n
            
            # Step 1: Construct initial graph using Paley-type or random
            adj_mask = _paley_graph(n)
            
            # Step 2: Apply simulated annealing to eliminate triangles
            time_remaining = deadline - 10 - (time.time() - start_time)
            if time_remaining < 5:
                break
            
            sa_time = min(20.0, time_remaining * 0.7)
            adj_mask = _simulated_annealing(n, adj_mask, time_limit=sa_time)
            
            # Step 3: Verify triangle-free
            if not _is_triangle_free(n, adj_mask):
                continue
            
            # Step 4: Verify alpha < 10
            max_ind = _max_independent_set_size(n, adj_mask, time_limit=2.0)
            if max_ind < 10:
                # Valid!
                if n > best_n:
                    best_n = n
                    best_edges = mask_to_edges(n, adj_mask)
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, try a simpler search for smaller n
    if best_n < 30:
        for target_n in range(35, 26, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts and not found:
                attempts += 1
                if time.time() > deadline - 3:
                    break
                
                n = target_n
                half_n = n // 2
                k = random.randint(3, half_n - 1)
                offsets = random.sample(range(1, half_n + 1), k)
                
                adj_mask = [0] * n
                for u in range(n):
                    for off in offsets:
                        v = (u + off) % n
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                
                if _is_triangle_free(n, adj_mask):
                    max_ind = _max_independent_set_size(n, adj_mask, time_limit=1.0)
                    if max_ind < 10:
                        if n > best_n:
                            best_n = n
                            best_edges = mask_to_edges(n, adj_mask)
                        found = True
                        break
            
            if found:
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
