import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s buffer for safety
    
    # Target sizes to try in descending order
    target_ns = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    
    for n in target_ns:
        if time.time() > deadline - 10.0:
            break
        result = _search_for_graph(n, deadline)
        if result is not None:
            return result
    
    # Fallback: Return a trivially valid small graph
    return {'n': 5, 'edges': []}

def _search_for_graph(n, deadline):
    # Precompute subsets of size 6
    # For n <= 5, no subsets of size 6 exist, so any graph is valid.
    if n < 6:
        return {'n': n, 'edges': []}
    
    # Generate all subsets of size 6
    vertices = list(range(n))
    subsets_6 = list(itertools.combinations(vertices, 6))
    
    # For incremental updates, we need to know which subsets contain a given edge (u, v)
    # Map edge index to list of subset indices
    # Edge index: i < j, idx = i * n + j (or similar)
    # Let's use a 2D array for edges for clarity, or bitstring.
    # Bitstring is faster for flipping.
    # Let's use a list of edges for the final output.
    
    # Simulated Annealing
    # Initial state: random graph
    # Energy: count of regular induced subgraphs of size 6
    
    # To speed up, we only check size 6. 
    # If size 6 count is 0, we then check size 7, 8...
    
    # Precompute for each edge (u,v), the list of subsets of size 6 containing u and v
    # This allows O(1) lookup of affected subsets when an edge is flipped.
    edge_to_subsets = {}
    for idx, sub in enumerate(subsets_6):
        for u, v in itertools.combinations(sub, 2):
            key = (u, v) if u < v else (v, u)
            if key not in edge_to_subsets:
                edge_to_subsets[key] = []
            edge_to_subsets[key].append(idx)
    
    # State: adjacency matrix (0/1)
    # We will use a numpy array for the adjacency matrix for fast degree calculations?
    # Actually, for incremental updates, we just need to update degrees for the affected subsets.
    
    # Let's represent the graph as a set of edges or a 2D boolean array.
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Initialize randomly
    for i in range(n):
        for j in range(i+1, n):
            adj[i, j] = 1 if np.random.random() < 0.5 else 0
            adj[j, i] = adj[i, j]
            
    # Current Energy
    current_energy = _calc_energy(adj, subsets_6)
    
    # If energy is 0, check larger subsets
    if current_energy == 0:
        if _check_larger_subsets(adj, n):
            return _get_edges(adj)
        
    best_energy = current_energy
    best_adj = adj.copy()
    
    T = 1.0
    cooling = 0.999
    # Adjust iterations based on time
    # We don't know how many iterations we can do, so loop until deadline
    
    while time.time() < deadline - 5.0:
        # Pick a random edge to flip
        u = np.random.randint(n)
        v = np.random.randint(n)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate delta energy
        # Affected subsets are those in edge_to_subsets[(u, v)]
        affected = edge_to_subsets.get((u, v), [])
        
        delta = 0
        for s_idx in affected:
            sub = subsets_6[s_idx]
            # Before flip, check if sub is regular
            # We need degrees in sub
            # This is expensive to calculate from scratch every time.
            # We should maintain degree arrays for each subset?
            # That's 38760 subsets * 6 ints = 232k ints. Manageable.
            
        # This incremental approach with maintained state is complex.
        # Alternative: Recalculate energy from scratch if population is small?
        # No, SA needs fast delta.
        
        # Let's try a different approach: Local Search with full recalculation but optimized.
        # Or just use the full recalculation if it's fast enough.
        # 38760 subsets * 6 vertices * 6 edges check?
        # Check regularity: 
        # For a subset of 6, there are 15 edges. 
        # We can precompute the edge mask for each subset.
        # Then for a given graph, we just need to count edges in the subset.
        # But we need degrees.
        
        # Let's stick to the maintained state approach if possible, 
        # or just do random restarts with local search (hill climbing) which is simpler.
        
        pass # Placeholder for the actual loop logic below

    # Fallback if SA loop isn't implemented correctly above
    # Let's implement a simple hill climbing with random restarts instead, it's easier to get right.
    return _hill_climb(n, deadline)

def _calc_energy(adj, subsets_6):
    energy = 0
    n = adj.shape[0]
    for sub in subsets_6:
        # Check if induced subgraph on sub is regular
        # Get degrees
        degs = []
        for v in sub:
            d = 0
            for u in sub:
                if u != v:
                    d += adj[v, u]
            degs.append(d)
        if len(set(degs)) == 1:
            energy += 1
    return energy

def _check_larger_subsets(adj, n):
    # Check subsets of size 7 to n
    vertices = list(range(n))
    for k in range(7, n + 1):
        for sub in itertools.combinations(vertices, k):
            degs = []
            for v in sub:
                d = 0
                for u in sub:
                    if u != v:
                        d += adj[v, u]
                degs.append(d)
            if len(set(degs)) == 1:
                return False
    return True

def _get_edges(adj):
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _hill_climb(n, deadline):
    # Random restarts with local search
    # This is more robust to implement correctly in a short time.
    
    vertices = list(range(n))
    subsets_6 = list(itertools.combinations(vertices, 6))
    
    # Precompute edge lists for each subset for faster degree calc?
    # Or just use the adjacency matrix.
    
    best_adj = None
    best_energy = float('inf')
    
    while time.time() < deadline - 5.0:
        # Random init
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i+1, n):
                if np.random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    
        energy = _calc_energy(adj, subsets_6)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try flipping each edge
            for i in range(n):
                for j in range(i+1, n):
                    # Flip
                    adj[i, j] ^= 1
                    adj[j, i] ^= 1
                    
                    new_energy = _calc_energy(adj, subsets_6)
                    
                    if new_energy < energy:
                        energy = new_energy
                        improved = True
                        if energy == 0:
                            break
                    else:
                        # Revert
                        adj[i, j] ^= 1
                        adj[j, i] ^= 1
                if energy == 0:
                    break
            
        if energy < best_energy:
            best_energy = energy
            best_adj = adj.copy()
            
        if best_energy == 0:
            # Check larger subsets
            if _check_larger_subsets(best_adj, n):
                return _get_edges(best_adj)
            
    # If we didn't find a perfect graph, return the best one found?
    # No, must be valid.
    # If best_energy > 0, this graph is invalid.
    # We should return None to try smaller n.
    return None

# Note: The above _hill_climb might be too slow if _calc_energy is called too many times.
# _calc_energy is O(n^6) roughly? No, O(1) per subset if precomputed, but here it's O(k^2) per subset.
# 38760 subsets * 36 ops = ~1.4M ops per evaluation.
# If we do 1000 evaluations, 1.4B ops. Too slow in Python.

# We need a faster fitness function.
# Let's use the incremental update in the hill climb.

def _hill_climb_fast(n, deadline):
    vertices = list(range(n))
    subsets_6 = list(itertools.combinations(vertices, 6))
    
    # Precompute for each subset the list of edges (indices in the edge list)
    # And for each edge, the list of subsets it belongs to.
    
    # Edge representation: (u, v) with u < v
    edges_list = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[(u, v)] = idx
            edges_list.append((u, v))
            idx += 1
            
    # Map edge to subsets
    edge_to_subs = {i: [] for i in range(len(edges_list))}
    # Map subset to edges
    sub_edges = []
    for sub in subsets_6:
        edges_in_sub = []
        for u, v in itertools.combinations(sub, 2):
            e_idx = edge_index[(u, v) if u < v else (v, u)]
            edges_in_sub.append(e_idx)
        sub_edges.append(edges_in_sub)
        for e_idx in edges_in_sub:
            edge_to_subs[e_idx].append(len(sub_edges) - 1)
            
    # State: bitstring of edges
    # Degrees for each subset: array of 6 ints
    # We maintain:
    # sub_degrees[s_idx][v_idx_in_sub]
    # Is it regular?
    
    # Initialize
    state = np.random.randint(0, 2, len(edges_list), dtype=np.uint8)
    
    # Initialize sub_degrees
    sub_degrees = np.zeros((len(subsets_6), 6), dtype=np.uint8)
    for s_idx, sub in enumerate(subsets_6):
        for v_idx, v in enumerate(sub):
            d = 0
            for u_idx, u in enumerate(sub):
                if u_idx == v_idx: continue
                e_idx = edge_index[(u, v) if u < v else (v, u)]
                d += state[e_idx]
            sub_degrees[s_idx, v_idx] = d
            
    # Initial energy
    energy = 0
    for s_idx in range(len(subsets_6)):
        if len(set(sub_degrees[s_idx])) == 1:
            energy += 1
            
    best_state = state.copy()
    best_energy = energy
    
    while time.time() < deadline - 5.0:
        # Pick random edge to flip
        e_idx = np.random.randint(len(edges_list))
        
        # Calculate delta energy
        delta = 0
        affected_subs = edge_to_subs[e_idx]
        
        # Store old degrees to revert if needed? 
        # Or just calculate new energy for affected subs.
        
        # Flip in state
        state[e_idx] ^= 1
        
        for s_idx in affected_subs:
            sub = subsets_6[s_idx]
            # Find positions of u and v in sub
            u, v = edges_list[e_idx]
            # Update degrees of u and v in this subset
            # Also update degrees of other vertices? 
            # Flipping edge (u,v) changes degree of u and v in the induced subgraph.
            # It does NOT change degrees of other vertices in the subset.
            
            # Find indices of u and v in sub
            idx_u = sub.index(u)
            idx_v = sub.index(v)
            
            old_d_u = sub_degrees[s_idx, idx_u]
            old_d_v = sub_degrees[s_idx, idx_v]
            
            # Check if subset was regular before
            was_reg = (len(set(sub_degrees[s_idx])) == 1)
            
            # Update degrees
            if state[e_idx] == 1:
                # Edge added
                sub_degrees[s_idx, idx_u] += 1
                sub_degrees[s_idx, idx_v] += 1
            else:
                # Edge removed
                sub_degrees[s_idx, idx_u] -= 1
                sub_degrees[s_idx, idx_v] -= 1
                
            # Check if subset is regular now
            is_reg = (len(set(sub_degrees[s_idx])) == 1)
            
            if was_reg and not is_reg:
                delta -= 1
            elif not was_reg and is_reg:
                delta += 1
            
        energy += delta
        
        if energy < best_energy:
            best_energy = energy
            best_state = state.copy()
            
        if best_energy == 0:
            # Check larger subsets
            # Reconstruct adj from best_state
            adj = np.zeros((n, n), dtype=np.uint8)
            for i, e_idx in enumerate(range(len(edges_list))):
                if best_state[e_idx] == 1:
                    u, v = edges_list[e_idx]
                    adj[u, v] = 1
                    adj[v, u] = 1
            if _check_larger_subsets(adj, n):
                return _get_edges(adj)

    # If we exit loop, check if best_state is valid
    if best_energy == 0:
        adj = np.zeros((n, n), dtype=np.uint8)
        for i, e_idx in enumerate(range(len(edges_list))):
            if best_state[e_idx] == 1:
                u, v = edges_list[e_idx]
                adj[u, v] = 1
                adj[v, u] = 1
        if _check_larger_subsets(adj, n):
            return _get_edges(adj)
            
    return None

# Replace the original _hill_climb with _hill_climb_fast in construct
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    target_ns = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    
    for n in target_ns:
        if time.time() > deadline - 10.0:
            break
        result = _hill_climb_fast(n, deadline)
        if result is not None:
            return result
            
    return {'n': 5, 'edges': []}
