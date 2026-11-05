"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def _popcount(x):
    return x.bit_count()

def _is_regular_subset(adj, mask, verts):
    """Check if the subset defined by mask (and verts list) is regular."""
    k = len(verts)
    if k < 6:
        return False
    first_deg = -1
    for v in verts:
        deg = _popcount(adj[v] & mask)
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _check_all_regular(adj, n):
    """Check if there is any regular induced subgraph of size >= 6."""
    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_subset(adj, mask, comb):
                return True, comb
    return False, None

def _local_search_v2(n, seed, time_limit):
    """
    Local search that minimizes the number of regular 6-vertex induced subgraphs.
    Once 0 is reached, checks for larger regular subgraphs.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Precompute all 6-subsets
    subsets6 = list(itertools.combinations(range(n), 6))
    num_subsets = len(subsets6)
    
    # Precompute masks for subsets
    subset_masks = [0] * num_subsets
    for idx, comb in enumerate(subsets6):
        m = 0
        for v in comb:
            m |= (1 << v)
        subset_masks[idx] = m
        
    # Precompute which subsets are affected by each edge
    # edge_key -> list of subset indices
    affected_by_edge = {}
    for idx, comb in enumerate(subsets6):
        for i in range(6):
            for j in range(i + 1, 6):
                u, v = comb[i], comb[j]
                key = (min(u, v), max(u, v))
                if key not in affected_by_edge:
                    affected_by_edge[key] = []
                affected_by_edge[key].append(idx)
                
    # Initial state: count regular 6-subsets
    is_reg = [False] * num_subsets
    reg_count = 0
    for idx in range(num_subsets):
        if _is_regular_subset(adj, subset_masks[idx], subsets6[idx]):
            is_reg[idx] = True
            reg_count += 1
            
    best_reg_count = reg_count
    
    # If we start with 0, check larger immediately
    if reg_count == 0:
        valid, viol = _check_all_regular(adj, n)
        if not valid:
            return adj
    
    for it in range(1000000): # High limit, time checked inside
        if time.time() > start_time + time_limit:
            break
            
        # If no regular 6-subsets, check for larger
        if reg_count == 0:
            valid, viol = _check_all_regular(adj, n)
            if not valid:
                return adj
            # If there is a larger violation, try to break it
            # Pick a random edge from the violation
            if viol:
                # viol is a tuple of vertices
                u, v = rng.sample(list(viol), 2)
            else:
                u, v = rng.sample(range(n), 2)
        else:
            # Pick a random regular 6-subset and flip an edge in it
            # To be efficient, we can just pick a random edge that is in a regular subset
            # Or pick a random regular subset index
            # Finding a random True in is_reg is O(N) if we scan. 
            # We can maintain a list of regular indices, but it's dynamic.
            # For now, just pick a random index and check. If not reg, try a few times.
            # Better: Pick a random edge, if it's in a reg subset, flip.
            # Let's stick to: Pick random edge (u,v).
            # If flipping it reduces reg_count, accept.
            # If it increases, accept with prob exp(-delta/T) (Simulated Annealing) or just accept if small increase.
            # Simpler: Hill climbing. Only accept if reg_count decreases.
            # If stuck, restart? Or random walk?
            
            # Let's do a random walk with acceptance:
            # 1. Pick random edge (u,v)
            # 2. Calculate new reg_count for affected subsets
            # 3. If new_count < old_count, accept.
            # 4. If new_count == old_count, accept with prob 0.5.
            # 5. If new_count > old_count, accept with prob 0.1 (to escape local minima).
            
            u, v = rng.sample(range(n), 2)
            key = (min(u, v), max(u, v))
            if key not in affected_by_edge:
                continue
                
            affected_indices = affected_by_edge[key]
            
            # Calculate delta
            old_count = reg_count
            new_count = reg_count
            
            # We need to check the state of these subsets BEFORE and AFTER flip
            # Since we only change one edge, the degree of vertices in the subset changes by +/- 1 for u and v.
            # Actually, it's easier to just re-evaluate the subset regularity.
            
            # Store old states
            old_states = []
            for idx in affected_indices:
                old_states.append(is_reg[idx])
                
            # Flip edge
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # Update states
            for i, idx in enumerate(affected_indices):
                new_reg = _is_regular_subset(adj, subset_masks[idx], subsets6[idx])
                if new_reg != old_states[i]:
                    if new_reg:
                        new_count += 1
                    else:
                        new_count -= 1
                is_reg[idx] = new_reg
                
            delta = new_count - old_count
            
            # Acceptance criteria
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if rng.random() < 0.5:
                    accept = True
            else:
                # Accept some uphill moves to escape local minima
                if rng.random() < 0.2:
                    accept = True
                    
            if accept:
                reg_count = new_count
                if reg_count < best_reg_count:
                    best_reg_count = reg_count
            else:
                # Revert flip
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                # Revert states
                for i, idx in enumerate(affected_indices):
                    is_reg[idx] = old_states[i]

    # If we exit loop, check if we have a valid graph (reg_count 0 and no larger)
    if reg_count == 0:
        valid, _ = _check_all_regular(adj, n)
        if not valid:
            return adj
            
    return None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_result = {'n': 5, 'edges': []}
    
    # Try n=21 first, then n=20
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        # Run local search with multiple seeds
        found = False
        for seed in range(100):
            if time.time() > deadline - 5:
                break
                
            # Allocate time per seed
            # If we have 100s left, and 100 seeds, 1s per seed?
            # Let's give 2s per seed initially, then reduce.
            remaining = deadline - time.time()
            seeds_left = 100 - seed
            time_per_seed = max(0.5, remaining / max(1, seeds_left))
            
            adj = _local_search_v2(n, seed, time_per_seed)
            if adj is not None:
                best_result = {'n': n, 'edges': _edges_from_adj(adj, n)}
                found = True
                break
                
        if found:
            continue
            
    return best_result
# EVOLVE-BLOCK-END
