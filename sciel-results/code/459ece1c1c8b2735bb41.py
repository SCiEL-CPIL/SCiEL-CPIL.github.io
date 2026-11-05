import time
import random

_START = time.time()
_DEADLINE = 140  # seconds, leave 10s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            # Common neighbors of i and j, with index > j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            # Extract bits
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick vertex with fewest neighbors in candidates (most constraining)
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (adj[v] & candidates).bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
        
        candidates &= ~(1 << best_v)
        
        # Branch 1: Include best_v
        new_candidates = candidates & ~adj[best_v]
        res = backtrack(new_candidates, current_set + [best_v])
        if res:
            return res
        
        # Branch 2: Exclude best_v
        res = backtrack(candidates, current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _count_k4s(adj, n):
    """Count K4s in the graph."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        count += 1
    return count

def _count_ind_6s(adj, n, limit=100):
    """Count independent 6-sets (up to limit for speed)."""
    count = 0
    
    def backtrack(candidates, current_set):
        nonlocal count
        if count >= limit:
            return
        if len(current_set) == 6:
            count += 1
            return
        if candidates.bit_count() < 6 - len(current_set):
            return
        
        # Pick vertex with fewest neighbors in candidates
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            cnt = (adj[v] & candidates).bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return
        
        candidates &= ~(1 << best_v)
        
        # Include best_v
        new_candidates = candidates & ~adj[best_v]
        backtrack(new_candidates, current_set + [best_v])
        
        # Exclude best_v
        backtrack(candidates, current_set)
    
    backtrack((1 << n) - 1, [])
    return count

def _try_local_search(n, time_limit, seed_adj=None):
    """Try to find a valid graph on n vertices using local search."""
    start = time.time()
    
    if seed_adj is not None:
        adj = seed_adj[:]
    else:
        # Initialize with random graph biased toward moderate density
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Target density ~0.5 for balance between K4 and independent sets
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    
    best_score = float('inf')
    best_adj = None
    
    while time.time() - start < time_limit:
        # Evaluate current state
        k4_count = _count_k4s(adj, n)
        ind6_count = _count_ind_6s(adj, n, limit=50)
        
        score = k4_count * 10 + ind6_count
        
        if best_adj is None or score < best_score:
            best_score = score
            best_adj = adj[:]
        
        if score == 0:
            # Found valid graph
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return edges
        
        # If too many violations, do a bigger perturbation
        if score > 50 and random.random() < 0.3:
            # Reset to random
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            continue
        
        # Try multiple random edge flips
        num_flips = random.randint(1, 3)
        for _ in range(num_flips):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Decide flip direction based on current state
            if adj[i] & (1 << j):
                # Edge exists: consider removing
                # Remove if it's in many K4s
                if k4_count > ind6_count and random.random() < 0.6:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                elif random.random() < 0.3:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            else:
                # No edge: consider adding
                # Add if it breaks many independent sets
                if ind6_count > k4_count and random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                elif random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Simulated annealing style: occasionally accept worse moves
        # (handled by the randomization above)
    
    # Return best found if it's valid
    if best_adj is not None:
        k4 = _has_k4(best_adj, n)
        ind6 = _has_ind_6(best_adj, n)
        if k4 is None and ind6 is None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return edges
    
    return None

def _extend_graph(base_adj, base_n, target_n):
    """Try to extend a valid graph by adding vertices."""
    adj = base_adj[:] + [0] * (target_n - base_n)
    
    # For each new vertex, connect to existing vertices
    # to maintain K4-free and no independent 6-set
    for new_v in range(base_n, target_n):
        # Try to add edges to make new_v fit
        # Connect to vertices that are in small cliques
        for old_v in range(base_n):
            if random.random() < 0.5:
                adj[new_v] |= (1 << old_v)
                adj[old_v] |= (1 << new_v)
    
    return adj

def construct():
    # Start with Paley 17 as base
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    best_adj = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    
    # Try to find larger graphs
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time allocation: more time for larger n
        if n <= 20:
            time_slice = 3.0
        elif n <= 25:
            time_slice = 2.0
        elif n <= 30:
            time_slice = 1.5
        else:
            time_slice = 1.0
        
        # Try multiple restarts
        found = None
        attempts = 3 if n <= 25 else 2
        
        for attempt in range(attempts):
            if _elapsed() > _DEADLINE - 5:
                break
            
            # Use seed from previous best for first attempt
            if attempt == 0 and best_n > 17:
                # Extend the best graph
                seed = _extend_graph(best_adj, best_n, n)
            else:
                seed = None
            
            t_slice = time_slice / attempts
            found = _try_local_search(n, t_slice, seed)
            
            if found is not None:
                break
        
        if found is not None:
            best_n = n
            best_edges = found
            # Update best_adj
            best_adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i] & (1 << j):
                        pass
            # Rebuild adj from edges
            best_adj = [0] * n
            for e in best_edges:
                u, v = e
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
    
    return {'n': best_n, 'edges': best_edges}
