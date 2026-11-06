import time
import random
import numpy as np
from math import comb

def _paleyn_tournament(p):
    """Build a Paley tournament on p vertices (p must be prime, p ≡ 3 mod 4)."""
    squares = set(x*x % p for x in range(1, p))
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _energy(n, orientation):
    """Count the number of 4-subsets that lack a common dominator."""
    # Build out-neighborhood bitsets
    # orientation[k] corresponds to pair (i,j) where k = i*(n-1) - i*(i+1)//2 + (j-i-1)
    # Actually let's use a simpler indexing: for pair (i,j) with i<j, index = sum_{m=0}^{i-1}(n-1-m) + (j-i-1)
    # Let's precompute offsets
    offset = [0] * (n + 1)
    for i in range(n):
        offset[i] = i * (n - 1) - i * (i + 1) // 2
    
    # out_mask[v] is a bitmask of vertices that v beats
    out_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            idx = offset[i] + (j - i - 1)
            if orientation[idx] == 0:
                # i -> j
                out_mask[i] |= (1 << j)
            else:
                # j -> i
                out_mask[j] |= (1 << i)
    
    # For each 4-subset {a,b,c,d}, check if there's a vertex v that beats all four
    # v can be in the subset or outside
    # A 4-subset is "bad" if no vertex dominates all 4
    
    bad_count = 0
    total_4 = comb(n, 4)
    
    # For each 4-subset, we need to check if any vertex v has all 4 in out_mask[v]
    # This is equivalent to: the intersection of out_mask over the 4 vertices (considering both directions)
    # Actually, for a fixed 4-set S = {a,b,c,d}, a vertex v dominates S if v->a, v->b, v->c, v->d
    # v can be in S or not. If v is in S, say v=a, then we need a->b, a->c, a->d
    # If v is not in S, we need v->a, v->b, v->c, v->d
    
    # More efficient: for each 4-set, compute the set of vertices that dominate it
    # A vertex v dominates {a,b,c,d} iff (out_mask[v] has bits a,b,c,d all set)
    
    # Let's iterate over all 4-subsets
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    # Check if any vertex dominates {a,b,c,d}
                    dominated = False
                    for v in range(n):
                        if ((out_mask[v] >> a) & 1) and ((out_mask[v] >> b) & 1) and \
                           ((out_mask[v] >> c) & 1) and ((out_mask[v] >> d) & 1):
                            dominated = True
                            break
                    if not dominated:
                        bad_count += 1
    
    return bad_count

def _energy_fast(n, orientation):
    """Faster energy using numpy bit operations."""
    offset = [0] * (n + 1)
    for i in range(n):
        offset[i] = i * (n - 1) - i * (i + 1) // 2
    
    # Build adjacency matrix as numpy array
    # adj[i,j] = 1 if i -> j, 0 if j -> i, -1 if i==j
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            idx = offset[i] + (j - i - 1)
            if orientation[idx] == 0:
                adj[i, j] = 1
                adj[j, i] = 0
            else:
                adj[i, j] = 0
                adj[j, i] = 1
    
    # For each 4-subset, count if any vertex dominates all 4
    # Use numpy for speed
    bad_count = 0
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    # For each vertex v, check if v->a, v->b, v->c, v->d
                    # adj[:, [a,b,c,d]] gives a column vector for each v
                    # We need all 1s
                    sub_adj = adj[:, [a, b, c, d]]  # shape (n, 4)
                    if np.all(sub_adj == 1, axis=1).any():
                        continue
                    else:
                        bad_count += 1
    
    return bad_count

def _simulated_annealing(n, max_time, seed=42):
    """Run simulated annealing to minimize energy."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    # Start with random tournament
    orientation = [rng.randint(0, 1) for _ in range(comb(n, 2))]
    current_energy = _energy(n, orientation)
    
    if current_energy == 0:
        return orientation
    
    best_orientation = orientation[:]
    best_energy = current_energy
    
    # Precompute offset for edge indexing
    offset = [0] * (n + 1)
    for i in range(n):
        offset[i] = i * (n - 1) - i * (i + 1) // 2
    
    # Build out_mask for fast energy delta computation
    def compute_out_mask(orient):
        out_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                idx = offset[i] + (j - i - 1)
                if orient[idx] == 0:
                    out_mask[i] |= (1 << j)
                else:
                    out_mask[j] |= (1 << i)
        return out_mask
    
    out_mask = compute_out_mask(orientation)
    
    # Precompute all 4-subsets
    four_sets = []
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    four_sets.append((a, b, c, d))
    
    def is_dominated(outm, quad):
        a, b, c, d = quad
        for v in range(n):
            if ((outm[v] >> a) & 1) and ((outm[v] >> b) & 1) and \
               ((outm[v] >> c) & 1) and ((outm[v] >> d) & 1):
                return True
        return False
    
    start_time = time.time()
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    
    step = 0
    while time.time() - start_time < max_time:
        if T < T_min:
            break
        
        # Pick a random edge to flip
        idx = rng.randint(0, comb(n, 2) - 1)
        
        # Find which pair this edge corresponds to
        # Binary search for the pair
        i = 0
        while offset[i + 1] > idx:
            i += 1
        j = i + 1 + (idx - offset[i])
        
        old_val = orientation[idx]
        new_val = 1 - old_val
        orientation[idx] = new_val
        
        # Update out_mask
        if old_val == 0:
            # Was i -> j, now j -> i
            out_mask[i] &= ~(1 << j)
            out_mask[j] |= (1 << i)
        else:
            # Was j -> i, now i -> j
            out_mask[j] &= ~(1 << i)
            out_mask[i] |= (1 << j)
        
        # Compute new energy (expensive, but we need it)
        # Optimization: only check 4-sets that include i or j
        new_energy = current_energy
        
        # First, undo the old contribution of 4-sets involving i or j
        # Then add the new contribution
        # Actually, let's just recompute energy for 4-sets involving i or j
        
        # 4-sets involving i: choose 3 from the other n-1 vertices
        # 4-sets involving j: choose 3 from the other n-1 vertices
        # But we might double-count 4-sets involving both i and j
        
        # Simpler: just recompute full energy. For n=31, comb(31,4) = 31465, which is manageable.
        new_energy = _energy(n, orientation)
        
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_orientation = orientation[:]
        else:
            # Revert
            orientation[idx] = old_val
            if old_val == 0:
                out_mask[i] |= (1 << j)
                out_mask[j] &= ~(1 << i)
            else:
                out_mask[j] |= (1 << i)
                out_mask[i] &= ~(1 << j)
        
        T *= cooling_rate
        step += 1
    
    return best_orientation

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # First, try Paley tournaments for known valid sizes
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    for p in candidates:
        if time.time() > deadline:
            break
        
        squares = {x*x % p for x in range(1, p)}
        
        # Quick validity check using bitsets
        out_mask = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    out_mask[i] |= (1 << j)
                else:
                    out_mask[j] |= (1 << i)
        
        valid = True
        for a in range(p - 3):
            for b in range(a + 1, p - 2):
                for c in range(b + 1, p - 1):
                    for d in range(c + 1, p):
                        dominated = False
                        for v in range(p):
                            if ((out_mask[v] >> a) & 1) and ((out_mask[v] >> b) & 1) and \
                               ((out_mask[v] >> c) & 1) and ((out_mask[v] >> d) & 1):
                                dominated = True
                                break
                        if not dominated:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
    
    # If no Paley tournament works, try simulated annealing for small n
    # Start from n=31 (lower bound) and go up
    remaining_time = deadline - time.time()
    
    if remaining_time > 5:
        # Try simulated annealing for n=31 to n=40
        for n in range(31, 41):
            if time.time() > deadline:
                break
            
            time_per_n = min(10, max(2, (deadline - time.time()) / (41 - n)))
            
            best_orient = _simulated_annealing(n, time_per_n, seed=n)
            energy = _energy(n, best_orient)
            
            if energy == 0:
                return {'n': n, 'orientation': best_orient}
    
    # Fallback: return Paley tournament on 83 vertices
    p = 83
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
