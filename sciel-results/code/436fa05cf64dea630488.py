import time
import random
import math

_START = time.time()
_DEADLINE = 140  # seconds

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns count and one K4 if found."""
    count = 0
    found_k4 = None
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            common &= ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                rest = common & ~( (1 << (k+1)) - 1 )
                if rest & adj[k]:
                    count += 1
                    if found_k4 is None:
                        r = rest & adj[k]
                        lsb2 = r & -r
                        l = lsb2.bit_length() - 1
                        found_k4 = [i, j, k, l]
    return count, found_k4

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns count and one set if found."""
    # Use a faster heuristic or exact count if possible
    # For SA, we just need to know if > 0
    # We can use a greedy approximation or exact backtracking
    # Exact backtracking is slow.
    # Let's use a greedy check first, then backtrack if needed.
    
    def backtrack(candidates, current_set, depth):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick vertex with fewest non-neighbors
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
            
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        res = backtrack(new_candidates, current_set + [v], depth + 1)
        if res:
            return res
        
        res = backtrack(candidates & ~(1 << v), current_set, depth + 1)
        if res:
            return res
        
        return None

    ind6 = backtrack((1 << n) - 1, [], 0)
    if ind6:
        return 1, ind6
    return 0, None

def _energy(adj, n):
    """Calculate energy: number of K4 + number of Ind6."""
    k4_count, _ = _has_k4(adj, n)
    ind6_count, _ = _has_ind_6(adj, n)
    return k4_count + ind6_count

def _try_simulated_annealing(n, adj_init, time_limit, seed=None):
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    adj = list(adj_init)
    current_energy = _energy(adj, n)
    best_adj = adj[:]
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995  # cooling rate
    
    while time.time() - start < time_limit:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate delta energy
        # This is expensive if we recalculate full energy.
        # But n <= 40, so it's okay.
        
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        new_energy = _energy(adj, n)
        delta = new_energy - current_energy
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_adj = adj[:]
                if best_energy == 0:
                    break
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    if best_energy == 0:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    return None

def _random_adj(n, p_edge):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    # Start with Paley 17
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs
    for n in range(18, 36):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time allocation
        if n <= 22:
            time_slice = 5.0
        elif n <= 28:
            time_slice = 4.0
        elif n <= 32:
            time_slice = 3.0
        else:
            time_slice = 2.0
            
        if _elapsed() + time_slice > _DEADLINE - 3:
            time_slice = _DEADLINE - 3 - _elapsed()
            if time_slice < 0.5:
                break
                
        found = False
        
        # Strategy 1: Simulated Annealing from random initialization
        for attempt in range(2):
            if _elapsed() > _DEADLINE - 3:
                break
            alloc = time_slice / 2
            adj_init = _random_adj(n, 0.4)
            edges = _try_simulated_annealing(n, adj_init, alloc)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
                
        if found:
            continue
            
        # Strategy 2: Simulated Annealing from Paley 17 extension
        if _elapsed() + 1.0 < _DEADLINE - 3:
            paley_adj = [0] * 17
            for i in range(17):
                for j in range(i + 1, 17):
                    if (j - i) % 17 in squares:
                        paley_adj[i] |= (1 << j)
                        paley_adj[j] |= (1 << i)
                        
            adj_ext = [0] * n
            for i in range(17):
                adj_ext[i] = paley_adj[i]
                
            for i in range(17, n):
                for j in range(i):
                    if random.random() < 0.4:
                        adj_ext[i] |= (1 << j)
                        adj_ext[j] |= (1 << i)
                        
            edges = _try_simulated_annealing(n, adj_ext, min(2.0, _DEADLINE - 3 - _elapsed()))
            if edges is not None:
                best_n = n
                best_edges = edges
                continue
                
    return {'n': best_n, 'edges': best_edges}
