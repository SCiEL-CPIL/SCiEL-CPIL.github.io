"""K4-free, no independent 6-set graph maximizer.
Strategy: Spectral + greedy construction.
1) Use a random symmetric matrix to generate an initial adjacency via thresholding.
2) Apply local search (edge flips) to maximize a score that penalizes K4s and independent 6-sets.
3) Use multiple random restarts with different seeds and time budgets.
4) Return the best valid graph found.
"""
import time
import random
import numpy as np

DEADLINE = 140  # safety margin before 150s kill


def _has_k4(adj, n):
    """Check if graph has a K4. adj is list of int bitmasks."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                if mask:
                    return True
    return False


def _has_independent_6(adj, n):
    """Check if graph has an independent set of 6."""
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                return True
    return False


def _validate(adj, n):
    if n < 1 or n > 40:
        return False
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True


def _count_k4s(adj, n, max_count=1000):
    """Count K4s, stop at max_count."""
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                count += mask.bit_count()
                if count >= max_count:
                    return count
    return count


def _count_independent_6(adj, n, max_count=1000):
    """Count independent 6-sets, stop at max_count."""
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                count += 1
                                if count >= max_count:
                                    return count
    return count


def _score(adj, n):
    """Score: negative count of K4s and independent 6-sets. Higher is better."""
    k4s = _count_k4s(adj, n, max_count=10000)
    ind6 = _count_independent_6(adj, n, max_count=10000)
    return -(k4s * 10 + ind6 * 10)


def _local_search(adj, n, time_limit, seed=42):
    """Local search via edge flips."""
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    current_score = _score(adj, n)
    best_adj = adj[:]
    best_score = current_score
    
    # Edge list for random selection
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for iteration in range(5000):
        if time.time() - t0 > time_limit:
            break
        
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip the edge
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_score = _score(adj, n)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            # Revert
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Early stop if valid
        if current_score == 0:
            break
    
    return best_adj, best_score


def _construct_random(n, density, seed):
    """Construct a random graph with given density."""
    random.seed(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _spectral_initial(n, seed):
    """Generate initial graph using spectral method."""
    np.random.seed(seed)
    # Random symmetric matrix
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    # Threshold at median
    median_val = np.median(A)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > median_val:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _greedy_fill(adj, n, time_limit, seed):
    """Greedy fill: add edges where safe."""
    t0 = time.time()
    random.seed(seed)
    
    for iteration in range(1000):
        if time.time() - t0 > time_limit:
            break
        
        # Find an edge that can be added without creating K4
        added = False
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            if (adj[u] >> v) & 1:
                continue
            
            # Check if adding edge creates K4
            # A K4 is created if there exist i, j such that:
            # u-v, u-i, u-j, v-i, v-j, i-j all exist
            # So check common neighbors of u and v that are adjacent to each other
            common = adj[u] & adj[v]
            mask = common & ~(1 << u) & ~(1 << v)
            # Check if any two in mask are adjacent
            creates_k4 = False
            m = mask
            while m:
                b1 = m & (-m)
                i = b1.bit_length() - 1
                m ^= b1
                rest = m
                while rest:
                    b2 = rest & (-rest)
                    j = b2.bit_length() - 1
                    rest ^= b2
                    if (adj[i] >> j) & 1:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if not creates_k4:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
    
    return adj


def construct():
    t0 = time.time()
    best_graph = None
    best_n = 0
    best_score = -10**9
    
    # Try multiple strategies with different seeds and time budgets
    strategies = []
    
    # Strategy 1: Random + local search
    for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for seed in range(10):
            strategies.append(('random', density, seed))
    
    # Strategy 2: Spectral + local search
    for seed in range(20):
        strategies.append(('spectral', 0, seed))
    
    # Strategy 3: Random + greedy fill + local search
    for density in [0.2, 0.3, 0.4]:
        for seed in range(5):
            strategies.append(('greedy_fill', density, seed))
    
    # Shuffle strategies
    random.seed(42)
    random.shuffle(strategies)
    
    for strategy, density, seed in strategies:
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # Time budget per strategy
        time_budget = min(5.0, DEADLINE - 30 - (time.time() - t0))
        if time_budget < 1.0:
            break
        
        n = 36  # Target size
        
        if strategy == 'random':
            adj = _construct_random(n, density, seed)
        elif strategy == 'spectral':
            adj = _spectral_initial(n, seed)
        elif strategy == 'greedy_fill':
            adj = _construct_random(n, density, seed)
            adj = _greedy_fill(adj, n, time_budget * 0.3, seed)
        else:
            continue
        
        # Local search
        adj, score = _local_search(adj, n, time_budget * 0.7, seed)
        
        # Check validity
        if score == 0:
            if n > best_n:
                best_n = n
                best_score = score
                # Convert to edge list
                edge_set = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edge_set.add((i, j))
                best_graph = {'n': n, 'edges': sorted(edge_set)}
        
        # Also try smaller sizes if we haven't found 36
        if best_n < 35:
            for smaller_n in [35, 34, 33]:
                if smaller_n <= best_n:
                    continue
                if time.time() - t0 > DEADLINE - 20:
                    break
                
                if strategy == 'random':
                    adj_small = _construct_random(smaller_n, density, seed)
                elif strategy == 'spectral':
                    adj_small = _spectral_initial(smaller_n, seed)
                else:
                    adj_small = _construct_random(smaller_n, density, seed)
                    adj_small = _greedy_fill(adj_small, smaller_n, time_budget * 0.2, seed)
                
                adj_small, score_small = _local_search(adj_small, smaller_n, time_budget * 0.5, seed)
                
                if score_small == 0:
                    if smaller_n > best_n:
                        best_n = smaller_n
                        best_score = score_small
                        edge_set = set()
                        for i in range(smaller_n):
                            for j in range(i + 1, smaller_n):
                                if (adj_small[i] >> j) & 1:
                                    edge_set.add((i, j))
                        best_graph = {'n': smaller_n, 'edges': sorted(edge_set)}
    
    # Fallback: try known small valid graphs
    if best_graph is None:
        # 5-vertex cycle
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}
    
    return best_graph
