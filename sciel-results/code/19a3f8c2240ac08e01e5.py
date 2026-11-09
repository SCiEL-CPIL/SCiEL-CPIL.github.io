"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    max_n = 128
    N = max_n
    
    # Precompute lcm table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute conflict structure
    # conflict_pairs[x] = list of (a, b) such that {a, b, x} is a bad triple
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Precompute bitmasks for conflict pairs for faster checking
    conflict_pairs_bits = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a, b in conflict_pairs[x]:
            conflict_pairs_bits[x].append((1 << (a-1), 1 << (b-1)))
    
    # Build weighted conflict graph for spectral approach
    # Edge weight between i and j = number of bad triples containing both i and j
    edge_weight = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a, b in conflict_pairs[x]:
            edge_weight[a][b] += 1
            edge_weight[b][a] += 1
    
    # Compute node "conflict degree" - how many bad triples each node participates in
    node_degree = [0] * (N + 1)
    for x in range(1, N + 1):
        node_degree[x] = len(conflict_pairs[x])
    
    # Spectral relaxation: build adjacency matrix and find leading eigenvector
    # Using power iteration on the conflict graph
    def spectral_embed():
        """Find the leading eigenvector of the conflict graph adjacency matrix."""
        # Initialize with random vector
        rng_spec = random.Random(123)
        v = [rng_spec.uniform(-1, 1) for _ in range(N + 1)]
        v[0] = 0
        
        # Power iteration
        for _ in range(50):
            new_v = [0.0] * (N + 1)
            for i in range(1, N + 1):
                s = 0.0
                for j in range(1, N + 1):
                    if edge_weight[i][j] > 0:
                        s += edge_weight[i][j] * v[j]
                new_v[i] = s
            # Normalize
            norm = math.sqrt(sum(x*x for x in new_v[1:]))
            if norm < 1e-10:
                break
            for i in range(1, N + 1):
                new_v[i] /= norm
            v = new_v
        
        return v
    
    # Greedy with bitmask
    def greedy(order, mask=0):
        for x in order:
            ok = True
            for bit_a, bit_b in conflict_pairs_bits[x]:
                if (mask & bit_a) and (mask & bit_b):
                    ok = False
                    break
            if ok:
                mask |= (1 << (x-1))
        return mask
    
    def mask_to_set(mask):
        return [i+1 for i in range(N) if (mask >> i) & 1]
    
    def mask_size(mask):
        return bin(mask).count("1")
    
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Spectral-guided initialization
    eigvec = spectral_embed()
    
    # Sort by eigenvector value (negative = less conflict participation, prefer these)
    # We want to INCLUDE elements with LOW conflict, so sort by ascending eigvec value
    spec_order = sorted(range(1, N + 1), key=lambda x: eigvec[x])
    m = greedy(spec_order)
    sz = mask_size(m)
    if sz > best_size:
        best_size = sz
        best_mask = m
    
    # Also try reverse spectral order
    spec_order_rev = sorted(range(1, N + 1), key=lambda x: -eigvec[x])
    m = greedy(spec_order_rev)
    sz = mask_size(m)
    if sz > best_size:
        best_size = sz
        best_mask = m
    
    # Phase 2: Standard greedy with various orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: node_degree[x]),
        sorted(range(1, N + 1), key=lambda x: -node_degree[x]),
    ]
    
    for order in orders:
        m = greedy(order)
        sz = mask_size(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Phase 3: Random restarts with greedy
    for trial in range(30000):
        if time.time() > deadline - 8.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        m = greedy(order)
        sz = mask_size(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Phase 4: Iterated Local Search with spectral guidance
    current_mask = best_mask
    current_size = best_size
    improved = True
    
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        for k in [1, 2, 3, 5, 8, 12]:
            if time.time() > deadline - 3.0:
                break
            
            set_bits = [i+1 for i in range(N) if (current_mask >> i) & 1]
            if len(set_bits) < k:
                continue
                
            # Remove k random elements, preferring high-conflict ones
            # Weighted sampling: higher node_degree = more likely to remove
            weights = [node_degree[x] + 1 for x in set_bits]
            total_w = sum(weights)
            removed = []
            available = set(set_bits)
            for _ in range(min(k, len(set_bits))):
                r = rng.random() * total_w
                cum = 0
                chosen = None
                for x in available:
                    cum += node_degree[x] + 1
                    if cum >= r:
                        chosen = x
                        break
                if chosen is None:
                    chosen = list(available)[0]
                removed.append(chosen)
                available.discard(chosen)
                total_w -= node_degree[chosen] + 1
            
            new_mask = current_mask
            for x in removed:
                new_mask &= ~(1 << (x-1))
            
            # Greedily add back using spectral-guided order
            candidates = [x for x in range(1, N + 1) if not (new_mask & (1 << (x-1)))]
            # Sort candidates by eigenvector value (prefer low conflict)
            candidates.sort(key=lambda x: eigvec[x])
            
            m = greedy(candidates, new_mask)
            sz = mask_size(m)
            
            if sz > current_size:
                current_size = sz
                current_mask = m
                if sz > best_size:
                    best_size = sz
                    best_mask = m
                improved = True
                break
    
    # Phase 5: Enhanced Local Search with 1-swap and 2-swap moves
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            set_bits = [i+1 for i in range(N) if (best_mask >> i) & 1]
            not_in = [x for x in range(1, N + 1) if not (best_mask & (1 << (x-1)))]
            
            # 1. Try 1-for-2 swaps (Remove 1, Add 2)
            for rem_x in set_bits:
                if time.time() > deadline - 2.0:
                    break
                trial_mask = best_mask & ~(1 << (rem_x - 1))
                added = 0
                # Sort candidates by spectral value to prefer "better" elements
                candidates_sorted = sorted(not_in, key=lambda x: eigvec[x])
                for add_x in candidates_sorted:
                    if time.time() > deadline - 2.0 or added >= 2:
                        break
                    ok = True
                    for bit_a, bit_b in conflict_pairs_bits[add_x]:
                        if (trial_mask & bit_a) and (trial_mask & bit_b):
                            ok = False
                            break
                    if ok:
                        trial_mask |= (1 << (add_x - 1))
                        added += 1
                if added >= 1:
                    sz = mask_size(trial_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = trial_mask
                        improved = True
                        break
            
            # 2. Try 2-for-2 swaps (Remove 2, Add 2)
            # Sample random pairs to remove to avoid O(N^2) exhaustive search
            if len(set_bits) >= 2:
                # Try a few random pairs
                for _ in range(50):
                    if time.time() > deadline - 2.0:
                        break
                    rem1, rem2 = rng.sample(set_bits, 2)
                    trial_mask = best_mask & ~(1 << (rem1 - 1)) & ~(1 << (rem2 - 1))
                    
                    added = 0
                    # Shuffle candidates to randomize, but maybe bias towards spectral
                    candidates = not_in[:]
                    rng.shuffle(candidates)
                    # Or sort by eigvec?
                    # Let's use random to explore different structures
                    for add_x in candidates:
                        if added >= 2:
                            break
                        ok = True
                        for bit_a, bit_b in conflict_pairs_bits[add_x]:
                            if (trial_mask & bit_a) and (trial_mask & bit_b):
                                ok = False
                                break
                        if ok:
                            trial_mask |= (1 << (add_x - 1))
                            added += 1
                    
                    sz = mask_size(trial_mask)
                    if sz > best_size:
                        best_size = sz
                        best_mask = trial_mask
                        improved = True
                        break
    
    # Final validation
    result = mask_to_set(best_mask)
    if not result:
        return [1]
    
    # Verify no bad triples
    from itertools import combinations
    valid = True
    for a, b, c in combinations(result, 3):
        if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
            valid = False
            break
    
    if not valid:
        # Fallback: rebuild greedily
        m = greedy(list(range(1, N + 1)))
        result = mask_to_set(m)
        if not result:
            return [1]
    
    return result
# EVOLVE-BLOCK-END
