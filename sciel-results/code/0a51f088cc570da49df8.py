import time
import random

def construct():
    start_time = time.time()
    # Leave 10s for safety
    deadline = start_time + 140
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def get_subsets(n, k):
        from itertools import combinations
        masks = []
        for comb in combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks.append(mask)
        return masks

    def is_regular(adj, mask, n):
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        if len(verts) < 6:
            return False
            
        d0 = popcount(adj[verts[0]] & mask)
        for v in verts[1:]:
            d = popcount(adj[v] & mask)
            if d != d0:
                return False
        return True

    def count_bad(adj, n, subsets_by_size):
        bad_count = 0
        for k, masks in subsets_by_size.items():
            for mask in masks:
                if is_regular(adj, mask, n):
                    bad_count += 1
        return bad_count

    def try_find(n, time_limit):
        end_time = time.time() + time_limit
        
        # Precompute subsets
        subsets_by_size = {}
        for k in range(6, n + 1):
            subsets_by_size[k] = get_subsets(n, k)
            
        # Initial random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        bad_count = count_bad(adj, n, subsets_by_size)
        
        if bad_count == 0:
            return adj, 0
            
        best_adj = adj[:]
        best_bad = bad_count
        
        while time.time() < end_time:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            bit = 1 << j
            is_present = bool(adj[i] & bit)
            
            # Flip
            if is_present:
                adj[i] ^= bit
                adj[j] ^= bit
            else:
                adj[i] |= bit
                adj[j] |= bit
                
            new_bad = count_bad(adj, n, subsets_by_size)
            
            if new_bad < best_bad:
                best_bad = new_bad
                best_adj = adj[:]
                if best_bad == 0:
                    return best_adj, 0
            else:
                # Revert
                if is_present:
                    adj[i] |= bit
                    adj[j] |= bit
                else:
                    adj[i] ^= bit
                    adj[j] ^= bit
                    
        return best_adj, best_bad

    # Try n=21 with 30s
    if time.time() < deadline - 100:
        adj21, bad21 = try_find(21, 30)
        if bad21 == 0:
            edges = []
            for i in range(21):
                for j in range(i + 1, 21):
                    if adj21[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 21, 'edges': edges}
            
    # Try n=20 with remaining time
    remaining = deadline - time.time() - 5
    if remaining > 10:
        adj20, bad20 = try_find(20, remaining)
        if bad20 == 0:
            edges = []
            for i in range(20):
                for j in range(i + 1, 20):
                    if adj20[i] & (1 << j):
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}
            
    # Fallback
    return {'n': 5, 'edges': []}
