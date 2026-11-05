import random
import time
import itertools

def construct():
    def search(n, time_limit):
        # Precompute subsets of size 6 and 7
        # For incremental updates, we need to know which subsets contain a given pair (u, v)
        subsets_6 = []
        subsets_7 = []
        all_subsets = []
        
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_6.append((mask, comb))
            all_subsets.append((mask, comb, 6))
            
        for comb in itertools.combinations(range(n), 7):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets_7.append((mask, comb))
            all_subsets.append((mask, comb, 7))
            
        # Map pair (u, v) to list of (mask, comb) that contain both
        # u < v to avoid duplicates
        pair_to_subsets = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_to_subsets[(i, j)] = []
                
        for mask, comb, size in all_subsets:
            # For each pair in comb, add this subset to the pair's list
            # This is O(k^2) per subset, which is fine for k=6,7
            for a in range(len(comb)):
                for b in range(a + 1, len(comb)):
                    u, v = comb[a], comb[b]
                    if u > v: u, v = v, u
                    pair_to_subsets[(u, v)].append((mask, comb))
        
        # Initialize graph
        adj = [0] * n
        # Start with random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Precompute initial regularity status
        # is_reg[mask] = True if induced subgraph on mask is regular
        # We use a dict or array indexed by mask? Mask can be large.
        # Let's use a dict: mask -> bool
        is_reg = {}
        current_cost = 0
        
        for mask, comb, size in all_subsets:
            degrees = []
            for v in comb:
                deg = (adj[v] & mask).bit_count()
                degrees.append(deg)
            reg = (len(set(degrees)) == 1)
            is_reg[mask] = reg
            if reg:
                current_cost += 1
                
        best_cost = current_cost
        best_adj = adj[:]
        
        start_time = time.time()
        temp = 1.0
        
        # Local Search
        while time.time() - start_time < time_limit:
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            # Update cost
            new_cost = current_cost
            affected = pair_to_subsets.get((u, v), [])
            for mask, comb in affected:
                # Recalculate degrees for this subset
                degrees = []
                for x in comb:
                    deg = (adj[x] & mask).bit_count()
                    degrees.append(deg)
                is_now_reg = (len(set(degrees)) == 1)
                
                was_reg = is_reg.get(mask, False)
                is_reg[mask] = is_now_reg
                
                if is_now_reg and not was_reg:
                    new_cost += 1
                elif not is_now_reg and was_reg:
                    new_cost -= 1
                    
            delta = new_cost - current_cost
            
            # Accept?
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < pow(2.718, -delta / temp)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        # Verify full validity (k >= 8)
                        # This is expensive, so only do it when cost 0 for 6 and 7 is reached
                        # But we must check k>=8 to be valid!
                        # The task says "No set of 6 or more".
                        # My cost function only tracks 6 and 7.
                        # If cost 6+7 is 0, we MUST check 8..n.
                        valid = True
                        for k in range(8, n + 1):
                            for comb in itertools.combinations(range(n), k):
                                mask = 0
                                for x in comb:
                                    mask |= (1 << x)
                                degrees = []
                                for x in comb:
                                    deg = (adj[x] & mask).bit_count()
                                    degrees.append(deg)
                                if len(set(degrees)) == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return {'n': n, 'edges': edges}
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
            temp *= 0.995
            
        return None

    # Try n=21 first
    # Allocate 60s for n=21
    result = search(21, 60)
    if result:
        return result
        
    # Try n=20
    # Allocate remaining time
    # To be safe, just give it 50s
    result = search(20, 50)
    if result:
        return result
        
    # Fallback: Return a small valid graph
    # n=5 empty graph is valid (no 6-subsets)
    return {'n': 5, 'edges': []}
