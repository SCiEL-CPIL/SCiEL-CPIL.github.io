"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute edge indices
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105

    def is_valid_mask(mask):
        """Fast validity check using bitmask adjacency and degree masks."""
        adj = [0] * n
        deg = [0] * n
        
        # Build adjacency and degrees
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        # Precompute degree masks
        # deg_to_mask[d] is a bitmask of vertices with degree d
        max_deg = max(deg) if deg else 0
        deg_to_mask = [0] * (max_deg + 1)
        for v in range(n):
            deg_to_mask[deg[v]] |= (1 << v)

        # Check all edges
        for i in range(E):
            if not (mask & (1 << i)):
                continue
            u, v = edge_list[i]
            
            # Neighbors of u excluding v
            nu = adj[u] & ~(1 << v)
            # Neighbors of v excluding u
            nv = adj[v] & ~(1 << u)
            
            # For each s in nu, check if there is t in nv with same degree and t != s
            # We iterate through bits of nu
            temp_nu = nu
            while temp_nu:
                # Get lowest set bit
                lsb = temp_nu & -temp_nu
                s = lsb.bit_length() - 1
                temp_nu ^= lsb
                
                ds = deg[s]
                # Vertices in nv with degree ds
                candidates_t = nv & deg_to_mask[ds]
                
                if candidates_t:
                    # Check if any t != s
                    if s in nv:
                        # s is a common neighbor. We need another t.
                        if candidates_t & ~(1 << s):
                            return False
                    else:
                        # s is not in nv, so any t in candidates_t is valid (t != s automatically)
                        return False
                        
        return True

    def get_edge_count(mask):
        return bin(mask).count('1')

    best_mask = 0
    best_count = 0
    rng = random.Random(42)

    # Strategy 1: Try different bipartite partitions
    # K_{k, 15-k}
    for k in range(3, 13):
        if time.time() > deadline - 5:
            break
        A = set(range(k))
        B = set(range(k, 15))
        
        init_mask = 0
        for a in A:
            for b in B:
                init_mask |= (1 << edge_to_idx[(a, b)])
        
        if is_valid_mask(init_mask):
            count = get_edge_count(init_mask)
            if count > best_count:
                best_mask = init_mask
                best_count = count

    # If we didn't find anything (should not happen), start from empty
    if best_count == 0:
        for i in range(E):
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1

    # Local Search
    # We use a set of moves:
    # 1. Add 1 edge
    # 2. Remove 1, Add 1
    # 3. Remove 1, Add 2
    # 4. Remove 2, Add 3
    
    present_list = []
    absent_list = []
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline - 5:
            break
            
        # Update present/absent lists
        present_list = [i for i in range(E) if best_mask & (1 << i)]
        absent_list = [i for i in range(E) if not (best_mask & (1 << i))]
        
        if not present_list:
            break

        improved = False

        # 1. Try single add
        if absent_list:
            for i in absent_list:
                new_mask = best_mask | (1 << i)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break

        if improved:
            continue

        # 2. Try 1-for-1 swap
        if absent_list and present_list:
            # Limit attempts to avoid timeout
            attempts = min(20, len(absent_list) * len(present_list) // 10)
            for _ in range(attempts):
                if time.time() > deadline - 5:
                    break
                ri = rng.choice(present_list)
                ai = rng.choice(absent_list)
                new_mask = (best_mask & ~(1 << ri)) | (1 << ai)
                # Count remains same, so we only accept if it allows further growth?
                # No, local search usually accepts same score if it leads to improvement.
                # But checking validity is expensive.
                # Let's only try if it might help.
                # Actually, 1-for-1 doesn't increase count.
                # We should focus on increasing count.
                # So skip 1-for-1 unless stuck in a way that allows 1-for-2.
                pass

        # 3. Try 1-for-2 swap
        if len(absent_list) >= 2 and present_list:
            attempts = min(50, max(5, len(absent_list) * len(present_list) // 10))
            for _ in range(attempts):
                if time.time() > deadline - 5:
                    break
                ri = rng.choice(present_list)
                a1, a2 = rng.sample(absent_list, 2)
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break

        if improved:
            continue

        # 4. Try 2-for-3 swap
        if len(absent_list) >= 3 and len(present_list) >= 2:
            attempts = min(30, max(3, len(absent_list) * len(present_list) // 20))
            for _ in range(attempts):
                if time.time() > deadline - 5:
                    break
                r1, r2 = rng.sample(present_list, 2)
                a1, a2, a3 = rng.sample(absent_list, 3)
                new_mask = (best_mask & ~(1 << r1) & ~(1 << r2)) | (1 << a1) | (1 << a2) | (1 << a3)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 2
                    improved = True
                    break

        if improved:
            continue

        # 5. Perturbation: Remove 3 random edges, re-add greedily
        if _ % 50 == 25:
            new_mask = best_mask
            if len(present_list) >= 3:
                for _ in range(3):
                    ri = rng.choice(present_list)
                    new_mask &= ~(1 << ri)
            
            # Re-add greedily
            for i in range(E):
                if new_mask & (1 << i):
                    continue
                try_mask = new_mask | (1 << i)
                if is_valid_mask(try_mask):
                    new_mask = try_mask
            
            new_count = get_edge_count(new_mask)
            if new_count > best_count:
                best_mask = new_mask
                best_count = new_count

    # Final Polish: Exhaustive single adds
    if time.time() < deadline - 2:
        absent_list = [i for i in range(E) if not (best_mask & (1 << i))]
        for i in absent_list:
            if time.time() > deadline - 2:
                break
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1

    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
