"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def popcount(x):
        return bin(x).count('1')

    def build_polarity_graph_q5():
        """Construct the polarity graph of PG(2,5): 31 vertices, C4-free."""
        points = []
        for x in range(5):
            for y in range(5):
                points.append((x, y, 1))
        for y in range(5):
            points.append((1, y, 0))
        points.append((0, 1, 0))
        
        n = len(points)
        adj_mask = [0] * n
        for i in range(n):
            xi, yi, zi = points[i]
            for j in range(i + 1, n):
                xj, yj, zj = points[j]
                if (xi * xj + yi * yj + zi * zj) % 5 == 0:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        return adj_mask

    def check_c4_free_mask(adj_mask):
        """O(n^2) C4-free check using bitmasks."""
        n = len(adj_mask)
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    if popcount(common) > 1:
                        return False
        return True

    def greedy_independent_number(adj_mask):
        """Fast greedy upper bound on alpha."""
        n = len(adj_mask)
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if (remaining >> v) & 1:
                    deg = popcount(adj_mask[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v < 0:
                break
            count += 1
            remaining &= ~(adj_mask[best_v] | (1 << best_v))
        return count

    def has_independent_set_mask(adj_mask, k, time_limit=None):
        """Exact check for independent set of size k using branch and bound."""
        n = len(adj_mask)
        if k <= 0:
            return True
        if n < k:
            return False
        
        sys.setrecursionlimit(10000)
        best = [0]
        
        def search(v, remaining, size):
            if time_limit is not None and time.time() > time_limit:
                return False
            if size == 0:
                return True
            if v == n:
                return False
            if popcount(remaining) < size:
                return False
            
            # Branch: include v if possible
            if (remaining >> v) & 1:
                new_rem = remaining & ~(adj_mask[v] | (1 << v))
                if search(v + 1, new_rem, size - 1):
                    return True
            
            # Branch: exclude v
            if search(v + 1, remaining, size):
                return True
            return False
        
        all_mask = (1 << n) - 1
        return search(0, all_mask, k)

    def try_greedy_construct(n, seed, deadline):
        """Greedy edge addition with alpha-reduction heuristic."""
        random.seed(seed)
        adj_mask = [0] * n
        
        # Precompute all pairs
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline - 5:
                break
            
            # Check C4-freeness if we add edge (u,v)
            # Condition: no two vertices share >1 common neighbor after addition
            # 1. |N(u) ∩ N(v)| <= 1
            common_uv = adj_mask[u] & adj_mask[v]
            if popcount(common_uv) > 1:
                continue
            
            # 2. For all a in N(v): N(u) ∩ N(a) must be empty
            valid = True
            for a in range(n):
                if (adj_mask[v] >> a) & 1:
                    if adj_mask[u] & adj_mask[a]:
                        valid = False
                        break
            if not valid:
                continue
            
            # 3. For all b in N(u): N(v) ∩ N(b) must be empty
            for b in range(n):
                if (adj_mask[u] >> b) & 1:
                    if adj_mask[v] & adj_mask[b]:
                        valid = False
                        break
            if not valid:
                continue
            
            # Tentatively add
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            # Quick alpha check via greedy
            alpha_ub = greedy_independent_number(adj_mask)
            if alpha_ub >= 11:
                # Revert - this edge doesn't help reduce alpha
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                continue
        
        # Final verification
        if not check_c4_free_mask(adj_mask):
            return None
        
        if has_independent_set_mask(adj_mask, 11, time_limit=deadline - 10):
            return None
        
        edge_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edge_list.append([u, v])
        return {'n': n, 'edges': edge_list}

    def try_extend_polarity(base_mask, target_n, seed, deadline):
        """Extend polarity graph by adding vertices."""
        random.seed(seed)
        n_base = len(base_mask)
        if target_n <= n_base:
            return None
        
        adj_mask = list(base_mask)
        for _ in range(target_n - n_base):
            adj_mask.append(0)
        
        for new_v in range(n_base, target_n):
            if time.time() > deadline - 5:
                break
            
            old_verts = list(range(n_base))
            random.shuffle(old_verts)
            
            for old_v in old_verts:
                if time.time() > deadline - 5:
                    break
                
                # Try adding edge (new_v, old_v)
                adj_mask[new_v] |= (1 << old_v)
                adj_mask[old_v] |= (1 << new_v)
                
                if not check_c4_free_mask(adj_mask):
                    adj_mask[new_v] &= ~(1 << old_v)
                    adj_mask[old_v] &= ~(1 << new_v)
        
        if has_independent_set_mask(adj_mask, 11, time_limit=deadline - 10):
            return None
        
        edge_list = []
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if (adj_mask[u] >> v) & 1:
                    edge_list.append([u, v])
        return {'n': target_n, 'edges': edge_list}

    # Build base: polarity graph of PG(2,5), 31 vertices
    base_mask = build_polarity_graph_q5()
    
    if not check_c4_free_mask(base_mask):
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    base_alpha_ok = not has_independent_set_mask(base_mask, 11, time_limit=deadline - 10)
    
    best_result = None
    
    # Try greedy construction for high n first
    for n in range(43, 38, -1):
        if time.time() > deadline - 15:
            break
        for seed in range(40):
            if time.time() > deadline - 10:
                break
            res = try_greedy_construct(n, seed, deadline)
            if res is not None:
                if best_result is None or res['n'] > best_result['n']:
                    best_result = res
                if res['n'] >= n:
                    return res
    
    # Try extending polarity graph
    if base_alpha_ok:
        for target_n in range(43, 38, -1):
            if time.time() > deadline - 10:
                break
            for seed in range(25):
                if time.time() > deadline - 5:
                    break
                res = try_extend_polarity(base_mask, target_n, seed, deadline)
                if res is not None:
                    if best_result is None or res['n'] > best_result['n']:
                        best_result = res
                    if res['n'] >= target_n:
                        return res
    
    if best_result is not None:
        return best_result
    
    if base_alpha_ok:
        edge_list = []
        n_base = len(base_mask)
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if (base_mask[u] >> v) & 1:
                    edge_list.append([u, v])
        return {'n': n_base, 'edges': edge_list}
    
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
