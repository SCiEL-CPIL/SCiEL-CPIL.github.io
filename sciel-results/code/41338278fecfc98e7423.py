import math
import sys

def construct():
    P_data = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    n = P_data['n']
    edges = P_data['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    # Precompute max_clique_size and max_ind_size
    max_clique_size = [0] * (1 << n)
    max_ind_size = [0] * (1 << n)
    
    for m in range(1, 1 << n):
        lsb = m & -m
        v_idx = lsb.bit_length() - 1
        rest = m ^ lsb
        
        is_clique[m] = is_clique[rest] and (rest & ~adj[v_idx] == 0)
        is_ind[m] = is_ind[rest] and (rest & adj[v_idx] == 0)
        
        max_clique_size[m] = max(max_clique_size[rest], 1 + max_clique_size[rest & adj[v_idx]])
        
        non_adj_v = full_mask & ~adj[v_idx] & ~lsb
        max_ind_size[m] = max(max_ind_size[rest], 1 + max_ind_size[rest & non_adj_v])

    # Precompute list of all valid masks (cliques or independent sets)
    valid_masks = []
    for m in range(1, 1 << n):
        if is_clique[m] or is_ind[m]:
            valid_masks.append(m)
            
    # Sort valid masks by size descending
    valid_masks.sort(key=lambda x: x.bit_count(), reverse=True)

    sys.setrecursionlimit(10000)

    def solve(k):
        # Greedy check
        def greedy_check(rem, parts_left):
            if rem == 0:
                return True
            if parts_left == 0:
                return False
            best = 0
            best_size = 0
            for vm in valid_masks:
                if (vm & rem) == vm:
                    if vm.bit_count() > best_size:
                        best_size = vm.bit_count()
                        best = vm
            if best == 0:
                return False
            return greedy_check(rem ^ best, parts_left - 1)

        if not greedy_check(full_mask, k):
            return None

        failed_states = {}

        def dfs(rem, parts_left, current_parts):
            if parts_left == 0:
                if rem == 0:
                    return current_parts
                return None
            
            if rem == 0:
                return current_parts

            # Pruning: Lower bound
            max_sz = max(max_clique_size[rem], max_ind_size[rem])
            if max_sz == 0:
                return None
            lb = math.ceil(rem.bit_count() / max_sz)
            if lb > parts_left:
                return None

            # Memoization
            if failed_states.get(rem, 0) >= parts_left:
                return None

            lsb = rem & -rem
            v_idx = lsb.bit_length() - 1
            
            candidates = []
            for vm in valid_masks:
                if (vm & rem) == vm and (vm & lsb) != 0:
                    candidates.append(vm)
            
            # candidates are already sorted by size descending from valid_masks
            # But we filtered them, so relative order is preserved.
            
            for S in candidates:
                new_rem = rem ^ S
                res = dfs(new_rem, parts_left - 1, current_parts + [S])
                if res is not None:
                    return res
            
            failed_states[rem] = parts_left
            return None

        return dfs(full_mask, k, [])

    for k in range(1, n + 1):
        res = solve(k)
        if res is not None:
            result = []
            for mask in res:
                part = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    part.append(idx)
                    m ^= lsb
                result.append(part)
            return result

    return [[i] for i in range(n)]
