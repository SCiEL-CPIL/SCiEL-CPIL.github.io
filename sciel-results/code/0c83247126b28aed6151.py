import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin under the 120s/150s limit

    # Fallback: Andrásfai graph And(9) on 26 vertices
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed)
                  if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def greedy_alpha(adj, n):
        """
        Returns the size of an independent set found by a greedy heuristic.
        This is a lower bound on alpha. If it returns >= 10, alpha >= 10.
        """
        mask = (1 << n) - 1
        size = 0
        while mask:
            best_v = -1
            min_deg = n + 1
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            size += 1
            # Exclude best_v and all its neighbors from future consideration
            mask &= ~(1 << best_v)
            mask &= ~adj[best_v]
            
        return size

    def has_independent_set_10(adj, n):
        """
        Exact branch and bound check for independent set of size 10.
        Returns True if such a set exists.
        """
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]

        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False

            # Heuristic: pick vertex with fewest non-neighbors in mask
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 0:
                        break
                temp_mask ^= lsb

            if best_v == -1:
                return False

            v_bit = 1 << best_v

            # Include best_v
            if dfs(non_neighbors[best_v] & mask, size + 1):
                return True

            # Exclude best_v
            if dfs(mask & ~v_bit, size):
                return True

            return False

        return dfs(all_mask, 0)

    # Search for larger n
    for n in range(40, best_result['n'], -1):
        if time.time() > deadline - 10:
            break
        
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        restarts = 0
        while time.time() < deadline - 5:
            restarts += 1
            if restarts > 20000: # Cap restarts per n to avoid infinite loop if time check fails
                break

            adj = [0] * n
            order = pairs[:]
            random.shuffle(order)

            # Greedy construction of maximal triangle-free graph
            for u, v in order:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Fast pre-check
            ga = greedy_alpha(adj, n)
            if ga >= 10:
                continue

            # Exact check
            if not has_independent_set_10(adj, n):
                best_result = {'n': n, 'edges': edges_from_adj(adj, n)}
                break # Found valid graph for this n, move to next smaller n if any (though we go down, so this is success)
                # Actually, we want MAX n. So if we find one at n=40, we are done?
                # The loop is `for n in range(40, ...)` so first success is highest n.
                return best_result

    return best_result
