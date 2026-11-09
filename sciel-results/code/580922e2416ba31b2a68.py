import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    # Precompute all 60 distinct Hamiltonian cycles on 6 labeled vertices
    # for final verification
    _cycles_cache = None

    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            # Generate unique cycles up to rotation and reflection
            # Fix vertex 0 at start. Permute 1..5.
            # 5! = 120 permutations. / 2 for reflection = 60 cycles.
            for perm in __import__('itertools').permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache

    def _count_paths4(c, i, adj_bm, v):
        """Count number of simple paths of length 4 from vertex i in color c among vertices 0..v-1."""
        count = 0
        mask_i = adj_bm[c][i]
        if mask_i == 0:
            return 0
        a_mask = mask_i
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            b_mask = adj_bm[c][a] & ~(1 << i)
            b_mask_iter = b_mask
            while b_mask_iter:
                b_bit = b_mask_iter & (-b_mask_iter)
                b = b_bit.bit_length() - 1
                b_mask_iter ^= b_bit
                e_mask = adj_bm[c][b] & ~(1 << i) & ~(1 << a)
                e_mask_iter = e_mask
                while e_mask_iter:
                    e_bit = e_mask_iter & (-e_mask_iter)
                    e = e_bit.bit_length() - 1
                    e_mask_iter ^= e_bit
                    d_mask = adj_bm[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                    count += bin(d_mask).count('1')
        return count

    def _verify(colors, n):
        """Verify no color class contains a C6 using bitmask adjacency."""
        if n < 6:
            return True
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1

        adj_bitmasks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)

        cycles = _get_cycles()

        for c in range(4):
            for subset in combinations(range(n), 6):
                subset_set = set(subset)
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if not (adj_bitmasks[c][u] & (1 << w)):
                            ok = False
                            break
                    if ok:
                        return False
        return True

    def _precompute_path4_bitmask(v, adj_bitmasks):
        """
        Compute path4_mask[c][i] for subgraph on vertices 0..v-1 using bitmasks.
        adj_bitmasks[c][i] is a bitmask of neighbors of i in color c.
        path4_mask[c][i] is a bitmask where bit d is set if there is a 
        simple path of length 4 from i to d in color c graph.
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                mask_i = adj_bitmasks[c][i]
                if mask_i == 0:
                    continue
                
                mask = 0
                a_mask = mask_i
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    b_mask = adj_bitmasks[c][a] & ~(1 << i)
                    if b_mask == 0:
                        continue
                    
                    b_mask_iter = b_mask
                    while b_mask_iter:
                        b_bit = b_mask_iter & (-b_mask_iter)
                        b = b_bit.bit_length() - 1
                        b_mask_iter ^= b_bit
                        
                        e_mask = adj_bitmasks[c][b] & ~(1 << i) & ~(1 << a)
                        if e_mask == 0:
                            continue
                        
                        e_mask_iter = e_mask
                        while e_mask_iter:
                            e_bit = e_mask_iter & (-e_mask_iter)
                            e = e_bit.bit_length() - 1
                            e_mask_iter ^= e_bit
                            
                            d_mask = adj_bitmasks[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                            mask |= d_mask
                
                path4_mask[c][i] = mask
        
        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_bitmasks = [[0] * n_target for _ in range(4)]
        node_count = [0]
        
        def _update_bitmasks(i, j, c, add):
            if add:
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
            else:
                adj_bitmasks[c][i] &= ~(1 << j)
                adj_bitmasks[c][j] &= ~(1 << i)

        def _assign_vertex(v, path4_mask, rng):
            """Assign colors to edges (v, 0), ..., (v, v-1) with constraint checking."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            masks = [0, 0, 0, 0]
            unassigned = set(range(v))
            
            def _backtrack_dynamic():
                if time.time() > deadline:
                    return False

                if not unassigned:
                    return True
                
                # Pick the vertex with the fewest valid colors
                best_u = -1
                best_count = 5
                best_colors = []
                
                for u in unassigned:
                    valid_colors = []
                    for c in range(4):
                        if path4_mask[c][u] & masks[c]:
                            continue
                        valid_colors.append(c)
                    if len(valid_colors) < best_count:
                        best_count = len(valid_colors)
                        best_u = u
                        best_colors = valid_colors
                        if best_count == 0:
                            return False
                        if best_count == 1:
                            break
                
                if best_u == -1:
                    return False

                # Shuffle colors to use randomness
                rng.shuffle(best_colors)

                for c in best_colors:
                    unassigned.remove(best_u)
                    masks[c] |= (1 << best_u)
                    adj[best_u][v] = c
                    adj[v][best_u] = c
                    _update_bitmasks(best_u, v, c, True)

                    if _backtrack_dynamic():
                        return True

                    _update_bitmasks(best_u, v, c, False)
                    masks[c] &= ~(1 << best_u)
                    adj[best_u][v] = -1
                    adj[v][best_u] = -1
                    unassigned.add(best_u)

                return False

            return _backtrack_dynamic()

        # Try to build the coloring vertex by vertex
        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            
            # Precompute Path4 for the current subgraph on vertices 0..v-1
            path4_mask = _precompute_path4_bitmask(v, adj_bitmasks)
            
            if not _assign_vertex(v, path4_mask, rng):
                return None

        # Extract colors in lexicographic order
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return colors

    def _local_search(colors, n, max_iters, deadline):
        """
        Local search polish: try to reduce C6 count by flipping edge colors.
        Returns improved colors or None if no improvement found.
        """
        if time.time() > deadline:
            return None
        
        local_rng = random.Random(42)
        
        # Build adjacency from colors
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        
        adj_bitmasks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
        
        def _count_c6s():
            """Count total C6s across all colors using bitmasks."""
            total = 0
            cycles = _get_cycles()
            for c in range(4):
                for subset in combinations(range(n), 6):
                    for cycle in cycles:
                        ok = True
                        for k in range(6):
                            u = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if not (adj_bitmasks[c][u] & (1 << w)):
                                ok = False
                                break
                        if ok:
                            total += 1
            return total
        
        current_c6s = _count_c6s()
        if current_c6s == 0:
            return colors
        
        best_colors = colors[:]
        best_c6s = current_c6s
        
        # Precompute edge index to (i,j) mapping
        edge_map = []
        idx2 = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx2 += 1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            e = local_rng.randint(0, n*(n-1)//2 - 1)
            i, j = edge_map[e]
            old_c = adj[i][j]
            new_c = (old_c + 1) % 4
            
            # Flip
            adj[i][j] = new_c
            adj[j][i] = new_c
            adj_bitmasks[old_c][i] &= ~(1 << j)
            adj_bitmasks[old_c][j] &= ~(1 << i)
            adj_bitmasks[new_c][i] |= (1 << j)
            adj_bitmasks[new_c][j] |= (1 << i)
            
            new_c6s = _count_c6s()
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = []
                for ii in range(n):
                    for jj in range(ii + 1, n):
                        best_colors.append(adj[ii][jj])
            else:
                # Revert
                adj[i][j] = old_c
                adj[j][i] = old_c
                adj_bitmasks[new_c][i] &= ~(1 << j)
                adj_bitmasks[new_c][j] &= ~(1 << i)
                adj_bitmasks[old_c][i] |= (1 << j)
                adj_bitmasks[old_c][j] |= (1 << i)
        
        return best_colors if best_c6s < current_c6s else None

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    # Use more seeds for smaller n since search is faster
    seed_counts = {17: 100, 16: 50, 15: 50, 14: 50, 13: 50, 12: 50, 11: 50, 10: 50}
    
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 15:
            break

        found = False
        max_seeds = seed_counts.get(n, 50)
        # Try multiple seeds
        for seed in range(max_seeds):
            if time.time() > deadline - 15:
                break

            colors = _search(n, seed, deadline - 15)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    # Try local search to fix it
                    if time.time() < deadline - 5:
                        polished = _local_search(colors, n, 2000, deadline - 5)
                        if polished is not None and _verify(polished, n):
                            best_n = n
                            best_colors = polished
                            found = True
                            break

        if found:
            break

    if best_n == 0:
        # Fallback to a known small valid coloring
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        
        if _verify(colors, n):
            best_n = n
            best_colors = colors
        else:
            n = 5
            best_n = n
            best_colors = [0] * (n*(n-1)//2)

    return {'n': best_n, 'colors': best_colors}
