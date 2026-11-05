    def _build_circulant_bits(n, jumps):
        adj_bits = [0] * n
        for i in range(n):
            mask = 0
            for j in jumps:
                v2 = (i + j) % n
                if v2 != i:
                    mask |= (1 << v2)
            adj_bits[i] = mask
        return adj_bits

    # Search for circulant graphs
    for n in range(40, 34, -1):
        if time.time() > deadline:
            break
            
        max_jump = n // 2
        best_n = 0
        best_edges = []
        found = [False]
        
        def backtrack(current_list, current_set, next_jump):
            if found[0]:
                return
            
            if n / (2 * len(current_list) + 1) >= 10:
                return

            adj_bits = _build_circulant_bits(n, current_list)
            alpha = _max_independent_set_size_bitset(adj_bits, n)
            
            if alpha < 10:
                found[0] = True
                nonlocal best_n, best_edges
                best_n = n
                best_edges = _get_edges_from_adj_bits(adj_bits, n)
                return
                
            for j in range(next_jump, max_jump + 1):
                can_add = True
                for x in current_list:
                    diff = abs(x - j)
                    if diff in current_set:
                        can_add = False
                        break
                        
                if can_add:
                    current_list.append(j)
                    current_set.add(j)
                    backtrack(current_list, current_set, j + 1)
                    current_list.pop()
                    current_set.remove(j)
                    
                    if found[0]:
                        return

        backtrack([], set(), 1)
        
        if found[0]:
            return {'n': best_n, 'edges': best_edges}
