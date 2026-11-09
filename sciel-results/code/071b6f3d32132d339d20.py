    def compute_threshold_fast(a_map, b_map):
        # Precompute a_trans for all masks
        # mask -> next_mask after applying a
        a_trans = [0] * (1 << N)
        for mask in range(1 << N):
            next_mask = 0
            for i in range(N):
                if mask & (1 << i):
                    next_mask |= (1 << a_map[i])
            a_trans[mask] = next_mask
        
        start = (1 << N) - 1
        visited = [False] * (1 << N)
        queue = deque([(start, 0)])
        visited[start] = True
        
        while queue:
            state, dist = queue.popleft()
            
            if state & (state - 1) == 0:
                return dist
            
            # a transition
            next_a = a_trans[state]
            if not visited[next_a]:
                visited[next_a] = True
                queue.append((next_a, dist + 1))
                
            # b transition
            next_b = 0
            for i in range(N):
                if state & (1 << i):
                    next_b |= (1 << b_map[i])
            
            if not visited[next_b]:
                visited[next_b] = True
                queue.append((next_b, dist + 1))
                
        return -1
    