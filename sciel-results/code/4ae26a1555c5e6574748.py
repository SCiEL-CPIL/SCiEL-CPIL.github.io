    def nondominated_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        S = [set() for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].add(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].add(i)
                    n_dom[i] += 1
        
        current_front = np.where(n_dom == 0)[0]
        rank = 0
        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                ranks[i] = rank
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = np.array(next_front)
            rank += 1
        
        # Calculate crowding distance for each front
        # Group indices by rank
        for r in range(rank):
            front_indices = np.where(ranks == r)[0]
            if len(front_indices) <= 2:
                crowding[front_indices] = np.inf
                continue
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_idx = front_indices[np.argsort(F[front_indices, m])]
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                
                f_min = F[sorted_idx[0], m]
                f_max = F[sorted_idx[-1], m]
                if f_max == f_min:
                    continue
                    
                for k in range(1, len(sorted_idx)-1):
                    idx_curr = sorted_idx[k]
                    idx_prev = sorted_idx[k-1]
                    idx_next = sorted_idx[k+1]
                    
                    crowding[idx_curr] += (F[idx_next, m] - F[idx_prev, m]) / (f_max - f_min)
        
        return ranks, crowding
