    def calc_score(points):
        n = len(points)
        
        partners = [set() for _ in range(n)]
        
        for i, j, k in itertools.combinations(range(n), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                partners[i].add((j, k))
                partners[j].add((i, k))
                partners[k].add((i, j))
                
        order = sorted(range(n), key=lambda v: len(partners[v]), reverse=True)
        
        best_size = 0
        chosen = []
        chosen_pairs = set()
        
        def backtrack(idx, current_size):
            nonlocal best_size
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            if current_size + (n - idx) <= best_size:
                return
                
            v = order[idx]
            
            # Option 1: Include points[v]
            valid_inclusion = True
            for pair in partners[v]:
                if pair in chosen_pairs:
                    valid_inclusion = False
                    break
                    
            if valid_inclusion:
                chosen.append(v)
                new_pairs = set()
                for u in chosen[:-1]: # chosen[:-1] is the previous chosen
                    p = (u, v) if u < v else (v, u)
                    new_pairs.add(p)
                    chosen_pairs.add(p)
                
                backtrack(idx + 1, current_size + 1)
                
                for p in new_pairs:
                    chosen_pairs.remove(p)
                chosen.pop()
            
            # Option 2: Exclude points[v]
            backtrack(idx + 1, current_size)
            
        backtrack(0, 0)
        return best_size
