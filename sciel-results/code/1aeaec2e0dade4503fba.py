    def solve_csp(K, time_limit=100):
        target_size = K + 1
        target_subsets = list(itertools.combinations(points, target_size))
        subset_to_idx = {s: i for i, s in enumerate(target_subsets)}
        
        triple_covers = {t: [] for t in all_triples}
        for s in target_subsets:
            idx = subset_to_idx[s]
            for t in itertools.combinations(s, 3):
                triple_covers[t].append(idx)
                
        subset_to_triples = {s: [] for s in target_subsets}
        for s in target_subsets:
            for t in itertools.combinations(s, 3):
                subset_to_triples[s].append(t)
                
        best_solution = None
        start_time = time.time()
        
        def backtrack(edges, covered_mask):
            nonlocal best_solution, start_time
            
            if time.time() - start_time > time_limit:
                return False
            
            if not any(not c for c in covered_mask):
                best_solution = edges[:]
                return True
                
            min_available = float('inf')
            chosen_S = None
            chosen_triples = None
            
            for i, s in enumerate(target_subsets):
                if not covered_mask[i]:
                    available = [t for t in subset_to_triples[s] if is_linear(t, edges)]
                    if len(available) < min_available:
                        min_available = len(available)
                        chosen_S = s
                        chosen_triples = available
                        if min_available == 0:
                            return False
                        if min_available == 1:
                            break
            
            if chosen_S is None:
                return True
            
            random.shuffle(chosen_triples)
            
            for t in chosen_triples:
                new_covered = covered_mask[:]
                for idx in triple_covers[t]:
                    new_covered[idx] = True
                    
                if backtrack(edges + [t], new_covered):
                    return True
                    
            return False

        initial_covered = [False] * len(target_subsets)
        
        if backtrack([], initial_covered):
            return best_solution
        
        return None
