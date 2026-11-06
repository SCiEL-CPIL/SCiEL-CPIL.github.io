    def run_heuristic(sort_key_func, randomize_ties=False):
        box_indices = list(range(n))
        
        if randomize_ties:
            # Calculate keys
            keys = [sort_key_func(i) for i in range(n)]
            # Group indices by key
            from collections import defaultdict
            groups = defaultdict(list)
            for i in range(n):
                groups[keys[i]].append(i)
            
            # Shuffle within each group
            for k in groups:
                random.shuffle(groups[k])
            
            # Reconstruct sorted list
            # Sort groups by key
            sorted_keys = sorted(groups.keys(), reverse=True)
            box_indices = []
            for k in sorted_keys:
                box_indices.extend(groups[k])
        else:
            box_indices.sort(key=sort_key_func)
        # ... rest of function
