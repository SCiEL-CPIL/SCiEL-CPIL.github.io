    # Find a valid default matching using the same backtracking but with max limit
    # This ensures we always have a valid fallback
    def find_any_matching():
        # Use a high limit so it finds something
        limit = max(all_indices) if all_indices else 1
        result = []
        used = set()
        
        def bt_any(used, result):
            if len(result) == 6:
                return True
            unused = [v for v in range(n) if v not in used]
            if not unused:
                return False
            # Pick most constrained
            best_v = -1
            best_count = float('inf')
            best_opts = None
            for v in unused:
                opts = [(idx, fs) for idx, fs in vertex_triples[v] if idx <= limit]
                if len(opts) < best_count:
                    best_count = len(opts)
                    best_v = v
                    best_opts = opts
                if best_count == 0:
                    return False
            if best_count == 0:
                return False
            
            for idx, fs in best_opts:
                if bt_any(used | fs, result + [list(fs)]):
                    return True
            return False
        
        if bt_any(set(), []):
            return result
        return None

    any_match = find_any_matching()
    if any_match:
        default_result = any_match
        default_score = max(triple_to_idx[frozenset(t)] for t in default_result)
    else:
        # Should not happen if a matching exists, but fallback to empty or error?
        # Task implies a solution exists.
        default_result = [] 
        default_score = float('inf')
