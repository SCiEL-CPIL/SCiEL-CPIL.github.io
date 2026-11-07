import time
import sys
import inspect

def construct(P=None):
    start_time = time.time()
    deadline = start_time + 50 # Safety margin

    # Robust P discovery
    if P is None:
        # 1. Check direct global 'P'
        if 'P' in globals():
            candidate = globals()['P']
            if isinstance(candidate, dict) and 'edge_order' in candidate:
                P = candidate
        
        # 2. Scan current module globals
        if P is None:
            for name, val in globals().items():
                if isinstance(val, dict) and 'edge_order' in val:
                    P = val
                    break
        
        # 3. Scan caller's local variables
        if P is None:
            try:
                frame = inspect.currentframe().f_back
                if frame is not None:
                    for name, val in frame.f_locals.items():
                        if isinstance(val, dict) and 'edge_order' in val:
                            P = val
                            break
            except Exception:
                pass

        # 4. Search loaded modules (excluding __main__ direct access to avoid NameError if not present)
        if P is None:
            for mod_name, mod in sys.modules.items():
                if mod_name == '__main__':
                    continue
                try:
                    mod_vars = vars(mod)
                    for name, val in mod_vars.items():
                        if isinstance(val, dict) and 'edge_order' in val:
                            P = val
                            break
                except Exception:
                    continue
                if P is not None:
                    break

    if P is None:
        raise NameError("Could not find data dictionary P")

    edge_order = P['edge_order']
    n = 18 # Task specifies 0..17
    
    # Precompute masks
    masks = []
    for triple in edge_order:
        m = 0
        for v in triple:
            m |= (1 << v)
        masks.append(m)
        
    full_mask = (1 << n) - 1
    
    # Helper to check if a matching exists using first K triples
    def exists_matching(K):
        """
        Returns True if a perfect matching can be formed using only triples with index < K.
        Uses DFS with memoization on (used_mask, start_idx) to avoid re-exploring states.
        Actually, simple DFS with pruning is often faster for small K due to overhead of memoization keys.
        But let's use a set for visited states in this specific K context to prune.
        """
        if K == 0:
            return False
            
        # We need to pick 6 disjoint triples from edge_order[0:K]
        
        def dfs(used_mask, count, start_idx):
            if time.time() > deadline:
                return False
            if count == 6:
                return used_mask == full_mask
                
            # Pruning: if we can't fill remaining slots with remaining triples
            # (This check is weak if K is large, but helps if K is tight)
            
            # Optimization: Find the lowest set bit in (full_mask ^ used_mask)
            # We must cover this vertex.
            remaining_mask = full_mask ^ used_mask
            if remaining_mask == 0:
                return count == 6 # Should have returned above if count==6, but safe
                
            # Get lowest vertex
            lowest_v = (remaining_mask & -remaining_mask).bit_length() - 1
            
            # Try all triples i in [start_idx, K) that contain lowest_v
            # Note: We must ensure we don't pick triples that overlap with used_mask.
            
            for i in range(start_idx, K):
                m = masks[i]
                if m & (1 << lowest_v):
                    if m & used_mask:
                        continue
                    # Try this triple
                    if dfs(used_mask | m, count + 1, i + 1):
                        return True
            return False
            
        return dfs(0, 0, 0)

    # Binary search for the minimum K
    lo, hi = 1, len(edge_order)
    best_K = None
    
    # First, ensure a solution exists in the full list
    if not exists_matching(len(edge_order)):
        # Should not happen per problem statement, but handle gracefully
        # Return any 6 disjoint triples? Or raise?
        # The problem implies a solution exists.
        pass

    while lo <= hi:
        mid = (lo + hi) // 2
        if exists_matching(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_K is None:
        # Fallback: return empty or raise
        return []
        
    # Now we know a solution exists in the first best_K triples.
    # We need to reconstruct the actual matching to return it.
    # We can reuse the DFS logic to find one specific matching.
    
    result_triples = []
    
    def find_matching(K):
        def dfs(used_mask, count, start_idx, current_triples):
            if time.time() > deadline:
                return False
            if count == 6:
                if used_mask == full_mask:
                    result_triples.extend(current_triples)
                    return True
                return False
                
            remaining_mask = full_mask ^ used_mask
            if remaining_mask == 0:
                return count == 6
                
            lowest_v = (remaining_mask & -remaining_mask).bit_length() - 1
            
            for i in range(start_idx, K):
                m = masks[i]
                if m & (1 << lowest_v):
                    if m & used_mask:
                        continue
                    if dfs(used_mask | m, count + 1, i + 1, current_triples + [edge_order[i]]):
                        return True
            return False
            
        return dfs(0, 0, 0, [])
        
    if not find_matching(best_K):
        # If reconstruction fails (shouldn't if exists_matching passed), try full range
        if not find_matching(len(edge_order)):
            return []

    # Convert to required format: list of lists of ints
    final_matching = []
    for triple in result_triples:
        final_triple = [int(v) for v in triple]
        final_matching.append(final_triple)
        
    return final_matching
