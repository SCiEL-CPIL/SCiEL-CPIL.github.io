import time
import sys

def construct(P=None):
    start_time = time.time()
    deadline = start_time + 55 # 5s margin

    # Robust P discovery
    if P is None:
        # Check globals
        if 'P' in globals():
            candidate = globals()['P']
            if isinstance(candidate, dict) and 'edge_order' in candidate:
                P = candidate
        
        # If still None, try to find in module globals explicitly
        if P is None:
            try:
                import __main__
                if hasattr(__main__, 'P'):
                    candidate = __main__.P
                    if isinstance(candidate, dict) and 'edge_order' in candidate:
                        P = candidate
            except:
                pass

    if P is None:
        # Last resort: scan all loaded modules for a dict with 'edge_order'
        for mod_name, mod in sys.modules.items():
            if mod_name in ('__main__', 'construct', 'seed'): # Avoid self-referential loops if possible, though checking all is safer
                pass
            try:
                for name, val in vars(mod).items():
                    if isinstance(val, dict) and 'edge_order' in val:
                        P = val
                        break
            except:
                continue
            if P is not None:
                break

    if P is None:
        raise NameError("Could not find data dictionary P")

    edge_order = P['edge_order']
    n = 18
    full_mask = (1 << n) - 1
    
    # Precompute masks for all triples
    masks = []
    for triple in edge_order:
        m = 0
        for v in triple:
            m |= (1 << v)
        masks.append(m)
        
    # Precompute vertex to indices mapping for faster DFS
    # vertex_to_indices[v] = list of indices i where v is in edge_order[i]
    vertex_to_indices = {v: [] for v in range(n)}
    for i, triple in enumerate(edge_order):
        for v in triple:
            vertex_to_indices[v].append(i)

    def exists_matching(K):
        """
        Returns True if a perfect matching can be formed using only triples with index < K.
        """
        if K == 0:
            return False
            
        def dfs(used_mask, count, start_idx):
            if time.time() > deadline:
                return False
            if count == 6:
                return used_mask == full_mask
                
            remaining_mask = full_mask ^ used_mask
            if remaining_mask == 0:
                return count == 6
                
            # Get lowest vertex
            lowest_v = (remaining_mask & -remaining_mask).bit_length() - 1
            
            # Try all triples i that contain lowest_v
            # We must pick i >= start_idx to ensure we don't reuse indices and maintain order
            # and i < K to respect the prefix limit
            candidates = vertex_to_indices[lowest_v]
            for i in candidates:
                if i < start_idx:
                    continue
                if i >= K:
                    break # Since candidates are sorted (indices added in order), we can break
                m = masks[i]
                if m & used_mask:
                    continue
                if dfs(used_mask | m, count + 1, i + 1):
                    return True
            return False
            
        return dfs(0, 0, 0)

    # Binary search for the minimum K
    lo, hi = 1, len(edge_order)
    best_K = None
    
    # Quick check if full set works
    if not exists_matching(len(edge_order)):
        return []

    while lo <= hi:
        mid = (lo + hi) // 2
        if exists_matching(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_K is None:
        return []
        
    # Reconstruct the matching
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
            
            candidates = vertex_to_indices[lowest_v]
            for i in candidates:
                if i < start_idx:
                    continue
                if i >= K:
                    break
                m = masks[i]
                if m & used_mask:
                    continue
                if dfs(used_mask | m, count + 1, i + 1, current_triples + [edge_order[i]]):
                    return True
            return False
            
        return dfs(0, 0, 0, [])
        
    if not find_matching(best_K):
        # Fallback to full range if something went wrong (shouldn't happen if exists_matching worked)
        if not find_matching(len(edge_order)):
            return []

    final_matching = []
    for triple in result_triples:
        final_triple = [int(v) for v in triple]
        final_matching.append(final_triple)
        
    return final_matching
