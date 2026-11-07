import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if not (adj[v] & lsb_r):
                    ok = False
                    break
                r ^= lsb_r
            is_clique[mask] = ok
        else:
            is_clique[mask] = False
            
        # Check independent
        if is_ind[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                u = lsb_r.bit_length() - 1
                if (adj[v] & lsb_r):
                    ok = False
                    break
                r ^= lsb_r
            is_ind[mask] = ok
        else:
            is_ind[mask] = False

    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            
    # Sort valid masks by size descending
    valid_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute popcount for speed? bit_count is fast.
    
    def solve(rem, parts_left):
        if rem == 0:
            return True
        if parts_left == 0:
            return False
        
        if parts_left == 1:
            return is_clique[rem] or is_ind[rem]
            
        # Pick a vertex v in rem with fewest candidates
        # Candidates for v: S in valid_masks such that (S & v_bit) and (S & rem == S)
        
        best_v_bit = -1
        best_candidates = None
        min_cand_count = float('inf')
        
        m = rem
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            cands = []
            for S in valid_masks:
                if (S & lsb) and (S & rem == S):
                    cands.append(S)
            
            if len(cands) < min_cand_count:
                min_cand_count = len(cands)
                best_v_bit = lsb
                best_candidates = cands
                if min_cand_count == 0:
                    return False
                if min_cand_count == 1:
                    break
                    
        if not best_candidates:
            return False
            
        # Sort candidates by size descending
        best_candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        # Pruning: max size
        max_s = best_candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return False
            
        for S in best_candidates:
            if solve(rem ^ S, parts_left - 1):
                return True
        return False

    best_k = None
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        if solve(full_mask, k):
            best_k = k
            break
            
    if best_k is None:
        # Fallback: greedy or random?
        # Just return a valid partition if we found none in time, though unlikely.
        # For safety, return a trivial partition (each vertex its own part)
        best_k = n
        
    # Reconstruct partition
    def find_partition(rem, parts_left, current_path):
        if rem == 0:
            return current_path
        if parts_left == 0:
            return None
        if parts_left == 1:
            if is_clique[rem] or is_ind[rem]:
                return current_path + [rem]
            return None

        best_v_bit = -1
        best_candidates = None
        min_cand_count = float('inf')
        
        m = rem
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            cands = []
            for S in valid_masks:
                if (S & lsb) and (S & rem == S):
                    cands.append(S)
            
            if len(cands) < min_cand_count:
                min_cand_count = len(cands)
                best_v_bit = lsb
                best_candidates = cands
                if min_cand_count == 0:
                    return None
                if min_cand_count == 1:
                    break
                    
        if not best_candidates:
            return None
            
        best_candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        max_s = best_candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return None
            
        for S in best_candidates:
            res = find_partition(rem ^ S, parts_left - 1, current_path + [S])
            if res:
                return res
        return None

    partition_masks = find_partition(full_mask, best_k, [])
    
    if partition_masks is None:
         # Fallback
         partition_masks = [1 << i for i in range(n)]

    # Convert masks to list of vertex lists
    result = []
    for mask in partition_masks:
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        result.append(verts)
        
    return result

if __name__ == "__main__":
    construct()
