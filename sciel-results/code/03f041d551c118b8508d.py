import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
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
    
    # Helper to get list of bits in mask
    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        return bits

    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # v must be connected to all in rest
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
            # v must NOT be connected to any in rest
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
            
    # Sort valid masks by size descending to try larger sets first
    valid_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute popcount for all masks? Python 3.8+ has bit_count
    # Let's use a lookup table for speed if needed, but bit_count is fast enough.
    
    # B&B Search
    # We want to find min k such that full_mask can be partitioned into k valid masks.
    
    # Lower bound helper
    # We can precompute max valid subset size for each mask?
    # max_size_in_mask[mask] = max(popcount(S) for S in valid_masks if S & mask == S)
    # This is O(2^n * |valid_masks|).
    # |valid_masks| is likely small.
    
    max_size_in_mask = [0] * (1 << n)
    # For each valid mask S, update all supersets? No, that's hard.
    # For each mask M, iterate valid masks S.
    # This is 2^14 * |valid_masks|.
    # Let's just compute it on the fly in the search if it's fast enough, or use a simpler bound.
    # Simpler bound: ceil(popcount(rem) / max_possible_size)
    # max_possible_size is the size of the largest valid subset of rem.
    
    def get_max_valid_size(rem):
        max_s = 0
        for S in valid_masks:
            if S & rem == S:
                s_size = S.bit_count()
                if s_size > max_s:
                    max_s = s_size
        return max_s

    # To speed up get_max_valid_size, we can precompute it.
    # Let's try to precompute it efficiently.
    # max_size_in_mask[0] = 0
    # For mask in 1..2^n-1:
    #   max_size_in_mask[mask] = max(max_size_in_mask[mask ^ lsb], max_valid_subset_containing_lsb_in_mask)
    # This is complex.
    
    # Given n=14, let's just use the on-the-fly check but optimize it.
    # We can filter valid_masks to those that are subsets of rem.
    
    def solve(rem, parts_left):
        if rem == 0:
            return True
        if parts_left == 0:
            return False
        
        # Lower bound
        # If parts_left == 1, check if rem itself is valid
        if parts_left == 1:
            return is_clique[rem] or is_ind[rem]
            
        # Pick a vertex v in rem
        # Use the vertex with the fewest candidates?
        # Or just the lowest bit.
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Candidates: S in valid_masks such that (S & v_bit) and (S & rem == S)
        # To prune, we can check if there exists ANY candidate.
        # Also, we can sort candidates by size desc.
        
        candidates = []
        for S in valid_masks:
            if (S & v_bit) and (S & rem == S):
                candidates.append(S)
                
        if not candidates:
            return False
            
        # Sort by size descending
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        # Additional pruning:
        # If the largest candidate size is S_max, we need at least ceil(popcount(rem)/S_max) parts.
        # If parts_left < ceil(popcount(rem)/S_max), return False.
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return False
            
        for S in candidates:
            if solve(rem ^ S, parts_left - 1):
                return True
        return False

    # Find minimum k
    best_k = None
    for k in range(1, n + 1):
        if solve(full_mask, k):
            best_k = k
            break
            
    if best_k is None:
        best_k = n # Should not happen
        
    # Reconstruct partition
    # We need to find the actual sets.
    def find_partition(rem, parts_left, current_path):
        if rem == 0:
            return current_path
        
        if parts_left == 0:
            return None
            
        if parts_left == 1:
            if is_clique[rem] or is_ind[rem]:
                return current_path + [rem]
            return None

        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        candidates = []
        for S in valid_masks:
            if (S & v_bit) and (S & rem == S):
                candidates.append(S)
                
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return None
            
        for S in candidates:
            res = find_partition(rem ^ S, parts_left - 1, current_path + [S])
            if res:
                return res
        return None

    partition_masks = find_partition(full_mask, best_k, [])
    
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
