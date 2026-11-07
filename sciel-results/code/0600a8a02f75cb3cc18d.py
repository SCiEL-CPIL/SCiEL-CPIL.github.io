# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

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
    
    def is_clique(mask):
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            others = mask ^ lsb
            if others & ~adj[u]:
                return False
        return True

    def is_independent(mask):
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            others = mask ^ lsb
            if others & adj[u]:
                return False
        return True

    valid_masks = set()
    for mask in range(1, full_mask + 1):
        if is_clique(mask) or is_independent(mask):
            valid_masks.add(mask)
            
    # dp[mask] = min parts
    # best_sub[mask] = the submask chosen to achieve dp[mask]
    dp = [float('inf')] * (1 << n)
    best_sub = [0] * (1 << n)
    dp[0] = 0
    
    # Iterate masks in increasing order of bit count? 
    # Actually, dp[mask] depends on dp[mask ^ sub], which has fewer bits.
    # So iterating mask from 1 to full_mask works if we ensure sub is a proper submask?
    # No, sub can be any submask. mask ^ sub has fewer bits than mask if sub is not empty.
    # So if we iterate mask from 1 to full_mask, dp[mask ^ sub] is already computed?
    # Yes, because mask ^ sub < mask is not necessarily true in integer value, 
    # but popcount(mask ^ sub) < popcount(mask).
    # However, simple integer iteration 1..2^n does not guarantee topological order by popcount.
    # Example: mask=3 (011), sub=1 (001), rest=2 (010). 2 < 3. OK.
    # mask=5 (101), sub=1 (001), rest=4 (100). 4 < 5. OK.
    # Is it always true that if S is a non-empty submask of M, then M^S < M?
    # Let M have bits. S has a subset of bits.
    # Let k be the highest bit where M and S differ? No, S is subset of M.
    # So M = S | (M^S). S and M^S are disjoint.
    # M = S + (M^S).
    # Since S > 0, M^S < M?
    # Yes, because the highest set bit of M is in either S or M^S.
    # If it's in M^S, then M^S has that bit, and S doesn't. So M^S >= 2^k. M = M^S + S.
    # Wait. If highest bit of M is in S, then M^S < 2^k. S >= 2^k. M = S + M^S.
    # Then M^S < M is true.
    # If highest bit of M is in M^S, then M^S >= 2^k. S < 2^k.
    # M = S + M^S. M^S < M is true.
    # So yes, M^S < M always holds for non-empty proper submask S?
    # Actually, if S=M, M^S=0 < M.
    # So iterating mask from 1 to full_mask is a valid topological order.
    
    for mask in range(1, full_mask + 1):
        # Iterate submasks of mask
        sub = mask
        while sub:
            if sub in valid_masks:
                rest = mask ^ sub
                val = 1 + dp[rest]
                if val < dp[mask]:
                    dp[mask] = val
                    best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    min_parts = dp[full_mask]
    
    # Reconstruct
    partition = []
    mask = full_mask
    while mask:
        sub = best_sub[mask]
        if sub == 0:
            # Should not happen if dp is correct
            break
        vertices = [i for i in range(n) if (sub >> i) & 1]
        partition.append(vertices)
        mask ^= sub
        
    return partition

# EVOLVE-BLOCK-END
