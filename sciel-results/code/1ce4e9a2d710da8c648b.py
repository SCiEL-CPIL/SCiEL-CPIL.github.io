import sys

sys.setrecursionlimit(10000)

def construct():
    N = 16
    return _odd_merge(N)

def _odd_merge(n):
    if n <= 1:
        return []
    
    # Split indices into evens and odds
    evens = [i for i in range(0, n, 2)]
    odds = [i for i in range(1, n, 2)]
    
    # Recursively sort the evens and odds
    net_evens = _odd_merge(len(evens))
    net_odds = _odd_merge(len(odds))
    
    # Map the recursive networks back to global indices
    mapped_evens = [(evens[i], evens[j]) for i, j in net_evens]
    mapped_odds = [(odds[i], odds[j]) for i, j in net_odds]
    
    # Merge step
    # For two sorted lists of size m = n//2
    m = n // 2
    merge_comparators = []
    
    # The merge of two sorted lists L and R of size m:
    # 1. Split L into L_even, L_odd and R into R_even, R_odd
    # 2. Recursively merge L_even and R_even
    # 3. Recursively merge L_odd and R_odd
    # 4. Compare L_odd[k] with R_even[k] for k=0..m/2-1
    
    # However, we are working with global indices.
    # L is the list of 'evens' indices. R is the list of 'odds' indices.
    
    # We need a helper to generate the merge network for two specific lists of indices.
    # But the structure is the same as _odd_merge, just the "sort" part is replaced by "merge".
    
    # Actually, the merge step itself is an _odd_merge-like operation on the pairs.
    
    # Let's just generate the comparators directly based on the pattern.
    # For k in range(m // 2):
    #    # Compare L[2*k] with R[2*k]? No.
    
    # Correct pattern for merge of size m:
    # It is equivalent to _odd_merge(m) but with specific index mapping?
    # No.
    
    # Let's use the recursive definition of the merge.
    def merge_network(list_a, list_b):
        # list_a and list_b are lists of indices
        if len(list_a) <= 1:
            return [(list_a[0], list_b[0])] if len(list_a) == 1 else []
        
        a_even = list_a[0::2]
        a_odd = list_a[1::2]
        b_even = list_b[0::2]
        b_odd = list_b[1::2]
        
        net = merge_network(a_even, b_even) + merge_network(a_odd, b_odd)
        
        # Final comparisons
        # Compare a_odd[k] with b_even[k]
        for k in range(len(a_odd)):
            net.append((a_odd[k], b_even[k]))
            
        return net

    merge_comps = merge_network(evens, odds)
    
    return mapped_evens + mapped_odds + merge_comps

