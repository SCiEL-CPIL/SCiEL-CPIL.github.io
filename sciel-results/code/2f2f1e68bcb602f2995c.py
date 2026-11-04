import numpy as np

N = 400

def construct():
    # Target size for the support.
    # Sidon sets in Z_N have size ~ sqrt(N).
    # Let's aim for M=20 or 21.
    M = 20
    
    # Greedy construction of a Sidon set (B_2 sequence)
    # We want a set S such that all differences s_i - s_j (i != j) are distinct modulo N?
    # Or just distinct integers?
    # "Step function on [-1/4, 1/4]" with 400 intervals.
    # This is a discrete grid.
    # If it's linear convolution, differences are integers.
    # If it's circular, differences are mod N.
    # Let's assume linear first, as "on [-1/4, 1/4]" is a finite interval.
    # However, usually these problems are on the torus.
    # Let's try to minimize collisions in linear differences.
    
    support = []
    diffs = set()
    
    # Start from 0
    support.append(0)
    
    for k in range(1, N):
        # Check if k can be added
        valid = True
        for s in support:
            d1 = k - s
            d2 = s - k
            # For linear Sidon, we just need k-s not to equal any existing diff
            # Actually, if we track all differences s_i - s_j, we need k - s_i != s_j - s_k?
            # No, a Sidon set requires that the sums s_i + s_j are unique.
            # This is equivalent to differences s_i - s_j being unique for i!=j?
            # Not exactly. 
            # If s_i + s_j = s_k + s_l, then s_i - s_k = s_l - s_j.
            # So yes, distinct differences (up to sign) is the condition for B_2.
            
            # We maintain a set of all differences s_a - s_b for a,b in support.
            # If we add k, new diffs are k - s and s - k for all s in support.
            # We must ensure these new diffs are not in the existing set.
            
            for s in support:
                d_pos = k - s
                d_neg = s - k
                if d_pos in diffs or d_neg in diffs:
                    valid = False
                    break
            if not valid:
                continue
            
            # Add k and update diffs
            support.append(k)
            for s in support[:-1]: # Exclude k itself? No, diffs with existing elements
                diffs.add(k - s)
                diffs.add(s - k)
            
            if len(support) == M:
                break
    
    if len(support) < M:
        # If we couldn't find a full Sidon set, fill with random or just use what we have
        # But for now, let's assume we can find 20.
        pass
        
    support = np.array(support)
    
    # Assign signs
    # Since it's a Sidon set, each difference appears at most once (in absolute value? No, signed).
    # The convolution peak will be determined by the sum of signs for pairs with the same difference.
    # In a Sidon set, there are no pairs with the same difference.
    # So Y[d] is either 1, -1, or 0.
    # Max |Y[d]| = 1.
    # Signs don't matter for the peak magnitude, only for the specific values.
    # But we return the function.
    signs = np.ones(len(support))
    
    # Construct x
    # Normalize to L2 = 1
    x = np.zeros(N)
    x[support] = signs / np.sqrt(len(support))
    
    return x
