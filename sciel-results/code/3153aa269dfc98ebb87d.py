import time

N = 17
DEADLINE = 90

def construct():
    start_time = time.time()
    
    def bitonic_merge(n, k, reverse=False):
        # n: total size
        # k: sequence length
        # reverse: if True, sort in descending order
        comps = []
        if k == 1:
            return comps
        
        # j is the step size
        for j in range(k // 2, 0, -1):
            for i in range(0, n, k):
                for l in range(i, i + j):
                    if l + j < n:
                        # Determine direction
                        # If (l // k) % 2 == 0, then ascending
                        # else descending
                        # But for arbitrary N, the logic is slightly different
                        # Standard Bitonic Sort assumes power of 2
                        # For arbitrary N, we can just generate the comparators
                        # and rely on the min/max to sort
                        comps.append((l, l + j))
                        
        return comps

    def bitonic_sort(n):
        comps = []
        # k is the current sequence length
        k = 2
        while k <= n:
            # j is the current step
            for j in range(k // 2, 0, -1):
                for i in range(0, n, k):
                    for l in range(i, i + j):
                        if l + j < n:
                            comps.append((l, l + j))
            k *= 2
        return comps

    # This simple implementation might not sort correctly for non-power-of-2
    # because the "bitonic" property might be broken.
    
    # Safer: Use Batcher's Odd-Merge with a fixed depth limit or correct base case.
    
    # Let's try a known small network for 17?
    # Or just use the seed (289) if I can't get a better one quickly.
    
    # I will use the 32-wire Bitonic network but map wires 17-31 to wire 16?
    # No, that breaks the network.
    
    # I will return the seed network to be safe, but try to improve if possible.
    # The seed is 289.
    
    # Let's try to generate Batcher's Odd-Merge correctly.
    # I'll use the recursion with a depth limit.
    
    def odd_merge(A, B, depth=0):
        if depth > 20:
            return []
        if not A or not B:
            return []
        
        # Interleave
        X = []
        for i in range(max(len(A), len(B))):
            if i < len(A):
                X.append(A[i])
            if i < len(B):
                X.append(B[i])
            
        comps = []
        # Compare adjacent
        for i in range(0, len(X)-1, 2):
            comps.append((X[i], X[i+1]))
            
        # Split
        U = X[::2]
        V = X[1::2]
        
        # Recurse
        comps.extend(odd_merge(U, V, depth + 1))
        
        return comps

    def batcher_odd_merge(n):
        if n == 1:
            return []
        
        # Split
        n1 = n // 2
        n2 = n - n1
        
        A = list(range(n1))
        B = list(range(n1, n))
        
        # Sort A and B
        comps = []
        if n1 > 1:
            comps.extend(batcher_odd_merge(n1))
        if n2 > 1:
            # Shift B indices
            b_comps = batcher_odd_merge(n2)
            comps.extend([(i + n1, j + n1) for i, j in b_comps])
            
        # Merge
        comps.extend(odd_merge(A, B))
        
        return comps

    comps = batcher_odd_merge(N)
    
    if time.time() - start_time > DEADLINE - 5:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return comps
