    def check(m, L):
        # Calculate candidate indices for each i
        # Use local variables for speed
        cand = [None] * 13
        for i in range(1, 13):
            start_k = (-1 - m) % i
            if start_k >= L:
                return False
            cand[i] = range(start_k, L, i) # range object is lazy
            
        # Sort indices by number of candidates
        # We need len, so we must materialize or calculate len
        # len = (L - 1 - start_k) // i + 1
        lengths = [0] * 13
        for i in range(1, 13):
            start_k = (-1 - m) % i
            if start_k < L:
                lengths[i] = (L - 1 - start_k) // i + 1
            else:
                lengths[i] = 0
                
        if any(lengths[i] == 0 for i in range(1, 13)):
            return False
            
        indices = sorted(range(1, 13), key=lambda x: lengths[x])
        
        # DFS with used array
        used = [False] * L
        
        # Unroll or optimize DFS?
        # Python recursion is slow. Let's try to use a stack?
        # Or just rely on the fact that N=12 is small.
        
        def dfs(idx):
            if idx == 12:
                return True
            i = indices[idx]
            for k in range(start_k_i, L, i): # Need start_k_i
                if not used[k]:
                    used[k] = True
                    if dfs(idx + 1):
                        return True
                    used[k] = False
            return False
