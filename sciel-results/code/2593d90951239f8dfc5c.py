import numpy as np

N = 400

def construct():
        # 1. Select Support
        # Aim for M around 20-30.
        M = 25
        
        # Greedy selection to minimize difference collisions
        # We want to pick indices such that differences are as unique as possible.
        # Start with a random set, then improve.
        
        # Initialize with random permutation
        support = np.random.permutation(N)[:M]
        
        # Heuristic improvement: Swap indices to reduce max collision count
        # Calculate difference counts
        def get_diff_counts(supp):
            diffs = []
            for i in range(len(supp)):
                for j in range(i+1, len(supp)):
                    diffs.append(abs(supp[i] - supp[j]))
            # Count occurrences of each diff
            counts = np.bincount(diffs, minlength=N)
            return counts

        counts = get_diff_counts(support)
        max_count = np.max(counts)
        
        # Try to reduce max_count by swapping
        for _ in range(50): # Few iterations
            if max_count <= 1: break
            # Find a diff with max_count
            bad_diffs = np.where(counts == max_count)[0]
            bad_diff = bad_diffs[0]
            
            # Find a pair in support that creates this diff
            # This is expensive to do naively, but M is small.
            # Actually, let's just do a random swap and accept if it improves max_count
            # or total collision score.
            
            # Randomly pick two indices in support to swap
            i1, i2 = np.random.choice(M, 2, replace=False)
            old_i1, old_i2 = support[i1], support[i2]
            
            # Pick new random indices not in support
            remaining = np.setdiff1d(np.arange(N), support)
            new_i1, new_i2 = np.random.choice(remaining, 2, replace=False)
            
            support[i1] = new_i1
            support[i2] = new_i2
            
            new_counts = get_diff_counts(support)
            new_max = np.max(new_counts)
            # Also consider total squared collisions to handle distribution
            score_old = np.sum(counts**2)
            score_new = np.sum(new_counts**2)
            
            if new_max < max_count or (new_max == max_count and score_new < score_old):
                counts = new_counts
                max_count = new_max
            else:
                # Revert
                support[i1] = old_i1
                support[i2] = old_i2

        # 2. Assign Signs via Local Search
        x = np.zeros(N)
        # Initialize signs randomly
        signs = np.random.choice([-1.0, 1.0], size=M)
        
        # Map support to indices for easy access
        # support is sorted? No, keep as is.
        
        def calc_peak(signs, support):
            # Calculate convolution peaks
            # y_d = sum s_i s_{i+d}
            # We only care about d where both i, i+d in support
            # Efficiently:
            # For each pair (i, j) in support, d = |i-j|
            # Add s_i s_j to Y_d
            Y = np.zeros(N)
            for k in range(M):
                for l in range(k+1, M):
                    d = abs(support[k] - support[l])
                    Y[d] += signs[k] * signs[l]
            return np.max(np.abs(Y))

        current_peak = calc_peak(signs, support)
        
        # Local Search: Flip a sign if it reduces the peak
        for _ in range(100):
            improved = False
            for k in range(M):
                old_sign = signs[k]
                signs[k] = -old_sign
                new_peak = calc_peak(signs, support)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    signs[k] = old_sign
            if not improved:
                break

        # 3. Construct Final Array
        # Scale by 1/sqrt(M) to keep L2 norm 1
        # The seed used 1/sqrt(N). 
        # If the metric is "upper bound on C3", it might be normalized differently.
        # Let's stick to 1/sqrt(N) scaling to be comparable to seed, 
        # OR use 1/sqrt(M) if we want to maximize amplitude for the same L2.
        # The task says "improve an autoconvolution upper bound".
        # Usually, bounds are tighter for smaller L2 norms or specific normalizations.
        # If the bound is on C3 = ||x*x||_inf / ||x||_2^2 or similar, scaling doesn't matter.
        # If it's just ||x*x||_inf, smaller is better.
        # Seed used 1/sqrt(N). 
        # Let's use 1/sqrt(N) to be safe and comparable.
        
        x[support] = signs / np.sqrt(N)
        
        return x
