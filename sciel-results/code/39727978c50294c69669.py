import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin

    n = 128
    targets = list(range(128, 257))
    target_mask = np.zeros(257, dtype=bool)
    target_mask[128:257] = True

    def compute_counts(A_set):
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 256:
                    counts[s] += 1
        return counts

    def get_peak_and_sq(counts):
        c = counts[128:257]
        peak = np.max(c)
        sq = np.sum(c**2)
        return peak, sq

    def is_covered(counts):
        return np.all(counts[128:257] > 0)

    def spectral_init():
        # Construct a random regular graph on 129 vertices [0, 128]
        # and use eigenvector centrality to select vertices.
        # The "sum-difference" constraint suggests edges between i and j if i+j is in [128, 256].
        # Let's build an adjacency matrix where A[i][j] = 1 if i+j in [128, 256] and i != j.
        # This graph captures the "validity" of pairs for our target interval.
        # We want a set A such that the convolution (A+A) is flat.
        # High eigenvector centrality in this graph might correlate with "central" elements
        # that participate in many valid sums.
        
        N = 129
        # Adjacency matrix
        adj = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: continue
                s = i + j
                if 128 <= s <= 256:
                    adj[i][j] = 1.0
        
        # To make it regular-ish or just use the spectral properties of this specific graph
        # The graph is bipartite-ish? No, i+j even/odd.
        # Let's just compute the principal eigenvector.
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            # Principal eigenvector is the last one
            principal_vec = eigvecs[:, -1]
            # Take absolute values to handle sign
            scores = np.abs(principal_vec)
        except:
            scores = np.ones(N)
            
        # Normalize scores to sum to 1? Or just rank them.
        # We want a set of size k.
        # Start with k=64? Or estimate based on density.
        # Full set [0, 128] has size 129.
        # Greedy started with ~30-40 elements.
        # Let's pick the top 64 elements by score.
        top_k = 64
        indices = np.argsort(-scores)
        A = set(indices[:top_k].tolist())
        
        # Ensure coverage. If not, add random elements or greedy fill.
        counts = compute_counts(A)
        if not is_covered(counts):
            # Fallback to greedy fill
            uncovered = set(targets)
            while uncovered and len(A) < N:
                best_x = -1
                best_new_cov = 0
                for x in range(N):
                    if x in A: continue
                    new_sums = set()
                    for a in A:
                        new_sums.add(x + a)
                    new_sums.add(2*x)
                    cov = len(new_sums & uncovered)
                    if cov > best_new_cov:
                        best_new_cov = cov
                        best_x = x
                if best_x == -1 or best_new_cov == 0:
                    break
                A.add(best_x)
                new_sums = set()
                for a in A:
                    new_sums.add(best_x + a)
                new_sums.add(2*best_x)
                uncovered -= new_sums
                
        return A

    # 1. Initialization
    # Try spectral init, and also a standard greedy, pick the better one?
    # Or just use spectral init as it's a structural change.
    A = spectral_init()
    
    counts = compute_counts(A)
    curr_peak, curr_sq = get_peak_and_sq(counts)
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    # If spectral init failed coverage (shouldn't with greedy fill), fallback to full
    if not is_covered(counts):
        A = set(range(n + 1))
        counts = compute_counts(A)
        curr_peak, curr_sq = get_peak_and_sq(counts)
        best_peak = curr_peak
        best_sq = curr_sq
        best_A = set(A)

    # 2. Local Search with Bottleneck Focus
    # We will perform local search to minimize peak, then sq.
    # Key improvement: Efficient delta updates and focusing on bottleneck sums.
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Identify bottleneck sums (highest counts)
        # c = counts[128:257]
        # bottleneck_sums = np.where(c == curr_peak)[0] + 128
        
        # 1. Try Removals
        A_snapshot = list(A)
        random.shuffle(A_snapshot)
        
        best_rem_peak = curr_peak
        best_rem_sq = curr_sq
        best_rem_x = None
        best_rem_counts = None
        
        for x in A_snapshot:
            # Simulate removing x
            new_counts = counts.copy()
            
            # Decrement pairs involving x
            for z in A:
                if z == x:
                    continue
                s = x + z
                if s <= 256:
                    new_counts[s] -= 1
            
            s = 2 * x
            if s <= 256:
                new_counts[s] -= 1
                
            if not is_covered(new_counts):
                continue
                
            peak, sq = get_peak_and_sq(new_counts)
            
            # Accept if peak is lower, or peak same and sq lower
            if peak < best_rem_peak or (peak == best_rem_peak and sq < best_rem_sq):
                best_rem_peak = peak
                best_rem_sq = sq
                best_rem_x = x
                best_rem_counts = new_counts
        
        if best_rem_x is not None:
            A.remove(best_rem_x)
            counts = best_rem_counts
            curr_peak = best_rem_peak
            curr_sq = best_rem_sq
            improved = True
            if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                best_peak = curr_peak
                best_sq = curr_sq
                best_A = set(A)
            continue
        
        # 2. Try Swaps
        A_snapshot = list(A)
        candidates_out = [i for i in range(n + 1) if i not in A]
        
        if not candidates_out:
            break
            
        best_swap_peak = curr_peak
        best_swap_sq = curr_sq
        best_swap_x = None
        best_swap_y = None
        best_swap_counts = None
        
        random.shuffle(A_snapshot)
        random.shuffle(candidates_out)
        
        # Optimization: Limit swap checks if time is running out?
        # For now, check all. 129*129 is ~16k checks. Each check is O(|A|) ~ 60. Total ~1M ops.
        # This is fast enough in Python if we don't do it too many times per second.
        
        for x in A_snapshot:
            for y in candidates_out:
                new_counts = counts.copy()
                
                # Remove x
                for z in A:
                    if z == x:
                        continue
                    s = x + z
                    if s <= 256:
                        new_counts[s] -= 1
                s = 2 * x
                if s <= 256:
                    new_counts[s] -= 1
                    
                # Add y
                for z in A:
                    if z == x:
                        continue
                    s = y + z
                    if s <= 256:
                        new_counts[s] += 1
                s = 2 * y
                if s <= 256:
                    new_counts[s] += 1
                    
                if not is_covered(new_counts):
                    continue
                
                peak, sq = get_peak_and_sq(new_counts)
                
                if peak < best_swap_peak or (peak == best_swap_peak and sq < best_swap_sq):
                    best_swap_peak = peak
                    best_swap_sq = sq
                    best_swap_x = x
                    best_swap_y = y
                    best_swap_counts = new_counts
                    
        if best_swap_x is not None:
            A.remove(best_swap_x)
            A.add(best_swap_y)
            counts = best_swap_counts
            curr_peak = best_swap_peak
            curr_sq = best_swap_sq
            improved = True
            if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                best_peak = curr_peak
                best_sq = curr_sq
                best_A = set(A)
            continue

        # 3. Perturbation to escape local minima
        # Swap 2 random elements
        if len(A) > 2:
            x1, x2 = random.sample(list(A), 2)
            cands = [i for i in range(n + 1) if i not in A]
            if len(cands) >= 2:
                y1, y2 = random.sample(cands, 2)
                
                A_trial = set(A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                new_counts = compute_counts(A_trial)
                
                if is_covered(new_counts):
                    peak, sq = get_peak_and_sq(new_counts)
                    # Accept with probability based on improvement or just accept if valid
                    # to explore.
                    A = A_trial
                    counts = new_counts
                    curr_peak = peak
                    curr_sq = sq
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                        best_peak = curr_peak
                        best_sq = curr_sq
                        best_A = set(A)

    return sorted(list(best_A))
