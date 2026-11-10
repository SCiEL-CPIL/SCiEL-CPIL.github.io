import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 128
    targets = list(range(128, 257))
    
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

    def get_metrics(counts):
        c = counts[128:257]
        peak = int(np.max(c))
        sq = int(np.sum(c**2))
        covered = bool(np.all(c > 0))
        return peak, sq, covered

    def soft_peak(counts, beta=0.5):
        """Smooth surrogate for peak using log-sum-exp."""
        c = counts[128:257].astype(float)
        # log-sum-exp for stability
        m = np.max(c)
        if m == 0:
            return 0.0
        lse = m + np.log(np.sum(np.exp(c - m)))
        # Scale to approximate peak
        return lse / beta

    def greedy_init():
        A = set()
        for _ in range(10):
            A.add(random.randint(0, n))
        
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        while not covered:
            uncovered = [t for t in targets if counts[t] == 0]
            if not uncovered:
                break
            
            best_x = -1
            best_gain = -1
            
            for x in range(n + 1):
                if x in A:
                    continue
                new_sums = set()
                for a in A:
                    s = x + a
                    if 128 <= s <= 256:
                        new_sums.add(s)
                s = 2 * x
                if 128 <= s <= 256:
                    new_sums.add(s)
                
                gain = len([s for s in new_sums if counts[s] == 0])
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x == -1 or best_gain == 0:
                break
            
            A.add(best_x)
            counts = compute_counts(A)
            peak, sq, covered = get_metrics(counts)
            
        return A

    def spectral_init():
        N = 129
        adj = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: continue
                s = i + j
                if 128 <= s <= 256:
                    adj[i][j] = 1.0
        
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            principal_vec = np.abs(eigvecs[:, -1])
        except:
            principal_vec = np.ones(N)
            
        k = 50
        indices = np.argsort(-principal_vec)
        A = set(indices[:k].tolist())
        
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        if not covered:
            while not covered:
                uncovered = [t for t in targets if counts[t] == 0]
                if not uncovered: break
                best_x = -1
                best_gain = -1
                for x in range(n + 1):
                    if x in A: continue
                    new_sums = set()
                    for a in A:
                        s = x + a
                        if 128 <= s <= 256: new_sums.add(s)
                    s = 2*x
                    if 128 <= s <= 256: new_sums.add(s)
                    gain = len([s for s in new_sums if counts[s] == 0])
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain == 0: break
                A.add(best_x)
                counts = compute_counts(A)
                peak, sq, covered = get_metrics(counts)
                
        return A

    # 1. Initialization
    A = greedy_init()
    counts = compute_counts(A)
    curr_peak, curr_sq, curr_covered = get_metrics(counts)
    
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    A2 = spectral_init()
    counts2 = compute_counts(A2)
    p2, s2, c2 = get_metrics(counts2)
    if c2 and (p2 < best_peak or (p2 == best_peak and s2 < best_sq)):
        A = A2
        counts = counts2
        curr_peak, curr_sq, curr_covered = p2, s2, c2
        best_peak, best_sq = p2, s2
        best_A = set(A)

    # 2. Local Search with incremental updates
    no_improve_count = 0
    max_no_improve = 150
    beta = 0.3  # Temperature for soft peak

    def try_move(A_set, counts, x_remove, x_add):
        """Try removing x_remove and adding x_add. Returns (new_A, new_counts, improved)."""
        if x_remove is None and x_add is None:
            return A_set, counts, False
        if x_remove is None:
            # Add only
            if x_add in A_set:
                return A_set, counts, False
            new_A = set(A_set)
            new_A.add(x_add)
        elif x_add is None:
            # Remove only
            if x_remove not in A_set:
                return A_set, counts, False
            new_A = set(A_set)
            new_A.discard(x_remove)
        else:
            # Swap
            if x_remove not in A_set or x_add in A_set:
                return A_set, counts, False
            new_A = set(A_set)
            new_A.discard(x_remove)
            new_A.add(x_add)
        
        new_counts = compute_counts(new_A)
        p, s, cov = get_metrics(new_counts)
        if not cov:
            return new_A, new_counts, False
        
        # Use soft peak for comparison when peaks are equal
        if p < curr_peak:
            return new_A, new_counts, True
        elif p == curr_peak:
            # Compare soft peak for finer distinction
            soft_curr = soft_peak(counts, beta)
            soft_new = soft_peak(new_counts, beta)
            if soft_new < soft_curr or (abs(soft_new - soft_curr) < 1e-6 and s < curr_sq):
                return new_A, new_counts, True
        
        return new_A, new_counts, False

    while time.time() - start_time < time_limit:
        improved = False
        
        # Try Removals
        A_list = list(A)
        random.shuffle(A_list)
        for x in A_list:
            new_A, new_counts, is_improved = try_move(A, counts, x, None)
            if is_improved:
                A = new_A
                counts = new_counts
                curr_peak, curr_sq, curr_covered = get_metrics(counts)
                if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                    best_peak = curr_peak
                    best_sq = curr_sq
                    best_A = set(A)
                improved = True
                no_improve_count = 0
                break
        if improved: continue

        # Try Swaps (1 in, 1 out)
        A_list = list(A)
        C_list = [i for i in range(n + 1) if i not in A]
        random.shuffle(A_list)
        random.shuffle(C_list)
        
        for x in A_list:
            for y in C_list:
                new_A, new_counts, is_improved = try_move(A, counts, x, y)
                if is_improved:
                    A = new_A
                    counts = new_counts
                    curr_peak, curr_sq, curr_covered = get_metrics(counts)
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                        best_peak = curr_peak
                        best_sq = curr_sq
                        best_A = set(A)
                    improved = True
                    no_improve_count = 0
                    break
            if improved: break
        if improved: continue

        # Try 2-Swaps when stuck
        if no_improve_count > 30:
            A_list = list(A)
            C_list = [i for i in range(n + 1) if i not in A]
            
            num_attempts = min(200, max(50, len(A) * len(C_list) // 10))
            for _ in range(num_attempts):
                if len(A) < 2 or len(C_list) < 2: break
                x1, x2 = random.sample(A_list, 2)
                y1, y2 = random.sample(C_list, 2)
                
                A_trial = set(A)
                A_trial.discard(x1)
                A_trial.discard(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                counts_trial = compute_counts(A_trial)
                p, s, cov = get_metrics(counts_trial)
                if cov:
                    if p < curr_peak:
                        A = A_trial
                        counts = counts_trial
                        curr_peak, curr_sq, curr_covered = p, s, cov
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                            best_peak = curr_peak
                            best_sq = curr_sq
                            best_A = set(A)
                        improved = True
                        no_improve_count = 0
                        break
                    elif p == curr_peak:
                        soft_curr = soft_peak(counts, beta)
                        soft_new = soft_peak(counts_trial, beta)
                        if soft_new < soft_curr - 1e-6 or (abs(soft_new - soft_curr) < 1e-6 and s < curr_sq):
                            A = A_trial
                            counts = counts_trial
                            curr_peak, curr_sq, curr_covered = p, s, cov
                            if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                                best_peak = curr_peak
                                best_sq = curr_sq
                                best_A = set(A)
                            improved = True
                            no_improve_count = 0
                            break
                        
        if not improved:
            no_improve_count += 1
            
            # Perturbation
            if no_improve_count > max_no_improve:
                kick_size = min(4, len(A))
                if len(A) >= kick_size:
                    A_list = list(A)
                    C_list = [i for i in range(n + 1) if i not in A]
                    if len(C_list) >= kick_size:
                        found_valid = False
                        for _ in range(30):
                            x_out = random.sample(A_list, kick_size)
                            y_in = random.sample(C_list, kick_size)
                            A_trial = set(A)
                            for x in x_out:
                                A_trial.discard(x)
                            for y in y_in:
                                A_trial.add(y)
                            
                            counts_trial = compute_counts(A_trial)
                            p, s, cov = get_metrics(counts_trial)
                            if cov:
                                A = A_trial
                                counts = counts_trial
                                curr_peak, curr_sq, curr_covered = p, s, cov
                                found_valid = True
                                break
                        
                        if found_valid:
                            no_improve_count = 0
                        else:
                            A = set(best_A)
                            counts = compute_counts(A)
                            curr_peak, curr_sq, curr_covered = best_peak, best_sq, True
                            no_improve_count = 0

    return sorted(list(best_A))
