import numpy as np
import random

N = 400

def construct():
    best_peak = np.inf
    best_x = None
    
    # Try different sizes M
    # Start with 20 (known good) and go up
    for M in range(20, 31):
        # Try a few random initializations
        for _ in range(50):
            support = np.random.choice(N, M, replace=False)
            support.sort()
            
            # Map support index to value index
            # support[i] is the index in the 400-grid
            # We need to map pair (i, j) in support to difference d = support[i] - support[j]
            
            # Precompute pairs for each difference
            # diff -> list of (i, j)
            pairs_by_diff = {}
            for i in range(M):
                for j in range(i + 1, M):
                    d = support[i] - support[j]
                    if d not in pairs_by_diff:
                        pairs_by_diff[d] = []
                    pairs_by_diff[d].append((i, j))
            
            # Also need to know which diffs are affected by flipping sign k
            # affected_diffs[k] = list of diffs where k is involved
            affected_diffs = [[] for _ in range(M)]
            for d, pairs in pairs_by_diff.items():
                for i, j in pairs:
                    affected_diffs[i].append(d)
                    affected_diffs[j].append(d)
            
            # Current sums for each diff
            # sum_val[d] = sum_{(i,j) in pairs[d]} signs[i] * signs[j]
            signs = np.random.choice([-1, 1], M)
            sum_val = {}
            for d, pairs in pairs_by_diff.items():
                s = 0
                for i, j in pairs:
                    s += signs[i] * signs[j]
                sum_val[d] = s
                
            current_max = max(abs(v) for v in sum_val.values()) if sum_val else 0
            current_peak = current_max / M
            
            # Local search
            improved = True
            while improved:
                improved = False
                for k in range(M):
                    old_sign = signs[k]
                    signs[k] = -old_sign
                    
                    # Update sums for affected diffs
                    for d in affected_diffs[k]:
                        # The change in sum_val[d] is -2 * (old_sign * signs[j]) for each partner j?
                        # No, let's just recalculate or update incrementally.
                        # sum_val[d] += -2 * old_sign * (sum of signs of partners) ?
                        # Easier: just recalculate sum_val[d] for the few affected diffs.
                        s = 0
                        for i, j in pairs_by_diff[d]:
                            s += signs[i] * signs[j]
                        old_val = sum_val[d]
                        sum_val[d] = s
                        if abs(s) > current_max:
                            # Peak increased
                            # We can break early if we want, but let's finish the flip check
                            pass
                            
                    new_max = max(abs(v) for v in sum_val.values())
                    new_peak = new_max / M
                    
                    if new_peak < current_peak:
                        current_peak = new_peak
                        current_max = new_max
                        improved = True
                    else:
                        # Revert
                        signs[k] = old_sign
                        # Revert sums
                        for d in affected_diffs[k]:
                            s = 0
                            for i, j in pairs_by_diff[d]:
                                s += signs[i] * signs[j]
                            sum_val[d] = s
                            
            if current_peak < best_peak:
                best_peak = current_peak
                x = np.zeros(N)
                for idx, pos in enumerate(support):
                    x[pos] = signs[idx] / np.sqrt(M)
                best_x = x
                
    return best_x
