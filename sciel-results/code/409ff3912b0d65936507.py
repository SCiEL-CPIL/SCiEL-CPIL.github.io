"""Gradient-based optimization for low-discrepancy signs on arithmetic progressions."""
import time
import numpy as np

N = 256
DEADLINE = 75.0  # safety margin before 90s kill

def _precompute_aps():
    """Precompute all arithmetic progressions as index arrays (0-indexed)."""
    aps = []
    # Singletons
    for i in range(N):
        aps.append(np.array([i], dtype=np.int32))
    # Longer APs: start s (0-indexed), step d >= 1, length L >= 2
    for d in range(1, N):
        for s in range(N):
            idx = list(range(s, N, d))
            if len(idx) >= 2:
                aps.append(np.array(idx, dtype=np.int32))
    return aps

def _discretize(x):
    """Round continuous vector to ±1, with 0 mapped to +1."""
    return np.where(x >= 0, 1, -1).astype(np.int8)

def construct():
    t_start = time.time()
    
    # Precompute APs and build indicator matrix
    aps = _precompute_aps()
    n_aps = len(aps)
    
    # Build indicator matrix M: shape (n_aps, N), M[k,i]=1 if pos i in AP k
    M = np.zeros((n_aps, N), dtype=np.float32)
    for k, ap in enumerate(aps):
        M[k, ap] = 1.0
    
    def compute_sums(x):
        """x is array of length N, returns float32 array of AP sums."""
        return M @ x.astype(np.float32)
    
    def evaluate(signs):
        """signs is int8 array of length N, returns max absolute AP sum as int."""
        sums = M @ signs.astype(np.float32)
        return int(np.max(np.abs(sums)))
    
    rng = np.random.RandomState(12345)
    best_signs = None
    best_val = 999
    
    # Precompute position masks for local search
    pos_masks = []
    for i in range(N):
        mask = (M[:, i] != 0).astype(np.int8)
        pos_masks.append(mask)
    
    # === Phase 1: Gradient descent in continuous space ===
    for restart in range(15):
        if time.time() - t_start > DEADLINE - 15:
            break
        
        # Initialize
        if restart == 0:
            x = rng.randn(N).astype(np.float64)
            x /= max(1.0, np.max(np.abs(x)))
        elif best_signs is not None and restart % 2 == 0:
            # Perturb previous best
            x = best_signs.astype(np.float64) + rng.randn(N) * 0.5
        else:
            x = rng.randn(N).astype(np.float64)
            x /= max(1.0, np.max(np.abs(x)))
        
        lr = 0.1
        lr_decay = 0.998
        
        for step in range(8000):
            if time.time() - t_start > DEADLINE - 15:
                break
            
            sums = compute_sums(x)
            abs_s = np.abs(sums)
            excess = np.maximum(0.0, abs_s - 1.0)
            
            if np.sum(excess) < 1e-6:
                break
            
            grad_s = np.where(abs_s > 1.0, 2.0 * excess * np.sign(sums), 0.0)
            grad_x = M.T @ grad_s
            
            x -= lr * grad_x
            x = np.clip(x, -1.0, 1.0)
            
            if step % 20 == 0:
                signs = _discretize(x)
                val = evaluate(signs)
                if val < best_val:
                    best_val = val
                    best_signs = signs.copy()
            
            lr *= lr_decay
        
        # Evaluate final
        signs = _discretize(x)
        val = evaluate(signs)
        if val < best_val:
            best_val = val
            best_signs = signs.copy()
        
        if best_val <= 2:
            break  # Can't do better than 2 (singletons give 1, but maybe 2 is achievable)
    
    if best_signs is None:
        best_signs = _discretize(x)
        best_val = evaluate(best_signs)
    
    # === Phase 2: Local search (flip individual positions) ===
    if time.time() - t_start < DEADLINE - 5:
        current = best_signs.copy()
        current_val = best_val
        sums = compute_sums(current)
        
        improved = True
        while improved and time.time() - t_start < DEADLINE - 3:
            improved = False
            for i in range(N):
                if time.time() - t_start > DEADLINE - 3:
                    break
                old_val_i = current[i]
                mask = pos_masks[i]
                if not mask.any():
                    continue
                new_sums = sums.copy()
                new_sums[mask] -= 2.0 * old_val_i
                new_max = int(np.max(np.abs(new_sums)))
                if new_max < current_val:
                    current[i] = -old_val_i
                    sums = new_sums
                    current_val = new_max
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_signs = current.copy()
    
    # === Phase 3: Random restart local search ===
    if time.time() - t_start < DEADLINE - 5:
        for restart in range(100):
            if time.time() - t_start > DEADLINE - 3:
                break
            
            cand = best_signs.copy()
            n_flips = rng.randint(3, 12)
            flip_idx = rng.choice(N, size=n_flips, replace=False)
            for idx in flip_idx:
                cand[idx] = -cand[idx]
            
            val = evaluate(cand)
            if val > best_val + 3:
                continue
            
            # Local search from cand
            sums = compute_sums(cand)
            cur_val = val
            cur_signs = cand.copy()
            improved2 = True
            while improved2 and time.time() - t_start < DEADLINE - 3:
                improved2 = False
                for i in range(N):
                    if time.time() - t_start > DEADLINE - 3:
                        break
                    old_val_i = cur_signs[i]
                    mask = pos_masks[i]
                    if not mask.any():
                        continue
                    new_sums = sums.copy()
                    new_sums[mask] -= 2.0 * old_val_i
                    new_max = int(np.max(np.abs(new_sums)))
                    if new_max < cur_val:
                        cur_signs[i] = -old_val_i
                        sums = new_sums
                        cur_val = new_max
                        improved2 = True
                        if cur_val < best_val:
                            best_val = cur_val
                            best_signs = cur_signs.copy()
    
    return [int(s) for s in best_signs]
