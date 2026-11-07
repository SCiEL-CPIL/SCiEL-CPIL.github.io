"""Erdos minimum overlap: Binary Block Structure with Simulated Annealing."""
import time
import numpy as np

N = 100
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(h):
    """Compute the objective value: max correlation / N * 2."""
    if not np.all(np.isfinite(h)):
        return np.inf
    # np.correlate(h, 1-h, mode="full")
    # The max overlap is at the center of the full correlation (lag 0) if we consider 
    # standard correlation, but 'full' mode gives lags from -(N-1) to N-1.
    # The problem asks for max(np.correlate(h, 1-h, mode="full")).
    # Note: correlate(a, v) computes sum(a * conj(v)).
    # Here a=h, v=1-h.
    # At lag 0 (center index N-1 in 'full' mode), it is sum(h * (1-h)).
    # But we want the MAX over all lags.
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """Project h onto the affine hyperplane sum(h)=N/2 and box [0,1]."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    
    # Final check and correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    # Strategy: 
    # 1. Generate diverse "blocky" initial candidates. 
    #    These are sequences composed of segments of constant height (often 0 or 1, 
    #    or intermediate values) that sum to N/2.
    # 2. Use Simulated Annealing (SA) to refine these.
    #    SA moves: Flip a bit, Shift a block, Change block height, Swap segments.
    #    This allows escaping local minima that smooth gradient methods might get stuck in.
    
    rng = np.random.RandomState(100)
    best_h = None
    best_score = np.inf
    
    # --- Phase 1: Generate Initial Candidates ---
    candidates = []
    
    # Candidate 1: Alternating 0s and 1s (perfect anti-correlation at lag 0, but high at other lags?)
    # h = [1, 0, 1, 0, ...] sum = 50.
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Candidate 2: Blocks of 1s and 0s
    # e.g., 50 ones, 50 zeros
    h_block = np.zeros(N)
    h_block[:50] = 1.0
    candidates.append(h_block)
    
    # Candidate 3: Random Binary Sequences summing to 50
    for _ in range(20):
        h_bin = np.zeros(N)
        indices = rng.choice(N, size=50, replace=False)
        h_bin[indices] = 1.0
        candidates.append(h_bin)
        
    # Candidate 4: Random "Blocky" Sequences
    # Partition into K blocks, assign random heights in [0,1], then fix sum.
    for K in [5, 10, 20, 50]:
        for _ in range(5):
            block_sizes = np.full(K, N // K)
            remainder = N % K
            for i in range(remainder):
                block_sizes[i] += 1
            h_blocky = np.zeros(N)
            idx = 0
            for size in block_sizes:
                val = rng.uniform(0, 1)
                h_blocky[idx:idx+size] = val
                idx += size
            h_blocky = _fix_sum(h_blocky)
            candidates.append(h_blocky)

    # --- Phase 2: Simulated Annealing ---
    
    def sa_move(h):
        """Perform a random perturbation on h."""
        h_new = h.copy()
        move_type = rng.randint(4)
        
        if move_type == 0:
            # Flip a bit (if binary-ish) or randomize a value
            idx = rng.randint(N)
            if rng.random() < 0.5:
                h_new[idx] = 1.0 if h[idx] < 0.5 else 0.0
            else:
                h_new[idx] = rng.uniform(0, 1)
                
        elif move_type == 1:
            # Swap two values
            i, j = rng.choice(N, size=2, replace=False)
            h_new[i], h_new[j] = h_new[j], h_new[i]
            
        elif move_type == 2:
            # Shift a block
            # Pick a start and length
            start = rng.randint(N)
            length = rng.randint(1, 10)
            end = (start + length) % N
            # Extract block (handling wrap-around)
            if start + length <= N:
                block = h_new[start:start+length].copy()
                # Shift right by 1 (cyclic)
                h_new[start:end] = np.roll(block, 1)
            else:
                # Wrap around
                part1 = h_new[start:].copy()
                part2 = h_new[:end].copy()
                block = np.concatenate([part1, part2])
                h_new[start:] = np.roll(block, 1)
                h_new[:end] = np.roll(block, 1)
                
        else:
            # Local smoothing / blurring
            idx = rng.randint(N)
            delta = rng.uniform(-0.2, 0.2)
            h_new[idx] += delta
            if rng.random() < 0.5 and idx > 0:
                h_new[idx-1] += delta * 0.5
            if rng.random() < 0.5 and idx < N-1:
                h_new[idx+1] += delta * 0.5

    # Run SA for each candidate
    for h_init in candidates:
        if deadline_check():
            break
            
        h = _fix_sum(h_init)
        if not _is_valid(h):
            continue
            
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        # SA parameters
        T = 0.1
        cooling_rate = 0.999
        steps_per_temp = 50
        
        while T > 0.001 and not deadline_check():
            for _ in range(steps_per_temp):
                if deadline_check():
                    break
                    
                h_new = sa_move(h)
                h_new = _fix_sum(h_new)
                
                if not _is_valid(h_new):
                    continue
                    
                new_score = _score(h_new)
                
                # Accept if better, or with probability exp(-delta/T)
                if new_score < current_score:
                    h = h_new
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                else:
                    delta = new_score - current_score
                    if delta < 1e-9:
                        continue
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        h = h_new
                        current_score = new_score
                        
            T *= cooling_rate
            
        # Local refinement (Hill Climbing) after SA
        improved = True
        while improved and not deadline_check():
            improved = False
            for idx in range(N):
                if deadline_check():
                    break
                # Try small adjustments
                for delta in [-0.05, 0.05, -0.01, 0.01]:
                    h_try = h.copy()
                    h_try[idx] += delta
                    h_try = _fix_sum(h_try)
                    if _is_valid(h_try):
                        s_try = _score(h_try)
                        if s_try < current_score - 1e-12:
                            h = h_try
                            current_score = s_try
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_h = h.copy()
                            break # Restart loop for this idx if improved
        
    # Final validation
    if best_h is None or not _is_valid(best_h):
        best_h = np.full(N, 0.5)
        
    return best_h
