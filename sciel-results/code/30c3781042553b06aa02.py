import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1
    
    # Smooth cost function using log-sum-exp for peak approximation
    def get_cost(sums, smooth_beta=50.0):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1].astype(float)
        
        # Penalty for gaps
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 10000 + gaps * 100 + np.max(target_sums)
            
        # Smooth peak using log-sum-exp
        scaled = target_sums * smooth_beta
        max_val = np.max(scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        smooth_peak = log_sum_exp / smooth_beta
        
        # Secondary: sum of squares to flatten
        sq = np.sum(target_sums**2)
        
        return smooth_peak * 1000 + sq

    # Hard cost for final evaluation
    def get_hard_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX+1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 10000 + gaps * 100 + np.max(target_sums)
        peak = np.max(target_sums)
        sq = np.sum(target_sums**2)
        return peak * 1000 + sq

    # Compute sums from a set A
    def compute_sums(A):
        sums = np.zeros(257, dtype=np.int32)
        A_list = sorted(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        return sums

    # Helper to update sums for adding/removing x
    def apply_add(A, sums, x):
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        if 2*x <= 256:
            sums[2*x] += 1
        A.add(x)
        
    def apply_remove(A, sums, x):
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        if 2*x <= 256:
            sums[2*x] -= 1
        A.remove(x)

    # Gerchberg-Saxton style frequency domain refinement
    # We model x as binary vector of length N+1=129
    # Autocorrelation R[k] = sum_i x[i]*x[i+k] for k in [0, 256]
    # We want R[k] to be flat (approximately constant) for k in [128, 256]
    def gs_refine(A, max_iters=200, alpha=0.5):
        """
        Gerchberg-Saxton: alternate between 
        1. Binary projection (threshold FFT result)
        2. Autocorrelation magnitude projection (adjust FFT magnitudes)
        """
        x = np.zeros(N+1, dtype=np.float64)
        for a in A:
            x[a] = 1.0
        
        # Target: we want autocorrelation to be as flat as possible
        # The mean autocorrelation over target range gives us target value
        current_sums = compute_sums(A)
        target_sums = current_sums[TARGET_MIN:TARGET_MAX+1]
        mean_target = np.mean(target_sums)
        target_val = max(mean_target, 1.0)  # At least 1
        
        best_A = set(A)
        best_cost = get_cost(current_sums)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute FFT
            X = np.fft.fft(x, n=512)  # Pad to avoid circular correlation issues
            
            # Current autocorrelation (linear, not circular)
            # R[k] = sum_i x[i]*x[i+k]
            # In frequency: |X|^2 corresponds to circular autocorrelation
            # For linear, we need to be careful with padding
            
            # Compute linear autocorrelation
            R = np.zeros(2*(N+1), dtype=np.float64)
            for k in range(-(N), N+1):
                if k >= 0:
                    R[N+k] = np.dot(x[:N+1-k], x[k:])
                else:
                    R[N+k] = np.dot(x[-k:], x[:N+1+k])
            
            # Extract target range [128, 256]
            R_target = R[N+TARGET_MIN : N+TARGET_MAX+1]
            
            # Compute discrepancy
            discrepancy = np.sum((R_target - target_val)**2)
            
            # Adjust FFT magnitudes to reduce discrepancy
            # This is the key GS step: modify |X| to make autocorrelation flatter
            # Simple approach: scale magnitudes where autocorrelation is too high/low
            mag = np.abs(X)
            phase = np.angle(X)
            
            # Compute current circular autocorrelation from FFT
            circ_R = np.fft.ifft(np.abs(X)**2).real
            circ_R_target = circ_R[N+TARGET_MIN : N+TARGET_MAX+1]
            
            # Compute correction factor
            if np.max(circ_R_target) > 0:
                # Scale down where too high, scale up where too low
                correction = 1.0 + alpha * (target_val - circ_R_target) / np.max(circ_R_target)
                correction = np.clip(correction, 0.5, 1.5)
                
                # Apply correction to magnitudes (approximate)
                # This is heuristic - adjust overall magnitude based on mean discrepancy
                mean_disc = np.mean(circ_R_target - target_val)
                if mean_disc > 0:
                    # Autocorrelation too high, reduce
                    mag *= (1.0 - alpha * 0.1)
                else:
                    # Autocorrelation too low, increase
                    mag *= (1.0 + alpha * 0.1)
            
            # Reconstruct
            X_new = mag * np.exp(1j * phase)
            x_new = np.fft.ifft(X_new, n=512)[:N+1].real
            
            # Binary projection: threshold
            # Keep top k elements where k = len(A)
            k = len(A)
            if k > 0:
                indices = np.argsort(x_new)[::-1]
                x_binary = np.zeros(N+1, dtype=np.float64)
                x_binary[indices[:k]] = 1.0
            
            A_new = set(np.where(x_binary > 0.5)[0].tolist())
            
            if len(A_new) == 0:
                continue
                
            sums_new = compute_sums(A_new)
            cost_new = get_cost(sums_new)
            
            if cost_new < best_cost:
                best_cost = cost_new
                best_A = A_new
                x = x_binary
                A = A_new
        
        return best_A, best_cost

    # Polish: exhaustive local search to refine
    def polish(A, sums, max_iters=100):
        """Greedy local search: try all single moves, keep best improvement each pass"""
        improved = True
        iters = 0
        current_cost = get_hard_cost(sums)
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_move = None
            best_new_cost = current_cost
            best_new_A = None
            best_new_sums = None
            
            # Try removing each element
            A_list = list(A)
            for x in A_list:
                if len(A) <= 15:
                    break
                apply_remove(A, sums, x)
                new_cost = get_hard_cost(sums)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_move = ('remove', x)
                    best_new_A = set(A)
                    best_new_sums = sums.copy()
                apply_add(A, sums, x)
            
            # Try adding each element
            if best_new_cost >= current_cost:
                candidates = [x for x in range(N+1) if x not in A]
                for x in candidates:
                    apply_add(A, sums, x)
                    new_cost = get_hard_cost(sums)
                    if new_cost < best_new_cost:
                        best_new_cost = new_cost
                        best_move = ('add', x)
                        best_new_A = set(A)
                        best_new_sums = sums.copy()
                    apply_remove(A, sums, x)
            
            # Try swaps (limited to save time)
            if best_new_cost >= current_cost and iters < max_iters // 2:
                A_list = list(A)
                random.shuffle(A_list)
                for x in A_list[:20]:  # Limit swaps
                    candidates = [y for y in range(N+1) if y not in A]
                    random.shuffle(candidates)
                    for y in candidates[:30]:  # Limit candidates
                        apply_remove(A, sums, x)
                        apply_add(A, sums, y)
                        new_cost = get_hard_cost(sums)
                        if new_cost < best_new_cost:
                            best_new_cost = new_cost
                            best_move = ('swap', x, y)
                            best_new_A = set(A)
                            best_new_sums = sums.copy()
                        apply_remove(A, sums, y)
                        apply_add(A, sums, x)
            
            if best_move is not None:
                A = best_new_A
                sums = best_new_sums
                current_cost = best_new_cost
                improved = True
                
        return A, sums, current_cost

    # Initialize best solution
    best_A = None
    best_cost = float('inf')
    best_sums = None
    
    # Try multiple random initializations
    num_restarts = 12
    
    for init_idx in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random set of size ~20-40
        size = random.randint(20, 40)
        A = set(random.sample(range(N+1), size))
        
        sums = compute_sums(A)
        current_cost = get_cost(sums)
        
        # Simulated Annealing with smooth cost
        T = 30.0
        T_min = 0.5
        cooling = 0.997
        steps = 15000
        
        current_A = A
        current_sums = sums
        current_cost = get_cost(current_sums)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(current_A)
            best_sums = current_sums.copy()
            
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Early termination if we found a good solution
            hard_cost = get_hard_cost(current_sums)
            if hard_cost < 1000:
                break

            # Choose move: 10% Add, 10% Remove, 80% Swap
            move_type = random.random()
            
            if move_type < 0.1:
                candidates = [x for x in range(N+1) if x not in current_A]
                if not candidates: continue
                x = random.choice(candidates)
                
                apply_add(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_remove(current_A, current_sums, x)
                    
            elif move_type < 0.2:
                if len(current_A) < 15: continue
                x = random.choice(list(current_A))
                
                apply_remove(current_A, current_sums, x)
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_add(current_A, current_sums, x)
                    
            else:
                if len(current_A) == 0: continue
                x = random.choice(list(current_A))
                candidates = [y for y in range(N+1) if y not in current_A]
                if not candidates: continue
                y = random.choice(candidates)
                
                apply_remove(current_A, current_sums, x)
                apply_add(current_A, current_sums, y)
                
                new_cost = get_cost(current_sums)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(current_A)
                        best_sums = current_sums.copy()
                else:
                    apply_remove(current_A, current_sums, y)
                    apply_add(current_A, current_sums, x)
            
            T *= cooling
            if T < T_min:
                break
        
        # Apply GS refinement if time permits
        if best_A is not None and time.time() < deadline - 15:
            gs_A, gs_cost = gs_refine(best_A, max_iters=100)
            if gs_cost < best_cost:
                best_cost = gs_cost
                best_A = gs_A
                best_sums = compute_sums(best_A)

    # Final polish
    if best_A is not None and time.time() < deadline - 3:
        polished_A, polished_sums, polished_cost = polish(best_A, best_sums.copy(), max_iters=50)
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_A = polished_A
            best_sums = polished_sums

    if best_A is None:
        return list(range(64, 129))
        
    return sorted(best_A)
