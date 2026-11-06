import time
import random
import numpy as np
import math

def construct():
    n = 256
    # Deadline: finish well before 150s kill. Target 118s.
    deadline = time.time() + 118
    
    # 1. Precompute all 4-term APs
    aps_list = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps_list.append((p0, p1, p2, p3))
    
    aps_arr = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps_arr)
    
    # Precompute for greedy polish later
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append(idx)
        pos_aps[p1].append(idx)
        pos_aps[p2].append(idx)
        pos_aps[p3].append(idx)

    def count_mono_numpy(color):
        """Fast count using NumPy."""
        c = color[aps_arr]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def delta_flip_greedy(color, p):
        """
        Calculate change in mono count if we flip position p.
        Used for final polishing.
        """
        # This is slower than the delta in the original but simpler to implement correctly 
        # if we don't maintain mono state. 
        # For 256 elements, checking ~100 APs per flip is fine for a few hundred steps.
        # But we can be smarter:
        # An AP is mono if all 4 are same.
        # If we flip p, only APs containing p change.
        # Let's just use the numpy count for the polish too, it's robust.
        # Or implement a fast delta.
        
        # Fast delta implementation
        new_val = 1 - color[p]
        delta = 0
        for idx in pos_aps[p]:
            p0, p1, p2, p3 = aps_arr[idx]
            # Get current colors
            c0 = color[p0]
            c1 = color[p1]
            c2 = color[p2]
            c3 = color[p3]
            
            was_mono = (c0 == c1 == c2 == c3)
            
            # Determine new colors
            if p0 == p: nc0 = new_val
            else: nc0 = c0
            if p1 == p: nc1 = new_val
            else: nc1 = c1
            if p2 == p: nc2 = new_val
            else: nc2 = c2
            if p3 == p: nc3 = new_val
            else: nc3 = c3
            
            is_mono = (nc0 == nc1 == nc2 == nc3)
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        return delta

    def apply_flip(color, p):
        color[p] = 1 - color[p]

    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        
    def random_init(n):
        return np.random.randint(0, 2, size=n).astype(np.int8)
        
    def rademacher_init(n):
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
        
    def walsh_hadamard_init(n):
        result = []
        for i in range(n):
            rev = int(bin(i)[2:].zfill(8)[::-1], 2)
            result.append(bin(rev).count('1') % 2)
        return np.array(result, dtype=np.int8)

    def run_sa(color_init, time_limit, seed):
        """
        Vectorized Simulated Annealing with Batch Flips.
        """
        np.random.seed(seed)
        color = color_init.copy()
        current_energy = count_mono_numpy(color)
        best_energy = current_energy
        best_color = color.copy()
        
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        start_time = time.time()
        
        # Track stagnation
        no_improve_steps = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit and T > min_T:
            # Determine batch size k
            # Small k for fine tuning, larger k for exploration
            # Use a distribution that favors small k but allows larger
            if random.random() < 0.7:
                k = random.randint(1, 5)
            else:
                k = random.randint(5, 15)
                
            # Generate candidate
            if k == 1:
                p = np.random.randint(0, n)
                candidate = color.copy()
                candidate[p] = 1 - candidate[p]
            else:
                indices = np.random.choice(n, k, replace=False)
                candidate = color.copy()
                candidate[indices] = 1 - candidate[indices]
                
            new_energy = count_mono_numpy(candidate)
            delta_e = new_energy - current_energy
            
            # Metropolis acceptance
            if delta_e < 0:
                color = candidate
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_color = color.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Accept with probability exp(-delta_e / T)
                if random.random() < math.exp(-delta_e / T):
                    color = candidate
                    current_energy = new_energy
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
                    
            T *= cooling
            
            # If stuck, kick harder
            if no_improve_steps > max_no_improve:
                # Perform a large random kick
                kick_size = random.randint(10, 30)
                kick_indices = np.random.choice(n, kick_size, replace=False)
                color[kick_indices] = 1 - color[kick_indices]
                current_energy = count_mono_numpy(color)
                no_improve_steps = 0
                # Reheat slightly
                T = max(T, 1.0)
                
        return best_color, best_energy

    # Main Search Logic
    random.seed(42)
    np.random.seed(42)
    
    best_color_global = None
    best_score_global = float('inf')
    
    initializers = [
        thue_morse,
        random_init,
        rademacher_init,
        walsh_hadamard_init,
        lambda: 1 - thue_morse(n),
    ]
    
    # Allocate time to SA runs
    # We have ~100s for SA.
    # Let's do 5 runs of 20s each, or dynamic allocation.
    # Dynamic: run SA until time is low, then polish.
    
    sa_time_budget = 95.0
    sa_start = time.time()
    
    # Run multiple independent SA chains
    num_sa_runs = 5
    time_per_sa = sa_time_budget / num_sa_runs
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 20:
            break
            
        init_func = initializers[i % len(initializers)]
        init_color = init_func(n)
        
        # Run SA with specific time limit
        run_limit = time_per_sa - (time.time() - sa_start) - (num_sa_runs - i - 1) * 0.5 # reserve a bit for next
        if run_limit < 2:
            run_limit = 2
            
        c, e = run_sa(init_color, run_limit, seed=1000 + i)
        
        if e < best_score_global:
            best_score_global = e
            best_color_global = c
            
    # Final Polish: Greedy Descent
    if best_color_global is not None:
        polish_start = time.time()
        polish_deadline = deadline - 5
        color = best_color_global.copy()
        current_score = best_score_global
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            # Try all single flips
            for p in range(n):
                if time.time() > polish_deadline:
                    break
                d = delta_flip_greedy(color, p)
                if d < 0:
                    apply_flip(color, p)
                    current_score += d
                    improved = True
                    if current_score < best_score_global:
                        best_score_global = current_score
                        best_color_global = color.copy()
                        
        # If time remains, try 2-flips on the best solution
        if time.time() < polish_deadline - 2:
            improved = True
            while improved and time.time() < polish_deadline - 1:
                improved = False
                for p1 in range(n):
                    if time.time() > polish_deadline - 1:
                        break
                    for p2 in range(p1 + 1, n):
                        if time.time() > polish_deadline - 1:
                            break
                        # Calculate delta for 2 flips
                        # This is expensive in Python loops, so we limit this phase
                        # Or just skip if too slow. 
                        # Given n=256, 2-flips is ~32k checks. 
                        # Let's just do a few random 2-flips if time permits.
                        pass # Skipping exhaustive 2-flip to avoid timeout risk, greedy 1-flip is usually sufficient for final polish.
                        
    return list(best_color_global)

# Fallback if best_color_global is None (should not happen)
if 'best_color_global' not in dir() or best_color_global is None:
    def construct():
        return [0]*256
