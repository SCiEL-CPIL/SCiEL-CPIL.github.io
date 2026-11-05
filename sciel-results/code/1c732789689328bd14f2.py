import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    num_sums = target_max + 1  # 257
    
    # Precompute delta vectors for self-pairs
    delta_self = np.zeros((129, num_sums), dtype=np.int32)
    for x in range(129):
        s = 2 * x
        if target_min <= s <= target_max:
            delta_self[x, s] = 1
    
    def calculate_N(A_mask):
        """Calculate N vector: N[s] = count of unordered pairs (a,b) in A with a+b=s."""
        N = np.zeros(num_sums, dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        m = len(A_indices)
        if m == 0:
            return N
        i_idx, j_idx = np.triu_indices(m)
        sums = A_indices[i_idx] + A_indices[j_idx]
        valid = (sums >= 0) & (sums < num_sums)
        if np.any(valid):
            np.add.at(N, sums[valid], 1)
        return N
    
    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = int(np.max(target_N))
        sum_sq = int(np.sum(target_N**2))
        return (0, peak, sum_sq)
    
    def structural_init():
        """Initialize using structural insights: need elements near 64 to cover middle sums."""
        A_mask = np.zeros(129, dtype=bool)
        
        # Start with a base set that covers the target range
        # To cover sum=128: need pairs like (0,128), (1,127), ..., (64,64)
        # To cover sum=256: need (128,128)
        # Strategy: include elements from ~32 to ~128
        
        # Base: include a spread of elements
        base = set()
        # Include high elements to cover large sums
        for x in range(96, 129):
            base.add(x)
        # Include medium elements
        for x in range(48, 96, 2):
            base.add(x)
        # Include some low elements
        for x in range(0, 48, 8):
            base.add(x)
        
        for x in base:
            A_mask[x] = True
        
        # Repair: add elements to cover missing sums
        N = calculate_N(A_mask)
        
        for _ in range(100):  # Max repair iterations
            missing_sums = [s for s in range(target_min, target_max+1) if N[s] == 0]
            if not missing_sums:
                break
            
            A_indices = np.where(A_mask)[0]
            candidates = [x for x in range(129) if not A_mask[x]]
            
            best_x = None
            best_cover = 0
            
            for x in candidates:
                covered = 0
                for s in missing_sums:
                    y = s - x
                    if 0 <= y <= 128 and A_mask[y]:
                        covered += 1
                    if 2*x == s:
                        covered += 1
                if covered > best_cover:
                    best_cover = covered
                    best_x = x
            
            if best_x is None or best_cover == 0:
                break
            
            A_mask[best_x] = True
            N = calculate_N(A_mask)
        
        return A_mask
    
    def simulated_annealing(A_mask, time_limit):
        """Simulated annealing with swap moves."""
        A_mask = A_mask.copy()
        N = calculate_N(A_mask)
        
        A_indices = list(np.where(A_mask)[0])
        not_A_indices = [x for x in range(129) if not A_mask[x]]
        
        current_obj = get_objective(N)
        best_obj = current_obj
        best_A = A_mask.copy()
        best_N = N.copy()
        
        # Temperature schedule
        T0 = 10.0
        Tend = 0.1
        # We'll decay T based on time remaining
        total_steps_estimate = 5000
        step = 0
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < time_limit:
            step += 1
            # Decay temperature
            progress = min(1.0, step / total_steps_estimate)
            T = T0 * (Tend / T0) ** progress
            
            # Choose a random swap
            if not A_indices or not not_A_indices:
                break
                
            a = random.choice(A_indices)
            b = random.choice(not_A_indices)
            
            # Compute new N
            # Remove a, add b
            # New pairs involving b: (b, y) for y in A\{a}, and (b,b)
            # Old pairs involving a: (a, y) for y in A, and (a,a)
            
            # Efficient update:
            # N_new[s] = N[s] - count_pairs_with_a(s) + count_pairs_with_b(s)
            # where count_pairs_with_x(s) counts unordered pairs (x,y) in A with x+y=s,
            # but we need to be careful about double counting.
            
            # Actually, let's just recalculate N for simplicity in SA (it's fast enough)
            # Or do incremental:
            
            A_indices.remove(a)
            A_mask[a] = False
            
            # Update N: remove all pairs involving a
            for y in A_indices:
                s = a + y
                if 0 <= s < num_sums:
                    N[s] -= 1
            # Remove self-pair (a,a)
            s_aa = 2 * a
            if 0 <= s_aa < num_sums:
                N[s_aa] -= 1
            
            # Add b
            A_mask[b] = True
            A_indices.append(b)
            
            # Update N: add all pairs involving b
            for y in A_indices:
                if y != b:
                    s = b + y
                    if 0 <= s < num_sums:
                        N[s] += 1
            # Add self-pair (b,b)
            s_bb = 2 * b
            if 0 <= s_bb < num_sums:
                N[s_bb] += 1
            
            new_obj = get_objective(N)
            
            # Accept or reject
            accept = False
            if new_obj < best_obj:
                accept = True
            elif new_obj[0] == 0 and best_obj[0] == 0:
                # Both valid, compare peak then sum_sq
                if new_obj[1] < best_obj[1]:
                    accept = True
                elif new_obj[1] == best_obj[1] and new_obj[2] < best_obj[2]:
                    accept = True
                else:
                    # Metropolis criterion
                    delta = (new_obj[1] - best_obj[1]) * 1000 + (new_obj[2] - best_obj[2])
                    if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                        accept = True
            elif new_obj[0] == 0 and best_obj[0] > 0:
                accept = True  # Improved from invalid to valid
            elif new_obj[0] > 0 and best_obj[0] == 0:
                # Got worse (valid to invalid), only accept with low probability
                if random.random() < np.exp(-10 / max(T, 1e-10)):
                    accept = True
            
            if accept:
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = A_mask.copy()
                    best_N = N.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                A_mask[b] = False
                A_mask[a] = True
                A_indices.remove(b)
                A_indices.append(a)
                
                # Revert N: undo the changes
                for y in A_indices:
                    if y != a:
                        s = b + y
                        if 0 <= s < num_sums:
                            N[s] -= 1
                s_bb = 2 * b
                if 0 <= s_bb < num_sums:
                    N[s_bb] -= 1
                
                for y in A_indices:
                    s = a + y
                    if 0 <= s < num_sums:
                        N[s] += 1
                s_aa = 2 * a
                if 0 <= s_aa < num_sums:
                    N[s_aa] += 1
                
                no_improve += 1
            
            if no_improve > max_no_improve:
                break
        
        return best_A, best_N, best_obj
    
    best_peak = float('inf')
    best_A_mask = None
    best_N = None
    best_sum_sq = float('inf')
    
    # Main loop: multiple restarts with SA
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        A_mask = structural_init()
        
        # Optionally add some randomness
        if restart > 0:
            A_indices = list(np.where(A_mask)[0])
            # Randomly swap a few elements
            for _ in range(random.randint(2, 5)):
                if A_indices:
                    a = random.choice(A_indices)
                    candidates = [x for x in range(129) if not A_mask[x]]
                    if candidates:
                        b = random.choice(candidates)
                        A_mask[a] = False
                        A_mask[b] = True
        
        N = calculate_N(A_mask)
        obj = get_objective(N)
        
        # Check if already good
        if obj[0] == 0:
            peak = obj[1]
            sum_sq = obj[2]
            if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
                best_peak = peak
                best_sum_sq = sum_sq
                best_A_mask = A_mask.copy()
                best_N = N.copy()
        
        # Run SA
        time_left = deadline - time.time() - 3
        if time_left > 1:
            A_opt, N_opt, obj_opt = simulated_annealing(A_mask, time_left)
            
            if obj_opt[0] == 0:
                if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                    best_peak = obj_opt[1]
                    best_sum_sq = obj_opt[2]
                    best_A_mask = A_opt.copy()
                    best_N = N_opt.copy()
    
    if best_A_mask is None:
        # Fallback
        A_mask = np.zeros(129, dtype=bool)
        for x in range(64, 129):
            A_mask[x] = True
        N = calculate_N(A_mask)
        obj = get_objective(N)
        if obj[0] == 0:
            best_peak = obj[1]
            best_sum_sq = obj[2]
            best_A_mask = A_mask.copy()
        else:
            return list(range(129))
        
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
