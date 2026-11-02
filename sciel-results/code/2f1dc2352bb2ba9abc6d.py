"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U:
        return 0.0
    U_arr = np.array(sorted(U), dtype=np.int64)
    n = len(U_arr)
    max_val = int(U_arr[-1])
    if max_val < 1:
        return 1.0
    
    # Compute sumset U+U
    # Using convolution via FFT for efficiency
    # We need to find all a+b for a,b in U
    # Create a boolean array of length max_val+1
    size = max_val + 1
    # Use FFT-based convolution
    # Pad to power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * size - 1:
        fft_size *= 2
    
    a = np.zeros(fft_size, dtype=np.float64)
    a[U_arr] = 1.0
    
    # FFT convolution for sumset
    fft_a = np.fft.fft(a)
    fft_a_sq = fft_a * fft_a
    conv = np.fft.ifft(fft_a_sq).real
    
    # Sumset: indices where conv > 0.5 (since we're counting pairs)
    sumset_mask = conv > 0.5
    # The sumset ranges from 0 to 2*max_val
    sumset_count = np.sum(sumset_mask[:2*size-1])
    
    # Compute difference set U-U
    # Difference set: all a-b for a,b in U
    # This is equivalent to convolution of a with reversed a
    a_rev = a[::-1]
    fft_a_rev = np.fft.fft(a_rev)
    fft_diff = fft_a * fft_a_rev
    conv_diff = np.fft.ifft(fft_diff).real
    
    # The difference set ranges from -max_val to max_val
    # In the convolution, index i corresponds to value (i - (size-1))
    # So we need indices from (size-1 - max_val) to (size-1 + max_val)
    diff_start = size - 1 - max_val
    diff_end = size - 1 + max_val + 1
    diff_mask = conv_diff[diff_start:diff_end] > 0.5
    diffset_count = np.sum(diff_mask)
    
    if sumset_count <= 0 or diffset_count <= 0:
        return 0.0
    
    ratio = diffset_count / sumset_count
    if ratio <= 0:
        return 0.0
    
    log_denom = math.log(2 * max_val + 1)
    if log_denom <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / log_denom
    return score


def _evaluate_set(U):
    """Evaluate a set, returning (score, U+U size, U-U size, max_val)"""
    U_arr = np.array(sorted(U), dtype=np.int64)
    n = len(U_arr)
    max_val = int(U_arr[-1])
    
    if max_val < 1:
        return 1.0, 1, 1, 0
    
    size = max_val + 1
    fft_size = 1
    while fft_size < 2 * size - 1:
        fft_size *= 2
    
    a = np.zeros(fft_size, dtype=np.float64)
    a[U_arr] = 1.0
    
    fft_a = np.fft.fft(a)
    conv = np.fft.ifft(fft_a * fft_a).real
    sumset_count = np.sum(conv[:2*size-1] > 0.5)
    
    a_rev = a[::-1]
    conv_diff = np.fft.ifft(fft_a * np.fft.fft(a_rev)).real
    diff_start = size - 1 - max_val
    diff_end = size - 1 + max_val + 1
    diffset_count = np.sum(conv_diff[diff_start:diff_end] > 0.5)
    
    if sumset_count <= 0 or diffset_count <= 0:
        return 0.0, 0, 0, max_val
    
    ratio = diffset_count / sumset_count
    log_denom = math.log(2 * max_val + 1)
    score = 1.0 + math.log(ratio) / log_denom if log_denom > 0 else 0.0
    
    return score, int(sumset_count), int(diffset_count), max_val


def _random_set(n_elements, max_val):
    """Generate a random set containing 0 with n_elements elements in [0, max_val]"""
    if n_elements <= 1:
        return [0]
    # Always include 0
    others = np.random.choice(max_val, size=n_elements - 1, replace=False)
    U = list(others)
    U.append(0)
    return U


def _local_search(U, max_iter=500, time_limit=10.0):
    """Local search: try adding, removing, or replacing elements"""
    U_set = set(U)
    current_score, _, _, _ = _evaluate_set(U_set)
    best_score = current_score
    best_U = list(U_set)
    
    start_time = time.time()
    max_val = max(U_set) if U_set else 1
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try different moves
        move_type = np.random.choice(['add', 'remove', 'replace', 'shift'])
        
        new_U = set(U_set)
        
        if move_type == 'add' and len(new_U) < 200:
            # Add a random element
            max_possible = max(2 * max_val, 100)
            new_elem = np.random.randint(1, max_possible + 1)
            if new_elem not in new_U:
                new_U.add(new_elem)
        
        elif move_type == 'remove' and len(new_U) > 2:
            # Remove a random element (not 0)
            removable = [x for x in new_U if x != 0]
            if removable:
                new_U.remove(np.random.choice(removable))
        
        elif move_type == 'replace' and len(new_U) > 2:
            # Replace a random element
            removable = [x for x in new_U if x != 0]
            if removable:
                old_elem = np.random.choice(removable)
                new_elem = np.random.randint(1, 2 * max_val + 1)
                new_U.remove(old_elem)
                new_U.add(new_elem)
        
        elif move_type == 'shift':
            # Shift a random element by a small amount
            movable = [x for x in new_U if x != 0]
            if movable:
                elem = np.random.choice(movable)
                shift = np.random.randint(-10, 11)
                new_elem = elem + shift
                if new_elem >= 0 and new_elem not in new_U:
                    new_U.remove(elem)
                    new_U.add(new_elem)
        
        if new_U == U_set:
            continue
        
        new_score, _, _, _ = _evaluate_set(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = list(new_U)
            U_set = new_U
            current_score = new_score
            max_val = max(U_set) if U_set else 1
    
    return best_U, best_score


def _simulated_annealing(initial_U, start_temp=10.0, end_temp=0.01, n_iterations=2000, time_limit=120.0):
    """Simulated annealing to optimize the score"""
    U_set = set(initial_U)
    current_score, _, _, _ = _evaluate_set(U_set)
    best_score = current_score
    best_U = list(U_set)
    
    start_time = time.time()
    max_val = max(U_set) if U_set else 1
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit - 5:  # Safety margin
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / n_iterations)
        
        # Generate a neighbor
        move_type = np.random.choice(['add', 'remove', 'replace', 'shift', 'jump'])
        new_U = set(U_set)
        
        if move_type == 'add' and len(new_U) < 300:
            max_possible = max(3 * max_val, 200)
            new_elem = np.random.randint(1, max_possible + 1)
            if new_elem not in new_U:
                new_U.add(new_elem)
        
        elif move_type == 'remove' and len(new_U) > 3:
            removable = [x for x in new_U if x != 0]
            if removable:
                new_U.remove(np.random.choice(removable))
        
        elif move_type == 'replace' and len(new_U) > 3:
            removable = [x for x in new_U if x != 0]
            if removable:
                old_elem = np.random.choice(removable)
                new_elem = np.random.randint(1, 3 * max_val + 1)
                new_U.remove(old_elem)
                new_U.add(new_elem)
        
        elif move_type == 'shift':
            movable = [x for x in new_U if x != 0]
            if movable:
                elem = np.random.choice(movable)
                shift = np.random.randint(-20, 21)
                new_elem = elem + shift
                if new_elem >= 0 and new_elem not in new_U:
                    new_U.remove(elem)
                    new_U.add(new_elem)
        
        elif move_type == 'jump':
            # Larger jump
            movable = [x for x in new_U if x != 0]
            if movable:
                elem = np.random.choice(movable)
                new_elem = np.random.randint(1, 5 * max_val + 1)
                if new_elem not in new_U:
                    new_U.remove(elem)
                    new_U.add(new_elem)
        
        if new_U == U_set:
            continue
        
        new_score, _, _, _ = _evaluate_set(new_U)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < math.exp(delta / max(temp, 1e-10)):
            U_set = new_U
            current_score = new_score
            max_val = max(U_set) if U_set else 1
            
            if current_score > best_score:
                best_score = current_score
                best_U = list(U_set)
    
    return best_U, best_score


def construct():
    """Construct a finite set U to maximize the sumset/difference-set exponent lower bound."""
    start_time = time.time()
    time_budget = 110.0  # Leave 10s safety margin
    
    # Start with a reasonable initial set
    # Try a geometric progression as initial guess
    initial_U = [0]
    val = 1
    while val <= 1000 and len(initial_U) < 50:
        initial_U.append(val)
        val *= 2
    
    # Run simulated annealing
    best_U, best_score = _simulated_annealing(
        initial_U,
        start_temp=5.0,
        end_temp=0.001,
        n_iterations=5000,
        time_limit=time_budget - 20
    )
    
    # Local search refinement
    if time.time() - start_time < time_budget - 10:
        refined_U, refined_score = _local_search(
            best_U,
            max_iter=2000,
            time_limit=time_budget - 15
        )
        if refined_score > best_score:
            best_U = refined_U
            best_score = refined_score
    
    # Final check
    if time.time() - start_time < time_budget - 5:
        # One more round of SA from the best found
        final_U, final_score = _simulated_annealing(
            best_U,
            start_temp=2.0,
            end_temp=0.0005,
            n_iterations=3000,
            time_limit=time_budget - 10
        )
        if final_score > best_score:
            best_U = final_U
            best_score = final_score
    
    # Ensure 0 is in the set and return as list
    best_U = list(set(best_U))
    if 0 not in best_U:
        best_U.append(0)
    best_U.sort()
    
    return best_U
# EVOLVE-BLOCK-END
