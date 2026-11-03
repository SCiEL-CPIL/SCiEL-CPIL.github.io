import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    total_space = q ** n

    # Precompute neighbor indices for each word (for radius=1, this is 1 + n*(q-1) neighbors)
    # We'll use numpy arrays for fast coverage updates
    # For K_3(7,1): each word covers 1 + 7*2 = 15 words
    # Precompute as a list of arrays for fast slicing
    
    # Actually, let's precompute as a 2D numpy array: neighbors[w] = list of indices covered by w
    # For radius=1: neighbors[w] has 15 entries (w itself + 14 neighbors)
    
    # Build neighbor lists
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # For each word, compute all words within Hamming distance <= radius
    # For radius=1: the word itself + all words differing in exactly one position
    neighbor_lists = []
    for w in range(total_space):
        covered = [w]
        for pos in range(n):
            digit_val = (w // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nb = w + diff * powers[pos]
                covered.append(nb)
        neighbor_lists.append(covered)
    
    # Convert to numpy arrays for fast operations
    # neighbors_np[w] is a 1D array of indices
    neighbors_np = np.array(neighbor_lists, dtype=np.int32)
    
    # Precompute: for each position in the neighbor list, which word it belongs to
    # This allows fast coverage updates
    
    # Simulated Annealing approach:
    # State: a binary vector x of length total_space, where x[w]=1 means w is in the code
    # Objective: minimize sum(x) subject to coverage constraint
    # We use a penalty-based approach: minimize sum(x) + penalty * uncovered_count
    
    # Initialize with a greedy solution
    def run_greedy():
        covered = np.zeros(total_space, dtype=np.int32)
        selected = []
        remaining = np.ones(total_space, dtype=bool)
        
        while remaining.any():
            if time.time() > deadline - 20:
                break
            # Find word that covers most remaining
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = np.sum(remaining[neighbors_np[w]])
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1 or best_gain <= 0:
                break
            
            selected.append(best_word)
            covered[neighbors_np[best_word]] += 1
            remaining = (covered == 0)
        
        return selected
    
    # Get initial solution
    initial = run_greedy()
    
    if time.time() > deadline - 30:
        # Not enough time for SA, just return greedy
        return [[(w // powers[i]) % q for i in range(n)] for w in initial]
    
    # Set up SA
    x = np.zeros(total_space, dtype=np.int8)
    for w in initial:
        x[w] = 1
    
    # Coverage count
    coverage = np.zeros(total_space, dtype=np.int32)
    for w in initial:
        coverage[neighbors_np[w]] += 1
    
    # SA parameters
    T0 = 5.0  # Initial temperature
    T_end = 0.01
    cooling_rate = 0.9995
    max_iterations = 50000
    
    best_x = x.copy()
    best_size = x.sum()
    
    T = T0
    current_size = x.sum()
    current_uncovered = (coverage == 0).sum()
    
    # We minimize: size + lambda * uncovered
    # lambda should be large enough to penalize uncovered words
    # But we want to allow temporary increases in size if it reduces uncovered
    # Actually, let's use a different approach:
    # Accept moves that reduce size (if still valid) or increase size (if reduces uncovered)
    # Use temperature to control acceptance of "bad" moves
    
    for it in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Random move: add or remove a random word
        w = random.randint(0, total_space - 1)
        
        if x[w] == 1:
            # Try to remove w
            # Check if removal keeps coverage valid
            # After removal, coverage of neighbors[w] decreases by 1
            # If any neighbor would have coverage 0, removal is invalid
            new_coverage = coverage[neighbors_np[w]] - 1
            if (new_coverage < 0).any():
                # Invalid, skip
                continue
            
            # This is a "good" move (reduces size), always accept
            coverage[neighbors_np[w]] -= 1
            x[w] = 0
            current_size -= 1
            current_uncovered = (coverage == 0).sum()
            
            if current_size < best_size:
                best_size = current_size
                best_x = x.copy()
            
        else:
            # Try to add w
            # This increases size by 1 but may reduce uncovered count
            new_coverage = coverage[neighbors_np[w]] + 1
            # Count how many newly covered words
            newly_covered = ((coverage[neighbors_np[w]] == 0) & (new_coverage > 0)).sum()
            
            # Delta in objective: +1 (size) - lambda * newly_covered
            # We'll use a simple acceptance rule based on temperature
            # Accept with probability 1 if newly_covered > 0 and random < exp(newly_covered / T)
            # Otherwise reject
            
            if newly_covered > 0:
                # Accept with probability based on benefit
                prob = min(1.0, np.exp(newly_covered / T))
                if random.random() < prob:
                    coverage[neighbors_np[w]] += 1
                    x[w] = 1
                    current_size += 1
                    current_uncovered = (coverage == 0).sum()
            else:
                # No benefit, accept with small probability to explore
                if random.random() < np.exp(-1.0 / T):
                    coverage[neighbors_np[w]] += 1
                    x[w] = 1
                    current_size += 1
                    current_uncovered = (coverage == 0).sum()
        
        # Update temperature
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Final polish: remove redundant words
    x = best_x
    coverage = np.zeros(total_space, dtype=np.int32)
    for w in range(total_space):
        if x[w] == 1:
            coverage[neighbors_np[w]] += 1
    
    # Remove redundant words
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for w in range(total_space):
            if x[w] == 1:
                # Check if w is redundant
                new_coverage = coverage[neighbors_np[w]] - 1
                if not (new_coverage < 0).any():
                    coverage[neighbors_np[w]] -= 1
                    x[w] = 0
                    changed = True
    
    # Decode result
    result = []
    for w in range(total_space):
        if x[w] == 1:
            word = [(w // powers[i]) % q for i in range(n)]
            result.append(word)
    
    return result
