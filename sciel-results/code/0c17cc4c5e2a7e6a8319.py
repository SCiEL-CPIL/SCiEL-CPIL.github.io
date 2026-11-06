import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    N = 512
    K = 32

    def score_bits(bits):
        count = 0
        # d ranges from 1 to 255
        for d in range(1, 256):
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count

    def bits_to_list(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]

    # --- Phase 1: Greedy Initialization ---
    current_bits = 0
    current_set = set()
    
    # Greedy: Add the element that maximizes the score incrementally
    for _ in range(K):
        best_x = -1
        best_score = -1
        for x in range(1, N + 1):
            if x in current_set:
                continue
            new_bits = current_bits | (1 << (x - 1))
            sc = score_bits(new_bits)
            if sc > best_score:
                best_score = sc
                best_x = x
        if best_x != -1:
            current_set.add(best_x)
            current_bits |= (1 << (best_x - 1))
            
    best_score = best_score
    best_bits = current_bits
    current_score = best_score

    # --- Phase 2: Simulated Annealing ---
    T_start = 10.0
    T_end = 0.01
    
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Time-based cooling
        elapsed = time.time() - start_time
        progress = elapsed / time_limit
        # Ensure progress is between 0 and 1
        if progress > 1:
            progress = 1
        T = T_start * pow(T_end / T_start, progress)
        
        # Pick a random element to remove
        # list(current_set) is O(K), K=32, so it's fast
        x = random.choice(list(current_set))
        # Pick a random element to add
        y = random.randint(1, N)
        
        if y in current_set:
            continue
            
        # Calculate new state
        new_bits = current_bits
        new_bits &= ~(1 << (x - 1))
        new_bits |= (1 << (y - 1))
        
        new_score = score_bits(new_bits)
        delta = new_score - current_score
        
        # Acceptance probability
        # Use a small epsilon to avoid division by zero if T becomes 0, but T_end is 0.01
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept move
            current_set.remove(x)
            current_set.add(y)
            current_bits = new_bits
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Restart strategy: If stuck for a long time, perturb
        if no_improve_count > 5000:
            # Randomize 5 elements to escape local minimum
            for _ in range(5):
                x_rand = random.choice(list(current_set))
                y_rand = random.randint(1, N)
                if y_rand not in current_set:
                    current_set.remove(x_rand)
                    current_set.add(y_rand)
                    current_bits &= ~(1 << (x_rand - 1))
                    current_bits |= (1 << (y_rand - 1))
            current_score = score_bits(current_bits)
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits
            no_improve_count = 0

    return bits_to_list(best_bits)
