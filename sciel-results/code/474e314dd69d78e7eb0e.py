import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute prime factorization vectors
    vectors = {}
    for x in range(2, 129):
        vec = [0] * num_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vectors[x] = tuple(vec)
        
    def get_all_product_vectors(seq):
        n = len(seq)
        if n == 0:
            return set()
        products = set()
        # Optimize: use local variables
        vecs = vectors
        np = num_primes
        for i in range(n):
            current = [0] * np
            for j in range(i, n):
                v_j = vecs[seq[j]]
                for k in range(np):
                    current[k] += v_j[k]
                products.add(tuple(current))
        return products

    def is_valid(seq):
        n = len(seq)
        prods = get_all_product_vectors(seq)
        return len(prods) == n * (n + 1) // 2

    # Greedy Initialization
    def greedy_construct():
        seq = []
        for x in range(2, 129):
            # Since we iterate in increasing order, x > seq[-1] always holds if we append
            new_seq = seq + [x]
            if is_valid(new_seq):
                seq = new_seq
        return seq

    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    rng = random.Random(42)
    no_improve_count = 0
    
    while time.time() < deadline:
        seq = best_seq[:]
        current_len = len(seq)
        
        # Random Move Selection
        move_type = rng.random()
        
        if move_type < 0.8:
            # Try to insert a missing element
            # Optimization: Pre-compute missing? Or just filter.
            # Filtering is O(127) which is small.
            missing = [x for x in range(2, 129) if x not in seq]
            if not missing:
                continue
            x = rng.choice(missing)
            
            # Find insertion pos
            pos = 0
            for i, val in enumerate(seq):
                if x < val:
                    pos = i
                    break
                else:
                    pos = i + 1
            
            new_seq = seq[:pos] + [x] + seq[pos:]
            
            if is_valid(new_seq):
                seq = new_seq
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_len = len(seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Try a swap: remove one, add one
            if len(seq) == 0:
                continue
            missing = [x for x in range(2, 129) if x not in seq]
            if not missing:
                continue
            
            x_new = rng.choice(missing)
            y_old_idx = rng.randint(0, len(seq) - 1)
            
            # Remove y_old
            seq_without_y = seq[:y_old_idx] + seq[y_old_idx+1:]
            
            # Insert x_new into seq_without_y
            pos = 0
            for i, val in enumerate(seq_without_y):
                if x_new < val:
                    pos = i
                    break
                else:
                    pos = i + 1
            
            new_seq = seq_without_y[:pos] + [x_new] + seq_without_y[pos:]
            
            if is_valid(new_seq):
                if len(new_seq) > best_len:
                    best_seq = new_seq[:]
                    best_len = len(new_seq)
                    no_improve_count = 0
                elif len(new_seq) == best_len:
                    # Accept lateral move to explore
                    if rng.random() < 0.2:
                        best_seq = new_seq[:]
                        no_improve_count = 0
                # If len(new_seq) < best_len, we ignore it for best_seq update
                # but we might use it as current state for further moves?
                # To keep it simple, we only update best_seq if it's >= best_len
                # and we restart from best_seq each iteration.
                # This is "Best-First" search with random perturbations.
                
        # If stuck for a long time, try a more aggressive restart
        if no_improve_count > 2000:
            # Perturb best_seq: remove 2 random elements
            seq = best_seq[:]
            if len(seq) >= 2:
                idx1 = rng.randint(0, len(seq)-1)
                idx2 = rng.randint(0, len(seq)-1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(seq)
                # Sort indices to remove safely
                if idx1 > idx2: idx1, idx2 = idx2, idx1
                seq.pop(idx1)
                seq.pop(idx2)
            best_seq = seq[:]
            best_len = len(best_seq)
            no_improve_count = 0

    return best_seq
