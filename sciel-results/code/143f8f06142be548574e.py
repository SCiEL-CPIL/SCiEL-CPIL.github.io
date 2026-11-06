import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin
    
    # Primes up to 128 for vector representation
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    
    def get_exponent_vector(n):
        vec = [0] * num_primes
        for p, idx in prime_index.items():
            if p > n:
                break
            while n % p == 0:
                vec[idx] += 1
                n //= p
        return tuple(vec)
    
    # Precompute exponent vectors for all candidates
    candidates = list(range(2, 129))
    exp_vecs = {x: get_exponent_vector(x) for x in candidates}
    
    def add_vector(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
    
    def subtract_vector(v1, v2):
        return tuple(a - b for a, b in zip(v1, v2))
    
    def is_valid_insertion(seq, new_n, prods_set):
        """
        Check if new_n can be appended to seq (maintaining increasing order).
        Returns True if valid, False otherwise.
        Also returns the set of new products if valid.
        """
        # Compute suffix products' exponent vectors
        suffix_vec = [0] * num_primes
        new_prods = set()
        
        # The new singleton product is just new_n
        new_vec = exp_vecs[new_n]
        if new_vec in prods_set:
            return False, None
        new_prods.add(new_vec)
        
        # For each suffix of seq, compute product * new_n
        for i in range(len(seq) - 1, -1, -1):
            suffix_vec = add_vector(suffix_vec, exp_vecs[seq[i]])
            prod_vec = add_vector(suffix_vec, new_vec)
            if prod_vec in prods_set:
                return False, None
            new_prods.add(prod_vec)
        
        return True, new_prods
    
    def greedy_search(skip_prob=0.0, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        seq = []
        prods_set = set()
        
        for x in candidates:
            if time.time() > deadline - 10.0:
                break
            
            valid, new_prods = is_valid_insertion(seq, x, prods_set)
            if not valid:
                continue
            
            # Random skip for exploration
            if skip_prob > 0 and rng.random() < skip_prob:
                continue
            
            seq.append(x)
            prods_set.update(new_prods)
        
        return seq
    
    # First pass: simple greedy
    best_seq = greedy_search(skip_prob=0.0)
    best_len = len(best_seq)
    
    # Second pass: beam search
    # State: (seq, prods_set)
    # We'll use a beam of width 50
    
    beam_width = 50
    beam = [([], set())]
    
    for x in candidates:
        if time.time() > deadline - 15.0:
            break
        
        new_beam = []
        for seq, prods_set in beam:
            valid, new_prods = is_valid_insertion(seq, x, prods_set)
            if valid:
                new_seq = seq + [x]
                new_prods_set = prods_set.copy()
                new_prods_set.update(new_prods)
                new_beam.append((new_seq, new_prods_set))
        
        if not new_beam:
            break
        
        # Sort by length, then by lexicographic order for tie-breaking
        new_beam.sort(key=lambda s: (len(s[0]), s[0]), reverse=True)
        
        # Deduplicate: keep only unique sequences
        seen = set()
        deduped_beam = []
        for seq, prods_set in new_beam:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                deduped_beam.append((seq, prods_set))
            if len(deduped_beam) >= beam_width:
                break
        
        beam = deduped_beam
        
        # Update best
        if beam:
            if len(beam[0][0]) > best_len:
                best_len = len(beam[0][0])
                best_seq = beam[0][0]
    
    # Third pass: local search on the best sequence found
    # Try removing an element and then greedily adding back
    for _ in range(50):  # Limited restarts
        if time.time() > deadline - 5.0:
            break
        
        # Randomly pick a sequence to modify
        if random.random() < 0.3 and best_len > 0:
            # Start from a random prefix of best_seq
            mod_len = random.randint(1, best_len)
            seq = best_seq[:mod_len]
            prods_set = set()
            # Rebuild prods_set for this prefix
            for i in range(len(seq)):
                for j in range(i, len(seq)):
                    vec = [0] * num_primes
                    for k in range(i, j + 1):
                        vec = add_vector(vec, exp_vecs[seq[k]])
                    prods_set.add(tuple(vec))
        else:
            seq = list(best_seq)
            prods_set = set()
            for i in range(len(seq)):
                for j in range(i, len(seq)):
                    vec = [0] * num_primes
                    for k in range(i, j + 1):
                        vec = add_vector(vec, exp_vecs[seq[k]])
                    prods_set.add(tuple(vec))
        
        # Try to add elements after the last element
        last_val = seq[-1] if seq else 1
        for x in candidates:
            if x <= last_val:
                continue
            if time.time() > deadline - 5.0:
                break
            
            valid, new_prods = is_valid_insertion(seq, x, prods_set)
            if valid:
                seq.append(x)
                prods_set.update(new_prods)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    return best_seq
