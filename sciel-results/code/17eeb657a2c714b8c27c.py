import time

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of every
    nonempty consecutive block are distinct. Maximize length.
    
    Strategy: Beam Search over sets of numbers.
    State: (sorted_tuple, set_of_products)
    Expansion: Add each candidate number not in the set, check validity incrementally.
    Pruning: Keep top K states by length.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    all_numbers = list(range(2, 129))
    
    # State: (tuple, set)
    beam = [((), set())]
    best_seq = ()
    best_len = 0
    
    beam_width = 30
    max_steps = 40
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        next_beam = []
        
        for seq, prods in beam:
            for x in all_numbers:
                if x in seq:
                    continue
                
                # Find insertion index
                insert_idx = 0
                for i, val in enumerate(seq):
                    if val < x:
                        insert_idx = i + 1
                    else:
                        break
                
                # Check validity
                invalid = False
                new_prods = set()
                
                # Left suffixes: 1, seq[-1], seq[-2]*seq[-1], ...
                left_sufs = [1]
                p = 1
                for i in range(insert_idx - 1, -1, -1):
                    p *= seq[i]
                    left_sufs.append(p)
                    
                # Right prefixes: 1, seq[insert_idx], seq[insert_idx]*seq[insert_idx+1], ...
                right_prefs = [1]
                p = 1
                for i in range(insert_idx, len(seq)):
                    p *= seq[i]
                    right_prefs.append(p)
                
                for ls in left_sufs:
                    for rp in right_prefs:
                        val = ls * x * rp
                        if val in prods:
                            invalid = True
                            break
                        new_prods.add(val)
                    if invalid:
                        break
                
                if invalid:
                    continue
                
                # Valid
                new_seq_list = list(seq)
                new_seq_list.insert(insert_idx, x)
                new_seq = tuple(new_seq_list)
                new_prods_set = prods.union(new_prods)
                
                next_beam.append((new_seq, new_prods_set))
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        
        if not next_beam:
            break
            
        # Deduplicate by seq
        unique_beam = {}
        for seq, prods in next_beam:
            if seq not in unique_beam:
                unique_beam[seq] = prods
            
        next_beam = list(unique_beam.items())
        
        next_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = next_beam[:beam_width]
        
    return list(best_seq)
