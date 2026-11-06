import time
import random

def construct():
    best_seq = []
    deadline = time.time() + 55.0
    
    def is_valid_addition(seq, sums, prefixes, x):
        # seq is the current list of values
        # sums is the set of all block sums
        # prefixes is the list of prefix sums [0, a1, a1+a2, ...]
        # x is the candidate next value
        
        new_prefix = prefixes[-1] + x
        # New differences are new_prefix - p for p in prefixes
        # We need to check if any of these are in sums
        for p in prefixes:
            if new_prefix - p in sums:
                return False
        return True

    while time.time() < deadline:
        seq = []
        sums = set()
        prefixes = [0]
        last_val = 0
        
        while True:
            valid_candidates = []
            # Range is last_val + 1 to 256
            for x in range(last_val + 1, 257):
                if is_valid_addition(seq, sums, prefixes, x):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Randomly pick a candidate
            # Heuristic: Maybe picking a random one is good for exploration.
            # Sometimes picking the smallest valid one leads to dead ends quickly.
            # Picking a random one helps escape local optima.
            x = random.choice(valid_candidates)
            
            # Add x to the sequence
            seq.append(x)
            new_prefix = prefixes[-1] + x
            prefixes.append(new_prefix)
            
            # Update sums
            # New sums are new_prefix - p for p in prefixes[:-1] (old prefixes)
            # Note: The loop in is_valid_addition already checked against old sums.
            # We just need to add the new ones to the set.
            for p in prefixes[:-1]:
                sums.add(new_prefix - p)
            
            last_val = x
            
        if len(seq) > len(best_seq):
            best_seq = seq

    return best_seq
