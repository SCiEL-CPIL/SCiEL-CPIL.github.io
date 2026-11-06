import time
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    def reverse_greedy():
        """
        Builds the sequence by prepending the largest possible valid number.
        """
        seq = []
        sums_set = set()
        prefix_sums = {0}
        
        # We iterate from 256 down to 1
        for x in range(256, 0, -1):
            # Check if x can be prepended
            # Condition 1: x < seq[0] if seq is not empty
            if seq and x >= seq[0]:
                continue
            
            # Condition 2: No sum collision
            # New sums are x + p for p in prefix_sums
            # These must not be in sums_set
            valid = True
            new_sums = []
            for p in prefix_sums:
                ns = x + p
                if ns in sums_set:
                    valid = False
                    break
                new_sums.append(ns)
            
            if valid:
                # Update state
                seq.insert(0, x)
                sums_set.update(new_sums)
                # Update prefix sums
                # New prefix sums are {0} U {x + p for p in prefix_sums}
                new_prefix_sums = {0}
                for p in prefix_sums:
                    new_prefix_sums.add(x + p)
                prefix_sums = new_prefix_sums
        
        return seq

    def forward_greedy():
        """
        Builds the sequence by appending the smallest possible valid number.
        """
        seq = []
        sums_set = set()
        suffix_sums = {0} # Sums of suffixes of seq. Initially {0}
        
        for x in range(1, 257):
            # Condition 1: x > seq[-1] if seq is not empty
            if seq and x <= seq[-1]:
                continue
            
            # Condition 2: No sum collision
            # New sums are x + s for s in suffix_sums
            # These must not be in sums_set
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns in sums_set:
                    valid = False
                    break
                new_sums.append(ns)
            
            if valid:
                # Update state
                seq.append(x)
                sums_set.update(new_sums)
                # Update suffix sums
                # New suffix sums are {0} U {s + x for s in suffix_sums}?
                # Suffixes of new seq:
                # [x], [a_k, x], ...
                # Sums: x, a_k+x, ...
                # Old suffix sums were sums of suffixes of old seq.
                # New suffix sums = {0} U {x + s for s in old_suffix_sums}
                new_suffix_sums = {0}
                for s in suffix_sums:
                    new_suffix_sums.add(x + s)
                suffix_sums = new_suffix_sums
        
        return seq

    # Try both strategies and keep the best
    seq1 = reverse_greedy()
    seq2 = forward_greedy()
    
    best_seq = seq1 if len(seq1) > len(seq2) else seq2
    
    # Optional: Try to extend best_seq with a simple DFS if time permits?
    # Given the 60s limit and that greedy is O(N^2) or so, it's very fast.
    # We can afford a small DFS to see if we can beat the greedy result.
    
    def try_extend(seq):
        # Try to append a number > seq[-1]
        # This is just forward greedy starting from seq
        # But we might have missed some numbers?
        # Actually, forward_greedy already picks the smallest available.
        # Reverse_greedy picks largest available.
        # A mix might be better, but for now, return the best greedy.
        return seq

    # If we have time, we can try a local search:
    # Remove the last element and try a different one?
    # Or remove an element in the middle?
    # This is complex. Let's stick to the two greedy baselines first.
    
    return best_seq
