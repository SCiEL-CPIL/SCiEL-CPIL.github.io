import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        L = len(seq)
        # Prefix sums
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_increasing():
        """Standard greedy: add smallest possible next."""
        values = []
        sums = set()
        for x in range(1, 257):
            # Check if x can be appended to values
            # Since x > all v in values, new sums are suffixes of values + x, and x itself
            # Suffix sums of values:
            # Let V = [v1, ..., vk]. Suffixes are vk, vk+v_{k-1}, ..., sum(V).
            # New sums: x, x+vk, x+vk+v_{k-1}, ...
            
            potential_sums = set()
            potential_sums.add(x)
            
            # Calculate suffix sums of values
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            for s in suffix_sums:
                potential_sums.add(x + s)
            
            if not sums.intersection(potential_sums):
                values.append(x)
                sums.update(potential_sums)
        
        return values

    def greedy_decreasing():
        """Greedy: pick largest available, then next largest?
        If we pick 256, then 1? No, we pick in decreasing order of value.
        If we pick 256, it's the max.
        If we pick 255 next, it's inserted before 256.
        The incremental check for "prepending" (adding smallest) uses prefix sums.
        Let's implement a symmetric greedy for decreasing order of selection.
        """
        values = []
        sums = set()
        # We iterate x from 256 down to 1.
        # If we add x, it becomes the smallest element.
        # New sums are {x} U {x + P | P is prefix sum of values}
        for x in range(256, 0, -1):
            potential_sums = set()
            potential_sums.add(x)
            
            # Prefix sums of values (values is sorted ascending in memory, but we are adding smaller ones)
            # Wait, if we add x (small) to values (larger), x is at the front.
            # Prefix sums of values: v1, v1+v2, ...
            # New sums: x, x+v1, x+v1+v2, ...
            
            prefix_sums = []
            current_sum = 0
            for v in values: # values is sorted ascending
                current_sum += v
                prefix_sums.append(current_sum)
            
            for s in prefix_sums:
                potential_sums.add(x + s)
            
            if not sums.intersection(potential_sums):
                values.append(x)
                # Keep values sorted?
                # Actually, for the next step, we need values to be sorted to compute prefix sums correctly?
                # Yes, prefix sums depend on order.
                values.sort()
                sums.update(potential_sums)
        
        return values

    # 1. Get baseline from greedy
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    # Also try decreasing greedy?
    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)

    # 2. Local Search
    # Use the best found so far as starting point
    current_seq = list(best_seq)
    
    # To speed up is_valid, we can pass the set of sums if we track it, but re-calc is safer and simple.
    # Given L is small, O(L^2) is fine.
    
    # Randomized Local Search
    # We will try to improve by adding, removing, or swapping.
    
    # Shuffle candidates for random selection
    all_candidates = list(range(1, 257))
    
    while time.time() < deadline:
        # Randomly decide move type
        # Bias towards adding to maximize length
        r = random.random()
        
        if r < 0.6: # Add
            # Pick a random candidate not in current_seq
            # Optimization: Maintain a set of current_seq for O(1) lookup
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                new_seq = current_seq + [x]
                new_seq.sort()
                if is_valid(new_seq):
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
        
        elif r < 0.8: # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
                # Don't update best on remove
        
        else: # Swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    # Pick one to remove and one to add
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    if is_valid(new_seq):
                        current_seq = new_seq
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)

    return sorted(best_seq)
