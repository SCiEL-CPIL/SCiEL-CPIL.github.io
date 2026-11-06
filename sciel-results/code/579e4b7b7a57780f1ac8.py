import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    
    # Pre-compute potential candidates
    all_candidates = list(range(1, max_val + 1))
    
    def get_sums(s):
        """Return a set of all pairwise sums a+b where a<=b from sorted list s."""
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                b = s[j]
                sums.add(a + b)
        return sums

    def can_add(s, sums, x):
        """Check if x can be added to s without creating duplicate sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def incremental_add(s, sums, x):
        """
        Add x to sorted list s and update sums incrementally.
        Returns (new_s, new_sums).
        """
        new_sums = sums.copy()
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        new_s = s + [x]
        new_s.sort()
        return new_s, new_sums

    def remove_element(s, sums, x):
        """
        Remove x from sorted list s and update sums incrementally.
        Returns (new_s, new_sums).
        """
        new_sums = sums.copy()
        for y in s:
            if y != x:
                new_sums.discard(x + y)
        new_sums.discard(2 * x)
        new_s = [y for y in s if y != x]
        return new_s, new_sums

    def greedy_fill(s, sums, candidates, bias=None):
        """
        Greedily add elements from candidates to s.
        Returns the updated sorted list s and the updated sums set.
        bias: 'random', 'low', 'high'
        """
        s = sorted(s)
        sums = get_sums(s)
        
        # Filter candidates that are not in s
        s_set = set(s)
        avail = [c for c in candidates if c not in s_set]
        
        if bias == 'random':
            random.shuffle(avail)
        elif bias == 'low':
            avail.sort()
        elif bias == 'high':
            avail.sort(reverse=True)
        else:
            random.shuffle(avail)
            
        for x in avail:
            if time.time() > deadline - 2:
                break
            if can_add(s, sums, x):
                s, sums = incremental_add(s, sums, x)
                
        return s, sums

    def try_2swap(current_set, current_sums):
        """
        Try to swap two non-base elements for two different ones.
        Returns (new_set, new_sums) or (current_set, current_sums) if no swap found.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
        
        non_base_elements = [x for x in current_set if x not in base_set]
        if len(non_base_elements) < 2:
            return current_set, current_sums
        
        s_set = set(current_set)
        candidates = [c for c in all_candidates if c not in s_set]
        if len(candidates) < 2:
            return current_set, current_sums
        
        # Try a few random 2-swaps
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            # Pick two random elements to remove
            to_remove = random.sample(non_base_elements, 2)
            remove_set = set(to_remove)
            
            # Create temp set without removed elements using incremental removal
            temp_set = current_set[:]
            temp_sums = current_sums
            for x in to_remove:
                temp_set, temp_sums = remove_element(temp_set, temp_sums, x)
            
            # Try to find two candidates that can be added
            random.shuffle(candidates)
            added = []
            for x in candidates:
                if len(added) == 2:
                    break
                if can_add(temp_set + added, temp_sums, x):
                    temp_set, temp_sums = incremental_add(temp_set, temp_sums, x)
                    added.append(x)
            
            if len(added) == 2:
                return temp_set, temp_sums
        
        return current_set, current_sums

    def try_swap(current_set, current_sums):
        """
        Try to swap one non-base element for another to maintain size.
        Returns (new_set, new_sums) or (current_set, current_sums) if no swap found.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
        
        non_base_elements = [x for x in current_set if x not in base_set]
        if not non_base_elements:
            return current_set, current_sums
        
        s_set = set(current_set)
        candidates = [c for c in all_candidates if c not in s_set]
        if not candidates:
            return current_set, current_sums
        
        # Try swapping a random non-base element with a random candidate
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            to_remove = random.choice(non_base_elements)
            
            random.shuffle(candidates)
            for x in candidates:
                temp_set, temp_sums = remove_element(current_set, current_sums, to_remove)
                if can_add(temp_set, temp_sums, x):
                    new_set, new_sums = incremental_add(temp_set, temp_sums, x)
                    return new_set, new_sums
        return current_set, current_sums

    def kick_and_refill(current_set, current_sums, kick_size, bias=None):
        """
        Remove 'kick_size' non-base elements from current_set, then greedy refill.
        Uses incremental sum removal for efficiency.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
            
        non_base_elements = [x for x in current_set if x not in base_set]
        
        if len(non_base_elements) < kick_size:
            new_set, new_sums = greedy_fill(current_set, current_sums, all_candidates, bias=bias or 'random')
            return new_set, new_sums
            
        to_remove = random.sample(non_base_elements, kick_size)
        
        # Use incremental removal for efficiency
        new_set = current_set[:]
        new_sums = current_sums
        for x in to_remove:
            new_set, new_sums = remove_element(new_set, new_sums, x)
        
        best_new_set = new_set
        best_new_sums = new_sums
        best_new_size = len(new_set)
        
        for b in (bias, 'random', 'low', 'high'):
            if time.time() > deadline - 5:
                break
            if b is None:
                continue
            candidate_set, candidate_sums = greedy_fill(new_set[:], new_sums, all_candidates, bias=b)
            if len(candidate_set) > best_new_size:
                best_new_set = candidate_set
                best_new_sums = candidate_sums
                best_new_size = len(candidate_set)
        
        return best_new_set, best_new_sums

    def polish(current_set):
        """
        Final polish: try to add any missing elements one by one in sorted order.
        """
        s = sorted(current_set)
        sums = get_sums(s)
        s_set = set(s)
        
        for x in range(1, max_val + 1):
            if x in s_set:
                continue
            if can_add(s, sums, x):
                s, sums = incremental_add(s, sums, x)
                s_set.add(x)
        
        return s, sums

    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    biases = ['random', 'low', 'high', 'random', 'random']
    for bias in biases:
        if time.time() > deadline - 20:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)
            
    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 40
    
    while time.time() < deadline - 10:
        r = random.random()
        if r < 0.5:
            # Kick and refill (50%)
            r2 = random.random()
            if r2 < 0.4:
                kick_size = 1
            elif r2 < 0.8:
                kick_size = 2
            else:
                kick_size = 3
            new_set, new_sums = kick_and_refill(current_set, current_sums, kick_size)
        elif r < 0.75:
            # 1-swap move (25%)
            new_set, new_sums = try_swap(current_set, current_sums)
        else:
            # 2-swap move (25%)
            new_set, new_sums = try_2swap(current_set, current_sums)
        
        if len(new_set) > best_size:
            best_set = new_set
            best_sums = new_sums
            best_size = len(new_set)
            current_set = new_set
            current_sums = new_sums
            current_size = best_size
            no_improve_count = 0
        elif len(new_set) == best_size:
            if new_set != current_set:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
        else:
            if len(new_set) >= current_size - 1:
                current_set = new_set
                current_sums = new_sums
                current_size = len(new_set)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            current_set = best_set[:]
            current_sums = best_sums
            current_size = best_size
            no_improve_count = 0

    if time.time() < deadline - 2:
        polished_set, _ = polish(best_set)
        if len(polished_set) > best_size:
            best_set = polished_set
            best_size = len(best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
