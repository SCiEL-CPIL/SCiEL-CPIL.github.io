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
                s.append(x)
                s.sort()
                sums = get_sums(s)
                
        return s, sums

    def backtracking_extend(s, sums, target_size, max_depth=5):
        """
        Try to extend set s to reach target_size using backtracking with MRV heuristic.
        Returns (extended_set, extended_sums) if successful, else (s, sums).
        """
        if time.time() > deadline - 3:
            return s, sums
        
        s = sorted(s)
        sums = get_sums(s)
        current_size = len(s)
        
        if current_size >= target_size:
            return s, sums
        
        # Find all candidates that can be added
        s_set = set(s)
        available = [c for c in all_candidates if c not in s_set]
        
        # Score each candidate by how many new conflicts it would create
        # (i.e., how many other candidates it blocks)
        scored = []
        for x in available:
            if can_add(s, sums, x):
                # Count how many other available candidates this would block
                block_count = 0
                for y in available:
                    if y == x:
                        continue
                    if x + y in sums or (2 * y in sums and y != x):
                        # Adding x might block y if x+y conflicts with existing sums
                        # More precisely: after adding x, can y still be added?
                        # New sums would include x+y and 2x
                        # y can be added if for all z in s U {x}, y+z not in (sums U {x+z for z in s} U {2x})
                        # This is complex; use a simpler proxy: count direct conflicts
                        if x + y in sums:
                            block_count += 1
                scored.append((block_count, x))
        
        if not scored:
            return s, sums
        
        # MRV: try candidates that block the fewest others first (most flexible)
        scored.sort()
        
        # Limit branching factor for speed
        max_branch = min(len(scored), max(1, max_depth * 3))
        candidates_to_try = [x for _, x in scored[:max_branch]]
        
        # Shuffle for diversity
        random.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            if time.time() > deadline - 3:
                break
            if not can_add(s, sums, x):
                continue
            
            new_s = s + [x]
            new_s.sort()
            new_sums = get_sums(new_s)
            
            if len(new_s) >= target_size:
                return new_s, new_sums
            
            # Recurse if depth allows
            if max_depth > 1:
                ext_s, ext_sums = backtracking_extend(new_s, new_sums, target_size, max_depth - 1)
                if len(ext_s) > len(s):
                    return ext_s, ext_sums
        
        return s, sums

    def kick_and_refill(current_set, current_sums, kick_size):
        """
        Remove 'kick_size' non-base elements from current_set, then greedy refill.
        """
        if time.time() > deadline - 5:
            return current_set, current_sums
            
        # Identify non-base elements
        non_base_elements = [x for x in current_set if x not in base_set]
        
        if len(non_base_elements) < kick_size:
            # Can't kick enough, just refill
            new_set, new_sums = greedy_fill(current_set, current_sums, all_candidates, bias='random')
            return new_set, new_sums
            
        # Randomly select elements to remove
        to_remove = random.sample(non_base_elements, kick_size)
        remove_set = set(to_remove)
        
        # Create new set without removed elements
        new_set = [x for x in current_set if x not in remove_set]
        new_sums = get_sums(new_set)
        
        # Refill
        new_set, new_sums = greedy_fill(new_set, new_sums, all_candidates, bias='random')
        
        return new_set, new_sums

    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    # Try several initial greedy fills with different biases to find a good starting point
    biases = ['random', 'low', 'high', 'random', 'random']
    for bias in biases:
        if time.time() > deadline - 20:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias=bias)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)
            
    # Local Search Loop
    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 40
    
    while time.time() < deadline - 15:
        
        # Determine kick size
        r = random.random()
        if r < 0.4:
            kick_size = 1
        elif r < 0.8:
            kick_size = 2
        else:
            kick_size = 3
            
        new_set, new_sums = kick_and_refill(current_set, current_sums, kick_size)
        
        # Try to extend the result using backtracking
        if len(new_set) >= best_size - 1 and time.time() < deadline - 8:
            target = max(best_size + 1, len(new_set) + 1)
            ext_set, ext_sums = backtracking_extend(new_set, new_sums, target, max_depth=4)
            if len(ext_set) > len(new_set):
                new_set, new_sums = ext_set, ext_sums
        
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
            current_set = new_set
            current_sums = new_sums
            current_size = len(new_set)
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            current_set = best_set[:]
            current_sums = best_sums
            current_size = best_size
            no_improve_count = 0

    # Final polish: try to extend the best set one more time
    if time.time() < deadline - 5:
        target = best_size + 1
        ext_set, ext_sums = backtracking_extend(best_set, best_sums, target, max_depth=5)
        if len(ext_set) > best_size:
            best_set = ext_set
            best_sums = ext_sums
            best_size = len(ext_set)

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
