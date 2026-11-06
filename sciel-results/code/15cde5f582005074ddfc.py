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

    def add_element(s, sums, x):
        """Add x to s (sorted), update sums incrementally. Returns (new_s, new_sums)."""
        new_s = list(s)
        new_s.append(x)
        new_s.sort()
        new_sums = set(sums)
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_s, new_sums

    def remove_element(s, sums, x):
        """Remove x from s, update sums. Returns (new_s, new_sums)."""
        new_s = [y for y in s if y != x]
        # Recompute sums (simpler and still fast for n<=25)
        new_sums = get_sums(new_s)
        return new_s, new_sums

    def greedy_fill(s, sums, candidates, bias=None):
        """
        Greedily add elements from candidates to s.
        Returns the updated sorted list s and the updated sums set.
        bias: 'random', 'low', 'high', 'spread'
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
        elif bias == 'spread':
            # Alternate between low and high to spread out
            avail.sort()
            mid = len(avail) // 2
            low_half = avail[:mid]
            high_half = avail[mid:]
            random.shuffle(low_half)
            random.shuffle(high_half)
            merged = []
            li = hi = 0
            while li < len(low_half) and hi < len(high_half):
                if random.random() < 0.5:
                    merged.append(low_half[li]); li += 1
                else:
                    merged.append(high_half[hi]); hi += 1
            merged.extend(low_half[li:])
            merged.extend(high_half[hi:])
            avail = merged
        else:
            random.shuffle(avail)
            
        for x in avail:
            if time.time() > deadline - 2:
                break
            if can_add(s, sums, x):
                s, sums = add_element(s, sums, x)
                
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
        scored = []
        for x in available:
            if can_add(s, sums, x):
                # Count how many other available candidates this would block
                block_count = 0
                for y in available:
                    if y == x:
                        continue
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
            
            new_s, new_sums = add_element(s, sums, x)
            
            if len(new_s) >= target_size:
                return new_s, new_sums
            
            # Recurse if depth allows
            if max_depth > 1:
                ext_s, ext_sums = backtracking_extend(new_s, new_sums, target_size, max_depth - 1)
                if len(ext_s) > len(s):
                    return ext_s, ext_sums
        
        return s, sums

    def try_add_one(s, sums, time_budget=1.0):
        """
        Dedicated routine: try to add exactly one element to s.
        Uses exhaustive scan with scoring.
        Returns (new_s, new_sums) if successful, else (s, sums).
        """
        s = sorted(s)
        sums = get_sums(s)
        s_set = set(s)
        
        # Collect all valid additions
        valid = []
        for x in all_candidates:
            if x in s_set:
                continue
            if can_add(s, sums, x):
                # Score: how many other valid candidates would be blocked
                block = 0
                for y in all_candidates:
                    if y in s_set or y == x:
                        continue
                    if can_add(s, sums, y) and (x + y in sums):
                        block += 1
                valid.append((block, x))
        
        if not valid:
            return s, sums
        
        # Try in order of least blocking first
        valid.sort()
        
        for block, x in valid:
            if time.time() > start_time + time_budget:
                break
            new_s, new_sums = add_element(s, sums, x)
            return new_s, new_sums
        
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
        
        # Refill with a random bias
        new_set, new_sums = greedy_fill(new_set, new_sums, all_candidates, bias='random')
        
        return new_set, new_sums

    # Initial Solution
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    # Try several initial greedy fills with different biases to find a good starting point
    biases = ['random', 'low', 'high', 'spread', 'random', 'spread']
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
    max_no_improve = 30
    
    while time.time() < deadline - 15:
        
        # Determine kick size
        r = random.random()
        if r < 0.35:
            kick_size = 1
        elif r < 0.75:
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
        
        # Try dedicated add-one
        if len(new_set) == best_size and time.time() < deadline - 6:
            new_set, new_sums = try_add_one(new_set, new_sums, time_budget=0.5)
        
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
    
    # Final add-one attempt
    if time.time() < deadline - 2:
        best_set, best_sums = try_add_one(best_set, best_sums, time_budget=1.0)
        best_size = len(best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
