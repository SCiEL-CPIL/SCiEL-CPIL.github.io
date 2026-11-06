import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    def sums_set(S):
        """Compute all pairwise sums for set S."""
        s = set()
        elems = sorted(S)
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s.add(elems[i] + elems[j])
        return s
    
    def can_add(x, S, sums):
        """Check if x can be added to S without violating Sidon property."""
        if 2 * x in sums:
            return False
        for y in S:
            if x + y in sums:
                return False
        return True
    
    def greedy_fill(S, sums, max_steps=10000):
        """Greedy fill: add valid candidates one by one."""
        S = set(S)
        sums = set(sums)
        steps = 0
        while steps < max_steps:
            steps += 1
            added = False
            candidates = []
            for x in range(1, 285):
                if x not in S and can_add(x, S, sums):
                    candidates.append(x)
            if not candidates:
                break
            random.shuffle(candidates)
            # Try a few candidates, pick one that allows most future additions
            best_x = None
            best_future = -1
            for x in candidates[:min(20, len(candidates))]:
                new_S = S | {x}
                new_sums = set(sums)
                new_sums.add(2 * x)
                for y in S:
                    new_sums.add(x + y)
                # Count how many more we could add
                future = 0
                for x2 in range(1, 285):
                    if x2 not in new_S and can_add(x2, new_S, new_sums):
                        future += 1
                if future > best_future:
                    best_future = future
                    best_x = x
            if best_x is None:
                break
            S = S | {best_x}
            sums.add(2 * best_x)
            for y in S - {best_x}:
                sums.add(best_x + y)
        return S, sums
    
    def destroy_and_repair(S, sums, num_removals):
        """Remove some elements and try to rebuild."""
        S = set(S)
        sums = set(sums)
        elems = list(S)
        random.shuffle(elems)
        # Keep fixed elements
        to_remove = [x for x in elems if x not in fixed][:num_removals]
        for x in to_remove:
            S.discard(x)
        # Recompute sums
        sums = sums_set(S)
        # Greedy fill again
        S, sums = greedy_fill(S, sums)
        return S, sums
    
    # Initial: build from fixed
    initial_sums = sums_set(fixed)
    S, sums = greedy_fill(fixed, initial_sums)
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Try multiple destroy-and-repair iterations
    while time.time() < deadline:
        # Determine how many to remove
        current_size = len(S)
        if current_size <= 4:
            num_removals = max(1, current_size // 3)
        else:
            num_removals = max(1, random.randint(1, min(5, current_size - 4)))
        
        S, sums = destroy_and_repair(S, sums, num_removals)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
        
        # If we found size 20, we're done
        if best_size >= 20:
            break
        
        # If we've been stuck for a while, do a larger restart
        if random.random() < 0.1:
            S, sums = greedy_fill(fixed, initial_sums)
    
    return best_set
