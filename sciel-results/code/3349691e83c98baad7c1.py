import time
import random

def construct():
        n = 12
        m = 40
        total_masks = 1 << n
        start_time = time.time()
        deadline = start_time + 50.0 # Safety margin

        # Helper to check comparability
        def is_comp(a, b):
            return (a & b) == a or (a & b) == b

        # Helper to calculate score of a family
        def calc_score(family):
            score = 0
            for i in range(m):
                for j in range(i+1, m):
                    if is_comp(family[i], family[j]):
                        score += 1
            return score

        # Initialization
        # Start with a random family
        family = []
        in_family = [False] * total_masks
        while len(family) < m:
            mask = random.randint(0, total_masks - 1)
            if not in_family[mask]:
                family.append(mask)
                in_family[mask] = True
        
        current_score = calc_score(family)
        best_family = family[:]
        best_score = current_score

        # Simulated Annealing
        T = 100.0
        cooling = 0.999
        # Estimate iterations per second to tune? No, just run until deadline.
        
        while time.time() < deadline:
            # Pick random index to swap
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Pick new mask
            new_mask = random.randint(0, total_masks - 1)
            attempts = 0
            while in_family[new_mask] and attempts < 100:
                new_mask = random.randint(0, total_masks - 1)
                attempts += 1
            if in_family[new_mask]:
                continue # Skip if collision too frequent (unlikely with 40/4096)

            # Calculate delta score
            delta = 0
            for j in range(m):
                if i == j:
                    continue
                other = family[j]
                if is_comp(old_mask, other) and not is_comp(new_mask, other):
                    delta -= 1
                elif not is_comp(old_mask, other) and is_comp(new_mask, other):
                    delta += 1
            
            # Acceptance criterion
            if delta >= 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                family[i] = new_mask
                in_family[old_mask] = False
                in_family[new_mask] = True
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            
            T *= cooling
            if T < 0.1:
                T = 0.1 # Keep some randomness to escape local minima

        # Convert best_family to list of lists of ints
        result = []
        for mask in best_family:
            result.append([b for b in range(n) if (mask >> b) & 1])
        
        return result
