import random
import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin from 120s limit
    
    # Initial Set: Greedy Sidon Construction
    # This provides a strong baseline with high |U-U|
    U = [0]
    used_diffs = set()
    used_diffs.add(0)
    
    # Greedy construction of Sidon set
    while len(U) < 50: # Start with a decent size
        current_max = max(U)
        candidate = current_max + 1
        valid = True
        # Check if candidate creates duplicate differences
        # We only need to check positive differences to be efficient
        # diff = candidate - u for u in U
        # If any diff is in used_diffs (positive part), it's invalid
        # Actually, we need to check if candidate - u == v - w or u - v
        # Simpler: maintain set of all differences
        new_diffs = set()
        for u in U:
            d = candidate - u
            if d in used_diffs or -d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            U.append(candidate)
            for d in new_diffs:
                used_diffs.add(d)
                used_diffs.add(-d)
        
        if time.time() > deadline:
            break

    # Ensure 0 is in U and it's a list of ints
    U = sorted(U)
    
    def calculate_score(U):
        if not U or 0 not in U:
            return 0.0
        n = len(U)
        if n < 2:
            return 0.0
        max_u = max(U)
        
        # Calculate |U-U| and |U+U|
        # For efficiency, use sets
        diffs = set()
        sums = set()
        
        # Optimization: Only compute upper triangle and add symmetric
        # But sets handle duplicates automatically
        # O(n^2) is fine for n ~ 100-500
        for i in range(n):
            for j in range(i, n):
                d = U[i] - U[j]
                s = U[i] + U[j]
                diffs.add(d)
                sums.add(s)
                if i != j:
                    diffs.add(-d)
                    # sums is symmetric, so no need to add again for (j,i) as it's the same sum
        
        # Wait, the loop above adds d and -d.
        # If i=j, d=0, -d=0.
        # If i!=j, we add d and -d.
        # This correctly populates the difference set.
        # Sum set is just s.
        
        size_diff = len(diffs)
        size_sum = len(sums)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            # Log of ratio <= 0, score <= 1. 
            # We want to maximize, so if we can't beat 1, we return 1 (or slightly above if possible)
            # But the formula is 1 + log(ratio)/log(2max+1)
            # If ratio < 1, score < 1.
            # The prompt implies we should aim for > 1.
            pass
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        score = 1.0 + math.log(ratio) / denom
        return score

    best_score = calculate_score(U)
    best_U = U[:]

    # Simulated Annealing
    current_U = U[:]
    current_score = best_score
    temperature = 1.0
    cooling_rate = 0.999
    
    # We will try to improve the score by modifying the set
    # Moves: 
    # 1. Remove an element (reduces n, might improve ratio if element was "bad")
    # 2. Add an element (increases n, might improve ratio if element is "good")
    # 3. Swap an element
    
    while time.time() < deadline:
        if len(current_U) < 2:
            break
            
        # Choose a move
        move_type = random.choice(['remove', 'add', 'swap', 'shift'])
        
        new_U = current_U[:]
        
        if move_type == 'remove' and len(new_U) > 1:
            # Remove a random element (not 0)
            idx = random.randint(1, len(new_U) - 1)
            new_U.pop(idx)
            
        elif move_type == 'add':
            # Add a random element near the end or in a gap
            if new_U:
                max_val = max(new_U)
                # Try adding near max_val or in a random gap
                if random.random() < 0.5:
                    candidate = max_val + random.randint(1, 10)
                else:
                    # Random position
                    if len(new_U) > 1:
                        i = random.randint(0, len(new_U) - 2)
                        candidate = random.randint(new_U[i] + 1, new_U[i+1] - 1)
                        if candidate <= new_U[i] or candidate >= new_U[i+1]:
                            candidate = max_val + 1
                    else:
                        candidate = max_val + 1
                if candidate not in new_U:
                    new_U.append(candidate)
                    new_U.sort()
                else:
                    continue # Skip if duplicate
            
        elif move_type == 'swap':
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                max_val = max(new_U)
                # Try a new value
                candidate = old_val + random.randint(-10, 10)
                if candidate < 0 or candidate in new_U:
                    candidate = max_val + random.randint(1, 5)
                if candidate not in new_U:
                    new_U[idx] = candidate
                    new_U.sort()
                else:
                    continue
                    
        elif move_type == 'shift':
            # Shift all elements except 0 by a small amount? 
            # This changes max_val significantly.
            # Better: Shift a subset? 
            # Let's just try shifting the largest element
            if len(new_U) > 1:
                idx = len(new_U) - 1
                old_val = new_U[idx]
                candidate = old_val + random.randint(-5, 5)
                if candidate < new_U[idx-1] + 1 or candidate in new_U:
                    candidate = old_val + 1
                if candidate not in new_U:
                    new_U[idx] = candidate
                    new_U.sort()
                else:
                    continue
        else:
            continue
            
        if len(new_U) < 2:
            continue
            
        new_score = calculate_score(new_U)
        
        # Acceptance Criterion
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = new_U[:]
        else:
            # Metropolis acceptance
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / max(temperature, 1e-9))
            
            if accept:
                current_U = new_U
                current_score = new_score
        
        temperature *= cooling_rate

    return best_U

# EVOLVE-BLOCK-END
