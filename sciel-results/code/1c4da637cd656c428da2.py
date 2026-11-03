import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0  # seconds
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though task specifies 3,6,1
        # Simple greedy
        pass 

    # Specific solution for q=3, n=6, radius=1
    total_words = q ** n
    # Precompute neighbors for each word
    # word is an integer 0..728
    # convert to base-3 digits to find neighbors
    
    def to_base3(num, length):
        res = []
        for _ in range(length):
            res.append(num % 3)
            num //= 3
        return res

    def from_base3(digits):
        num = 0
        for i, d in enumerate(reversed(digits)):
            num += d * (3 ** i)
        return num

    neighbors = [[] for _ in range(total_words)]
    for w in range(total_words):
        digits = to_base3(w, n)
        # self
        neighbors[w].append(w)
        # neighbors at distance 1
        for i in range(n):
            orig = digits[i]
            for new_val in range(3):
                if new_val != orig:
                    new_digits = digits.copy()
                    new_digits[i] = new_val
                    new_word = from_base3(new_digits)
                    neighbors[w].append(new_word)
    
    # Initial code: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            # Construct word
            word_digits = h + list(tail)
            w = from_base3(word_digits)
            initial_codewords.append(w)
            
    # State: set of codewords, count of coverage for each word
    # We maintain a list of codewords and a coverage count array
    # To speed up removal, we check if all neighbors have count > 1
    
    best_set = set(initial_codewords)
    best_size = len(best_set)
    
    # Helper to calculate coverage counts for a set
    def get_coverage_counts(codeword_set):
        counts = [0] * total_words
        for c in codeword_set:
            for w in neighbors[c]:
                counts[w] += 1
        return counts

    current_set = set(initial_codewords)
    current_counts = get_coverage_counts(current_set)
    
    # Redundancy removal function
    def remove_redundants(c_set, c_counts):
        changed = True
        # List to allow modification during iteration? 
        # Better to iterate over a copy or list
        codewords_list = list(c_set)
        while changed:
            changed = False
            # Shuffle to avoid bias?
            random.shuffle(codewords_list)
            for c in codewords_list:
                if c not in c_set:
                    continue
                # Check if c is redundant
                is_redundant = True
                for w in neighbors[c]:
                    if c_counts[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    c_set.remove(c)
                    for w in neighbors[c]:
                        c_counts[w] -= 1
                    changed = True
        return c_set, c_counts

    # Initial redundancy removal
    current_set, current_counts = remove_redundants(current_set, current_counts)
    if len(current_set) < best_size:
        best_size = len(current_set)
        best_set = set(current_set)
        
    # Main search loop
    # Strategy: Add random word, remove redundants.
    # If size decreases, accept.
    # If size same, accept with small prob to wander.
    # If size increases, reject.
    
    while time.time() < deadline - 5:
        # Pick a random word
        w = random.randint(0, total_words - 1)
        
        # Tentatively add w
        if w in current_set:
            continue
            
        # Update counts
        for neighbor in neighbors[w]:
            current_counts[neighbor] += 1
        current_set.add(w)
        
        # Remove redundants
        new_set, new_counts = remove_redundants(current_set, current_counts)
        
        new_size = len(new_set)
        old_size = len(current_set) - 1 # size before adding w? No, current_set had w added.
        # Let's track size before adding
        # Actually, we just compare new_size with best_size and maybe current_size_before_add
        
        # We need to be careful with state management.
        # Let's snapshot the set size before adding.
        
        # Re-do logic for clarity in loop:
        # 1. Save current state
        # 2. Add w
        # 3. Remove redundants
        # 4. Compare
        
        # The code above modified current_set and current_counts in place.
        # This is efficient.
        
        if new_size < best_size:
            best_size = new_size
            best_set = set(new_set)
            # Keep current state as is (it is the new best)
            
        elif new_size == best_size:
            # Maybe accept to explore? 
            # If we accept, current_set becomes new_set.
            # This allows moving to another local minimum of the same size.
            # Let's accept with prob 0.5
            if random.random() < 0.5:
                # State is already updated to new_set/new_counts
                pass
            else:
                # Rollback? 
                # Rolling back is expensive if we don't save state.
                # To simplify, we only accept if size decreases.
                # If we want to wander, we need to save state.
                # Given time budget, let's just try to find smaller sets.
                # If we get stuck at 73, we stay at 73.
                # If we find 72, we stay at 72.
                pass
                
        else:
            # Size increased or stayed same (if we rejected same size)
            # We should rollback to previous state if we rejected.
            # Since we modified in place, we need to rollback if we don't accept.
            # This is tricky.
            
            # Let's use a cleaner approach:
            # Keep `current_set` and `current_counts` as the "committed" state.
            # For each step, create a tentative copy? No, too slow.
            # Or, only commit if accepted.
            
            # Let's restart the loop logic with proper rollback.
            break # Break to rewrite the loop structure properly below

    # Rewriting the search loop with proper state management
    
    # Reset to best found so far
    current_set = set(best_set)
    current_counts = get_coverage_counts(current_set)
    
    while time.time() < deadline - 2:
        w = random.randint(0, total_words - 1)
        if w in current_set:
            continue
            
        # Save state for rollback
        # Since we only add 1 word and remove some, the change is localized.
        # We can save the list of removed words and the added word.
        
        # 1. Add w
        current_set.add(w)
        for neighbor in neighbors[w]:
            current_counts[neighbor] += 1
            
        # 2. Remove redundants
        removed = []
        changed = True
        codewords_list = list(current_set)
        while changed:
            changed = False
            random.shuffle(codewords_list)
            for c in codewords_list:
                if c not in current_set:
                    continue
                is_redundant = True
                for neighbor in neighbors[c]:
                    if current_counts[neighbor] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    current_set.remove(c)
                    for neighbor in neighbors[c]:
                        current_counts[neighbor] -= 1
                    removed.append(c)
                    changed = True
                    
        new_size = len(current_set)
        
        # Decision
        if new_size < best_size:
            best_size = new_size
            best_set = set(current_set)
            # Accept state
            
        elif new_size == best_size:
            # Accept with prob 0.2 to allow wandering among same-size minima
            if random.random() < 0.2:
                pass # Accept
            else:
                # Rollback
                current_set.discard(w)
                for neighbor in neighbors[w]:
                    current_counts[neighbor] -= 1
                for c in removed:
                    current_set.add(c)
                    for neighbor in neighbors[c]:
                        current_counts[neighbor] += 1
        else:
            # Rollback
            current_set.discard(w)
            for neighbor in neighbors[w]:
                current_counts[neighbor] -= 1
            for c in removed:
                current_set.add(c)
                for neighbor in neighbors[c]:
                    current_counts[neighbor] += 1

    # Convert best_set back to list of lists
    result = []
    for w in best_set:
        digits = to_base3(w, n)
        result.append(digits)
        
    return result
