import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_words = q ** n
    if total_words > 10000:
        # Fallback for larger instances, though task is fixed
        return [[0] * n]

    # Precompute indices and balls
    # For q=3, n=6, radius=1: ball size = 1 + 6*(3-1) = 13
    # We represent balls as lists of indices for fast counting updates
    
    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        rem = i
        for _ in range(n):
            w.append(rem % q)
            rem //= q
        words.append(tuple(w))
    
    # Precompute neighbors for each word (including itself)
    balls = []
    for i in range(total_words):
        w = words[i]
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        nw = list(w)
                        nw[pos] = new_val
                        # Calculate index
                        idx = 0
                        pow_q = 1
                        for k in range(n):
                            idx += nw[k] * pow_q
                            pow_q *= q
                        ball.append(idx)
        balls.append(ball)

    # 1. Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    # Precompute ball masks as bitsets for faster intersection? 
    # Or just use sets/arrays.
    # For 729 words, sets are fine.
    uncovered_set = set(range(total_words))
    
    # Greedy step
    while uncovered_set and time.time() < deadline - 20.0:
        best_gain = -1
        best_candidate = -1
        # Optimization: Only check candidates that cover at least one uncovered word
        # But checking all 729 is fast enough (729 * 13 ops)
        for c in range(total_words):
            gain = 0
            for w in balls[c]:
                if w in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
            elif gain == best_gain and gain > 0:
                # Tie breaking: random or deterministic
                if random.random() < 0.1:
                    best_candidate = c
                    
        if best_candidate == -1:
            break
            
        selected.append(best_candidate)
        for w in balls[best_candidate]:
            uncovered_set.discard(w)
            
    # Convert to list for SA
    selected = list(selected)
    
    # Initialize coverage counts
    counts = np.zeros(total_words, dtype=np.int32)
    for c in selected:
        for w in balls[c]:
            counts[w] += 1
            
    covered_count = np.sum(counts > 0)
    
    # 2. Simulated Annealing / Local Search
    # Objective: Minimize len(selected) + penalty * (total_words - covered_count)
    # We want to reach covered_count == total_words and min len.
    
    current_code = selected
    current_counts = counts
    current_covered = covered_count
    current_cost = len(current_code) + 100.0 * (total_words - current_covered)
    
    best_code = list(current_code)
    best_cost = current_cost
    best_covered = current_covered
    
    temperature = 50.0
    cooling_rate = 0.99
    
    # Helper to update counts and return new covered count
    def add_word(w):
        for x in balls[w]:
            current_counts[x] += 1
        # Update covered count efficiently? 
        # We can just recompute sum if needed, but tracking is better.
        # However, np.sum is fast for 729 elements.
        return np.sum(current_counts > 0)

    def remove_word(w):
        for x in balls[w]:
            current_counts[x] -= 1
        return np.sum(current_counts > 0)

    # SA Loop
    while time.time() < deadline - 5.0:
        if temperature < 0.5:
            break
            
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Remove a random word
            if len(current_code) > 0:
                idx = random.randint(0, len(current_code) - 1)
                w_remove = current_code[idx]
                current_code.pop(idx)
                
                # Apply move
                new_covered = remove_word(w_remove)
                new_cost = len(current_code) + 100.0 * (total_words - new_covered)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_cost = new_cost
                    current_covered = new_covered
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_code = list(current_code)
                        best_covered = current_covered
                else:
                    # Revert
                    current_code.append(w_remove)
                    # Re-add counts
                    for x in balls[w_remove]:
                        current_counts[x] += 1
                        # Note: current_covered is not updated on revert, 
                        # but we rely on current_cost/current_covered being consistent with current_code.
                        # Wait, remove_word updated counts. Revert must undo that.
                        # The 'new_covered' was calculated. If we revert, counts are restored.
                        # We don't need to recalc covered if we know it was current_covered before.
                        # But we modified counts. So restoring counts restores state.
                        # current_covered variable holds the value BEFORE the move.
                        # So we just keep current_covered as is.
                        pass
                        
        elif move_type < 0.6:
            # Add a random word
            w_add = random.randint(0, total_words - 1)
            if w_add not in current_code:
                current_code.append(w_add)
                
                new_covered = add_word(w_add)
                new_cost = len(current_code) + 100.0 * (total_words - new_covered)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_cost = new_cost
                    current_covered = new_covered
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_code = list(current_code)
                        best_covered = current_covered
                else:
                    # Revert
                    current_code.pop()
                    for x in balls[w_add]:
                        current_counts[x] -= 1
                        pass
                        
        else:
            # Swap
            if len(current_code) > 0:
                idx = random.randint(0, len(current_code) - 1)
                w_remove = current_code[idx]
                w_add = random.randint(0, total_words - 1)
                
                if w_add == w_remove:
                    continue
                    
                if w_add in current_code:
                    continue
                    
                # Perform swap
                current_code[idx] = w_add
                
                # Update counts
                for x in balls[w_remove]:
                    current_counts[x] -= 1
                for x in balls[w_add]:
                    current_counts[x] += 1
                    
                new_covered = np.sum(current_counts > 0)
                new_cost = len(current_code) + 100.0 * (total_words - new_covered)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_cost = new_cost
                    current_covered = new_covered
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_code = list(current_code)
                        best_covered = current_covered
                else:
                    # Revert
                    current_code[idx] = w_remove
                    for x in balls[w_add]:
                        current_counts[x] -= 1
                    for x in balls[w_remove]:
                        current_counts[x] += 1
                    pass
                    
        temperature *= cooling_rate

    # Final Pruning: Try to remove redundant words from best_code
    final_code = list(best_code)
    # Recalculate counts for final_code to be sure
    final_counts = np.zeros(total_words, dtype=np.int32)
    for c in final_code:
        for w in balls[c]:
            final_counts[w] += 1
            
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        new_final = []
        for idx, w in enumerate(final_code):
            # Try removing w
            for x in balls[w]:
                final_counts[x] -= 1
            
            if np.all(final_counts > 0):
                # w is redundant
                changed = True
            else:
                # w is necessary, keep it
                for x in balls[w]:
                    final_counts[x] += 1
                new_final.append(w)
                
        if changed:
            final_code = new_final
            # Recount for next iteration if we removed some
            # Actually, if we removed some, final_counts is already updated for the removed ones (decremented).
            # But for the kept ones, we incremented back.
            # So final_counts is correct for final_code.
            
    # Ensure validity
    if not np.all(final_counts > 0):
        # Fallback: use greedy result if SA messed up
        # But SA should maintain validity mostly.
        # If invalid, we must fix it.
        # Simple fix: add all uncovered words
        uncovered_idx = np.where(final_counts == 0)[0]
        for u in uncovered_idx:
            final_code.append(u)
            for x in balls[u]:
                final_counts[x] += 1
        # Prune again
        changed = True
        while changed:
            changed = False
            new_final = []
            for w in final_code:
                for x in balls[w]:
                    final_counts[x] -= 1
                if np.all(final_counts > 0):
                    changed = True
                else:
                    for x in balls[w]:
                        final_counts[x] += 1
                    new_final.append(w)
            if changed:
                final_code = new_final

    # Convert back to list of lists
    result = []
    for idx in final_code:
        result.append(list(words[idx]))
        
    return result
