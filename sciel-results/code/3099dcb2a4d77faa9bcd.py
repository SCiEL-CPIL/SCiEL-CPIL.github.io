import random
import time

def construct():
    n = 128
    target_start = 128
    target_end = 256
    
    # Initial set: Seed
    A = set(range(64, 129))
    
    # Helper to update counts
    # counts[s] = number of pairs (a,b) in A such that a+b=s, a<=b
    # We maintain this incrementally.
    
    def build_counts(A_set):
        counts = [0] * (2 * n + 1)
        for a in A_set:
            for b in A_set:
                if a <= b:
                    counts[a + b] += 1
        return counts

    # Initial counts
    counts = build_counts(A)
    
    def is_valid(counts):
        for s in range(target_start, target_end + 1):
            if counts[s] == 0:
                return False
        return True

    def get_peak(counts):
        return max(counts[target_start:target_end + 1])

    current_peak = get_peak(counts)
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s buffer
    
    # Phase 1: Aggressive Removal
    # Try to remove elements to reduce set size and potentially peak multiplicity
    # We prioritize removing elements that contribute to high-multiplicity sums
    
    while time.time() - start_time < timeout:
        improved = False
        # Shuffle to avoid bias
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() - start_time > timeout:
                break
            
            # Tentatively remove a
            A.remove(a)
            
            # Update counts
            # Subtract pairs involving a
            # Pairs (a, b) where b in A (current, without a)
            # And pair (a, a)
            for b in A:
                counts[a + b] -= 1
            counts[2 * a] -= 1
            
            if is_valid(counts):
                # Check if peak improved or stayed same
                new_peak = get_peak(counts)
                if new_peak <= current_peak:
                    current_peak = new_peak
                    improved = True
                    # Keep removal
                    continue
                else:
                    # Restore
                    A.add(a)
                    for b in A:
                        counts[a + b] += 1
                    counts[2 * a] += 1
            else:
                # Restore
                A.add(a)
                for b in A:
                    counts[a + b] += 1
                counts[2 * a] += 1
        
        if not improved:
            # If no removals were made, try swaps
            # Swap a in A with b not in A
            swapped = False
            for _ in range(50): # Try a few swaps
                if time.time() - start_time > timeout:
                    break
                if len(A) < 1:
                    break
                a = random.choice(list(A))
                candidates = [x for x in range(n + 1) if x not in A]
                if not candidates:
                    break
                b = random.choice(candidates)
                
                # Remove a
                A.remove(a)
                for c in A:
                    counts[a + c] -= 1
                counts[2 * a] -= 1
                
                # Add b
                for c in A:
                    counts[b + c] += 1
                counts[2 * b] += 1
                A.add(b)
                
                if is_valid(counts):
                    new_peak = get_peak(counts)
                    if new_peak < current_peak:
                        current_peak = new_peak
                        swapped = True
                        # Keep swap
                        break # Restart swap search with new state
                    else:
                        # Restore
                        A.remove(b)
                        for c in A:
                            counts[b + c] -= 1
                        counts[2 * b] -= 1
                        
                        A.add(a)
                        for c in A:
                            counts[a + c] += 1
                        counts[2 * a] += 1
                else:
                    # Restore
                    A.remove(b)
                    for c in A:
                        counts[b + c] -= 1
                    counts[2 * b] -= 1
                    
                    A.add(a)
                    for c in A:
                        counts[a + c] += 1
                    counts[2 * a] += 1
            
            if not swapped and not improved:
                # Stuck in local minimum for removal and simple swaps
                # Try a more aggressive random restart or perturbation?
                # Or just break if we've been stuck for a while?
                # For now, just break to return the best found.
                # We can add a small random perturbation: add a random element, then try to remove others.
                # But time is limited. Let's just break.
                break
                
    return list(A)

# EVOLVE-BLOCK-END
