import time
import random

def construct():
    N = 256
    deadline = time.time() + 50  # 50s safety margin
    
    # Precompute weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i

    def is_valid_addition(S, x):
        """Check if adding x to S keeps it 3-AP free."""
        # A 3-AP {a, b, c} exists in S U {x} iff:
        # 1. x is the middle: a + c = 2x for some a, c in S
        # 2. x is an endpoint: a + b = 2x is not the form, rather x + c = 2a or x + a = 2c
        # Generally, for any a in S, check if 2x - a is in S (x is mid or endpoint relative to a)
        # AND if 2a - x is in S (a is mid or endpoint relative to x)
        
        # Actually, if {a, b, x} is a 3-AP, then either:
        # 2x = a + b  => b = 2x - a
        # 2a = x + b  => b = 2a - x
        # 2b = x + a  => a = 2b - x (which is covered by iterating b as 'a' in the loop above if we just check 2*mid - end)
        
        # So checking: for any a in S, is (2*x - a) in S? OR is (2*a - x) in S?
        # Note: 2*x - a == a implies x==a, but x not in S, so 2*x-a != a.
        # Also 2*x - a could be out of bounds [1, N], but `in S` handles that.
        
        for a in S:
            if (2 * x - a) in S:
                return False
            if (2 * a - x) in S:
                return False
        return True

    def score(S):
        return sum(weights[x] for x in S)

    # Initial Greedy Solution
    S = set()
    for i in range(1, N + 1):
        if is_valid_addition(S, i):
            S.add(i)
    
    best_S = S.copy()
    best_score = score(S)
    
    # Local Search with Random Restarts
    # Strategy: 
    # 1. Try to add smallest available number.
    # 2. If blocked, try to swap a large number in S with a small number not in S.
    # 3. If stuck, perturb (remove random elements) and try to refill.
    
    while time.time() < deadline:
        improved = False
        
        # Try to add any available number, preferring small ones (high weight)
        # We iterate through candidates in random order to escape local minima, 
        # but bias towards small numbers.
        
        # Generate list of candidates not in S
        candidates = [i for i in range(1, N + 1) if i not in S]
        
        # Sort candidates by weight (descending) i.e., value (ascending)
        # But to allow exploration, we can shuffle slightly or just iterate.
        # Let's try adding the best possible candidate first.
        candidates.sort(key=lambda x: -weights[x])
        
        added = False
        for x in candidates:
            if is_valid_addition(S, x):
                S.add(x)
                added = True
                # Update best if improved
                current_score = score(S)
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                break
        
        if not added:
            # No single addition possible. Try swaps.
            # Remove a low-value element y from S and add a high-value element x not in S.
            # We want to maximize weight, so remove large y (small weight), add small x (large weight).
            
            # Sort S by weight ascending (largest numbers first, lowest weight)
            sorted_S = sorted(S, key=lambda x: weights[x])
            # Sort candidates by weight descending (smallest numbers first, highest weight)
            sorted_candidates = sorted([i for i in range(1, N + 1) if i not in S], key=lambda x: -weights[x])
            
            swapped = False
            for y in sorted_S:
                if swapped: break
                for x in sorted_candidates:
                    if x == y: continue
                    # Check if S - {y} + {x} is valid
                    # This is slightly more expensive.
                    # We can temporarily remove y, check x, then put y back if fails.
                    S.remove(y)
                    if is_valid_addition(S, x):
                        S.add(x)
                        swapped = True
                        current_score = score(S)
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        break
                    else:
                        S.add(y) # Restore y
            
            if not swapped:
                # Stuck. Perturb.
                # Remove 1 to 3 random elements
                num_remove = random.randint(1, 3)
                if len(S) > num_remove:
                    to_remove = random.sample(list(S), num_remove)
                    for r in to_remove:
                        S.remove(r)
                    
                    # Try to refill greedily
                    # Re-run the addition loop a few times
                    for _ in range(5):
                        candidates = [i for i in range(1, N + 1) if i not in S]
                        candidates.sort(key=lambda x: -weights[x])
                        added = False
                        for x in candidates:
                            if is_valid_addition(S, x):
                                S.add(x)
                                added = True
                                break
                        if not added:
                            break
                    current_score = score(S)
                    if current_score > best_score:
                        best_score = current_score
                        best_S = S.copy()
                
                # If still stuck and score is bad, maybe restart completely?
                # Or just continue perturbation.
                # To avoid infinite loops on same bad state, we can randomize the greedy init occasionally.
                # But let's stick to perturbation for now.
                # If we've been stuck for a while, do a hard restart.
                # Simple heuristic: if no improvement in last 1000 checks, restart.
                # For simplicity, just keep perturbing.
        
        # Small sleep to avoid 100% CPU? No, Python GIL handles it, but deadline is wall clock.
        # The loop is tight.

    # Return the best solution found
    # Ensure it's a list
    return list(best_S)
