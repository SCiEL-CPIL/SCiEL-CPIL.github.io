import time
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 90s kill
    
    def run_greedy(order):
        """
        Greedily build a set from 'order' such that for any two subsets of different
        cardinalities, their sums are distinct.
        
        We maintain S[k] = set of sums of all k-element subsets of the current set.
        When adding x, we check if (S[j-1] + x) intersects S[i] for any i != j.
        """
        S = [set() for _ in range(61)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            valid = True
            # Check validity: for each j from 1 to current_size+1
            # new sums are S[j-1] + x
            # These must not be in S[i] for any i != j
            # Note: S[j] (old) is disjoint from S[i] for i!=j already.
            # We only need to check (S[j-1] + x) vs S[i] for i != j.
            
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                # Check (S[j-1] + x) against S[i] for i != j
                # Optimization: pre-compute the union of all S[i] for i != j?
                # No, S[i] sizes vary. 
                # But we can check: is there any s in S[j-1] such that s+x in S[i] for some i != j?
                
                # To speed up, we can create a set of all forbidden values for this j.
                # Forbidden = Union of S[i] for i in [0, current_size] where i != j.
                # If (S[j-1] + x) intersects Forbidden, it's invalid.
                
                # Building the forbidden set every time is O(total_sums).
                # Total sums is sum of |S[i]|.
                # For small current_size, this is fine.
                
                # Alternative: iterate s in S[j-1], check s+x in each S[i].
                # This is sum_{s in S[j-1]} (current_size+1) checks.
                # Since |S[j-1]| can be large, this might be slow.
                
                # Let's try the forbidden set approach if current_size is small.
                # Actually, let's just iterate. It's simpler and likely fast enough for n<20.
                
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                # Update S
                # S[k] = S[k] U (S[k-1] + x) for k from current_size down to 1
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
                # Note: S[0] remains {0}
        
        return A

    # Deterministic run first
    best_set = run_greedy(list(range(1, 257)))
    
    # Randomized runs
    # Use a fixed seed for reproducibility in case of tie-breaking? No, random is fine.
    # We want to explore different orderings.
    
    # Strategy: 
    # 1. Random shuffles.
    # 2. Maybe try sorting by some heuristic? e.g., larger numbers first?
    #    Larger numbers create larger gaps, potentially reducing collisions?
    #    Or smaller numbers first to fill in sums?
    #    The current greedy tries 1..256.
    #    Let's try reverse order 256..1.
    
    reverse_order = list(range(256, 0, -1))
    candidate = run_greedy(reverse_order)
    if len(candidate) > len(best_set):
        best_set = candidate
    
    # Random shuffles
    while time.time() - start_time < deadline:
        order = list(range(1, 257))
        random.shuffle(order)
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    return [int(x) for x in best_set]
