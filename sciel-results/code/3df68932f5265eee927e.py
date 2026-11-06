import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for evaluator
    
    best_solution = None
    best_score = float('inf')
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    def evaluate(A):
        """
        Returns (score, is_valid, max_elem, missing_count, collision_count)
        Score is to be minimized.
        """
        if not A:
            return float('inf'), False, 0, 30, 0
            
        # Check distinctness of elements
        if len(set(A)) != len(A):
            return float('inf'), False, max(A), 30, 1000

        diffs = []
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d <= 0:
                    continue
                diffs.append(d)
        
        diff_set = set(diffs)
        
        # Sidon check: all positive differences must be distinct
        # If len(diffs) != len(diff_set), there are collisions
        collisions = len(diffs) - len(diff_set)
        
        # Coverage check
        missing = target_diffs - diff_set
        missing_count = len(missing)
        
        max_elem = max(A)
        
        # Penalty structure
        # 1. Must be Sidon (no collisions)
        # 2. Must cover all diffs
        # 3. Minimize max element
        
        if collisions > 0:
            # Heavy penalty for not being Sidon
            score = max_elem + 10000 * collisions + 100 * missing_count
            is_valid = False
        elif missing_count > 0:
            # Medium penalty for missing diffs
            score = max_elem + 1000 * missing_count
            is_valid = False
        else:
            # Valid solution
            score = max_elem
            is_valid = True
            
        return score, is_valid, max_elem, missing_count, collisions

    def run_sa(k, max_iter, time_budget):
        """
        Run Simulated Annealing for a set of size k.
        """
        # Initial random set
        # Ensure 0 is included
        A = [0] + [random.randint(1, 100) for _ in range(k - 1)]
        A = sorted(set(A))
        if len(A) < k:
             # Fill with randoms
             while len(A) < k:
                 A.append(random.randint(1, 100))
             A = sorted(set(A))
             # If still not k distinct, just pad with large numbers? 
             # Actually, let's just restart if this happens, which is rare.
             if len(A) != k:
                 return None, float('inf')

        current_score, current_valid, _, _, _ = evaluate(A)
        best_A = A.copy()
        best_score_local = current_score
        
        T = 50.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit - 5.0:
                break
            if time.time() > start_time + time_budget:
                break
                
            # Generate move
            idx = random.randint(0, k - 1)
            # New value range: around current max
            current_max = max(A)
            # Try to keep numbers small
            upper_bound = max(100, current_max + 20)
            new_val = random.randint(0, upper_bound)
            
            # Ensure distinctness
            if new_val in A and A[idx] != new_val:
                continue
                
            new_A = A.copy()
            new_A[idx] = new_val
            new_A = sorted(new_A)
            
            new_score, new_valid, _, _, _ = evaluate(new_A)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / max(T, 1e-9)):
                A = new_A
                current_score = new_score
                current_valid = new_valid
                
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_A = A.copy()
                    if current_valid:
                        # Found a valid solution, try to improve further
                        pass
            
            T *= cooling
            
        return best_A, best_score_local

    # Try different sizes
    # k=9 is the minimum size for 30 diffs (36 pairs)
    # k=10 gives more flexibility (45 pairs)
    for k in [9, 10, 11]:
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Run multiple restarts
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() - start_time > time_limit - 10.0:
                break
                
            A, score = run_sa(k, 2000, (time_limit - (time.time() - start_time)) / num_restarts / 2)
            
            if A is not None:
                # Re-evaluate to be sure
                s, valid, _, _, _ = evaluate(A)
                if valid and s < best_score:
                    best_score = s
                    best_solution = A

    if best_solution is None:
        # Fallback: Return a valid but possibly non-optimal set
        # Construct a trivial Sidon set that covers 1..30
        # e.g. [0, 1, 3, 7, 12, 18, 25, 33, 42, 52] ... 
        # Just use the greedy from seed as fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_solution
