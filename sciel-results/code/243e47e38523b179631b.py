import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 256
    
    def count_3ap(A):
        """Count number of 3-term APs in set A."""
        count = 0
        A_set = set(A)
        for b in A_set:
            for d in range(1, min(b - 1, N - b) + 1):
                if (b - d) in A_set and (b + d) in A_set:
                    count += 1
        return count
    
    def is_4ap_free(A):
        """Check if set A has no 4-term AP."""
        A_set = set(A)
        for a in A_set:
            for d in range(1, (N - a) // 3 + 1):
                if (a + d) in A_set and (a + 2*d) in A_set and (a + 3*d) in A_set:
                    return False
        return True
    
    def greedy_construct():
        """Greedy construction: add elements that maximize 3-AP gain while maintaining 4-AP-free."""
        A = set()
        remaining = set(range(1, N + 1))
        
        def is_valid_add(x, A):
            A_set = set(A)
            # x is 1st in 4-AP
            for d in range(1, (N - x) // 3 + 1):
                if (x + d) in A_set and (x + 2*d) in A_set and (x + 3*d) in A_set:
                    return False
            # x is 2nd in 4-AP
            for d in range(1, min(x - 1, (N - x) // 2) + 1):
                if (x - d) in A_set and (x + d) in A_set and (x + 2*d) in A_set:
                    return False
            # x is 3rd in 4-AP
            for d in range(1, min(x // 2, N - x) + 1):
                if (x - 2*d) in A_set and (x - d) in A_set and (x + d) in A_set:
                    return False
            # x is 4th in 4-AP
            for d in range(1, (x - 1) // 3 + 1):
                if (x - 3*d) in A_set and (x - 2*d) in A_set and (x - d) in A_set:
                    return False
            return True
        
        def count_3ap_gain(x, A):
            A_set = set(A)
            gain = 0
            # x is 1st in 3-AP
            for d in range(1, (N - x) // 2 + 1):
                if (x + d) in A_set and (x + 2*d) in A_set:
                    gain += 1
            # x is 2nd in 3-AP
            for d in range(1, min(x - 1, N - x) + 1):
                if (x - d) in A_set and (x + d) in A_set:
                    gain += 1
            # x is 3rd in 3-AP
            for d in range(1, x // 2 + 1):
                if (x - 2*d) in A_set and (x - d) in A_set:
                    gain += 1
            return gain
        
        while remaining and time.time() < deadline - 5:
            best_x = -1
            best_score = -1
            for x in remaining:
                if is_valid_add(x, A):
                    score = count_3ap_gain(x, A)
                    if score > best_score:
                        best_score = score
                        best_x = x
            if best_x == -1:
                break
            A.add(best_x)
            remaining.remove(best_x)
        
        return A
    
    def simulated_annealing(initial_A, max_time=10.0):
        """Simulated annealing to improve 3-AP count while maintaining 4-AP-free."""
        A = set(initial_A)
        current_score = count_3ap(A)
        best_A = set(A)
        best_score = current_score
        
        start = time.time()
        T = 100.0
        cooling = 0.995
        
        while time.time() < start + max_time:
            if not A:
                break
            
            # Generate a random move: remove one element and add another
            if len(A) < 2:
                # Try to add a random element
                candidates = [x for x in range(1, N + 1) if x not in A]
                if not candidates:
                    break
                x_add = random.choice(candidates)
                new_A = A | {x_add}
                if not is_4ap_free(new_A):
                    continue
                new_score = count_3ap(new_A)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    A = new_A
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
            else:
                # Remove a random element
                x_remove = random.choice(list(A))
                A_without = A - {x_remove}
                
                # Try to add a random element
                candidates = [x for x in range(1, N + 1) if x not in A_without]
                if not candidates:
                    A = A_without
                    current_score = count_3ap(A)
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
                    continue
                
                x_add = random.choice(candidates)
                new_A = A_without | {x_add}
                
                if not is_4ap_free(new_A):
                    # Try just removing
                    if not is_4ap_free(A_without):
                        continue
                    new_score = count_3ap(A_without)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        A = A_without
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_A = set(A)
                    continue
                
                new_score = count_3ap(new_A)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    A = new_A
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
            
            T *= cooling
        
        return best_A, best_score
    
    # Run multiple restarts with different strategies
    best_overall_A = set()
    best_overall_score = 0
    
    # Strategy 1: Pure greedy
    A1 = greedy_construct()
    score1 = count_3ap(A1)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_A = set(A1)
    
    # Strategy 2: Greedy + SA
    if time.time() < deadline - 15:
        A2, score2 = simulated_annealing(A1, max_time=10.0)
        if score2 > best_overall_score:
            best_overall_score = score2
            best_overall_A = set(A2)
    
    # Strategy 3: Random initial + SA
    if time.time() < deadline - 15:
        random.seed(42)
        A3_init = set(random.sample(range(1, N + 1), min(64, N)))
        # Ensure 4-AP-free by removing problematic elements
        A3 = set()
        for x in sorted(A3_init):
            if is_4ap_free(A3 | {x}):
                A3.add(x)
        A3, score3 = simulated_annealing(A3, max_time=10.0)
        if score3 > best_overall_score:
            best_overall_score = score3
            best_overall_A = set(A3)
    
    # Strategy 4: Another greedy with different tie-breaking
    if time.time() < deadline - 15:
        random.seed(123)
        A4 = set()
        remaining4 = set(range(1, N + 1))
        
        def is_valid_add4(x, A):
            A_set = set(A)
            for d in range(1, (N - x) // 3 + 1):
                if (x + d) in A_set and (x + 2*d) in A_set and (x + 3*d) in A_set:
                    return False
            for d in range(1, min(x - 1, (N - x) // 2) + 1):
                if (x - d) in A_set and (x + d) in A_set and (x + 2*d) in A_set:
                    return False
            for d in range(1, min(x // 2, N - x) + 1):
                if (x - 2*d) in A_set and (x - d) in A_set and (x + d) in A_set:
                    return False
            for d in range(1, (x - 1) // 3 + 1):
                if (x - 3*d) in A_set and (x - 2*d) in A_set and (x - d) in A_set:
                    return False
            return True
        
        def count_3ap_gain4(x, A):
            A_set = set(A)
            gain = 0
            for d in range(1, (N - x) // 2 + 1):
                if (x + d) in A_set and (x + 2*d) in A_set:
                    gain += 1
            for d in range(1, min(x - 1, N - x) + 1):
                if (x - d) in A_set and (x + d) in A_set:
                    gain += 1
            for d in range(1, x // 2 + 1):
                if (x - 2*d) in A_set and (x - d) in A_set:
                    gain += 1
            return gain
        
        while remaining4 and time.time() < deadline - 5:
            best_x = -1
            best_score = -1
            for x in remaining4:
                if is_valid_add4(x, A4):
                    score = count_3ap_gain4(x, A4)
                    # Add some randomness to tie-breaking
                    if score > best_score or (score == best_score and random.random() < 0.3):
                        best_score = score
                        best_x = x
            if best_x == -1:
                break
            A4.add(best_x)
            remaining4.remove(best_x)
        
        A4, score4 = simulated_annealing(A4, max_time=8.0)
        if score4 > best_overall_score:
            best_overall_score = score4
            best_overall_A = set(A4)
    
    # Final local search: try all single swaps
    if time.time() < deadline - 3 and best_overall_A:
        current_A = set(best_overall_A)
        current_score = count_3ap(current_A)
        
        for _ in range(3):  # A few passes
            improved = False
            for a in list(current_A):
                for b in range(1, N + 1):
                    if b in current_A:
                        continue
                    new_A = (current_A - {a}) | {b}
                    if is_4ap_free(new_A):
                        new_score = count_3ap(new_A)
                        if new_score > current_score:
                            current_A = new_A
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_A = current_A
    
    return sorted(best_overall_A)
