import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for a set of integers A."""
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(A[i] + A[j])
                prod_set.add(A[i] * A[j])
        return max(len(sum_set), len(prod_set))
    
    def local_search(A, max_iter=5000):
        """Perform local search with multiple mutation strategies."""
        A = A[:]
        current_score = evaluate(A)
        best_score = current_score
        best_A = A[:]
        
        for _ in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            # Try different mutation strategies
            strategy = random.choice(['replace', 'swap', 'perturb'])
            
            if strategy == 'replace':
                new_A = A[:]
                idx = random.randint(0, 19)
                # Try to find a value that minimizes marginal increase
                best_val = None
                best_delta = float('inf')
                for _ in range(20):  # Try 20 random candidates
                    new_val = random.randint(1, 10000)
                    if new_val in new_A:
                        continue
                    new_A[idx] = new_val
                    delta = evaluate(new_A) - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_val = new_val
                if best_val is not None:
                    new_A[idx] = best_val
                    new_score = evaluate(new_A)
                else:
                    continue
            elif strategy == 'swap':
                new_A = A[:]
                i, j = random.sample(range(20), 2)
                new_A[i], new_A[j] = new_A[j], new_A[i]
                new_score = evaluate(new_A)
            else:  # perturb
                new_A = A[:]
                idx = random.randint(0, 19)
                new_A[idx] = max(1, min(10000, new_A[idx] + random.randint(-10, 10)))
                if new_A[idx] in new_A:
                    continue
                new_score = evaluate(new_A)
            
            delta = new_score - current_score
            if delta < 0:
                A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
        
        return best_A, best_score
    
    # Initial candidates: multiple structured sets
    candidates = [
        list(range(1, 21)),  # AP
        [2**i for i in range(20)],  # GP
        [i*5 for i in range(1, 21)],  # AP with step 5
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 100, 200, 300, 400, 500, 600, 700],
    ]
    
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(set(cand)) != 20:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_A = cand[:]
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = evaluate(best_A)
    
    # Phase 1: Simulated Annealing with multiple mutations
    T = 50.0
    cooling_rate = 0.9998
    current_A = best_A[:]
    current_score = best_score
    
    max_iterations = 200000
    for i in range(max_iterations):
        if time.time() - start_time > timeout - 10:  # Reserve 10s for local search
            break
        
        # Generate new candidate with diverse strategies
        new_A = current_A[:]
        strategy = random.choice(['replace', 'swap', 'perturb', 'insert'])
        
        if strategy == 'replace':
            idx = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            attempts = 0
            while new_val in new_A and attempts < 10:
                new_val = random.randint(1, 10000)
                attempts += 1
            if new_val in new_A:
                continue
            new_A[idx] = new_val
        elif strategy == 'swap':
            i, j = random.sample(range(20), 2)
            new_A[i], new_A[j] = new_A[j], new_A[i]
        elif strategy == 'perturb':
            idx = random.randint(0, 19)
            new_A[idx] = max(1, min(10000, new_A[idx] + random.randint(-20, 20)))
            if new_A[idx] in new_A:
                continue
        else:  # insert: replace with value near existing ones
            idx = random.randint(0, 19)
            base = random.choice(current_A)
            new_val = max(1, min(10000, base + random.randint(-50, 50)))
            attempts = 0
            while new_val in new_A and attempts < 10:
                new_val = max(1, min(10000, base + random.randint(-100, 100)))
                attempts += 1
            if new_val in new_A:
                continue
            new_A[idx] = new_val
        
        new_score = evaluate(new_A)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_A = new_A
                    current_score = new_score
        
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
    
    # Phase 2: Local search polish on best solution
    if time.time() - start_time < timeout - 5:
        polished_A, polished_score = local_search(best_A, max_iter=10000)
        if polished_score < best_score:
            best_score = polished_score
            best_A = polished_A
    
    return best_A
